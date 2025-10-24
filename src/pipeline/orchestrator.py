#!/usr/bin/env python3
"""
Pipeline Orchestrator - Frigate-Inspired Main System Controller

This module acts as the central coordinator for the doorbell security pipeline,
similar to Frigate's main process. It manages all pipeline stages, worker pools,
and inter-process communication.
"""

import sys
import time
import signal
import logging
import threading
from typing import Dict, Any, Optional, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.communication.message_bus import MessageBus
from src.communication.events import PipelineEvent, EventType
from src.communication.queues import QueueManager
from src.pipeline.frame_capture import FrameCaptureWorker
from src.pipeline.motion_detector import MotionDetectionWorker
from src.pipeline.face_detector import FaceDetectionWorker
from src.pipeline.face_recognizer import FaceRecognitionWorker
from src.pipeline.event_processor import EventProcessor
from src.storage.event_database import EventDatabase
from src.hardware.camera_handler import CameraHandler
from src.hardware.gpio_handler import GPIOHandler
from config.pipeline_config import PipelineConfig
from config.logging_config import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """Represents a single pipeline stage with its worker and configuration."""
    name: str
    worker: Any
    thread: Optional[threading.Thread] = None
    process: Optional[Any] = None
    enabled: bool = True
    critical: bool = False


class PipelineOrchestrator:
    """
    Main pipeline orchestrator inspired by Frigate NVR architecture.
    
    Manages the complete processing pipeline from GPIO events through
    face recognition to event enrichment and notifications.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the pipeline orchestrator."""
        self.config = PipelineConfig(config or {})
        self.message_bus = MessageBus()
        self.queue_manager = QueueManager()
        self.event_database = EventDatabase()
        
        # Pipeline state
        self.running = False
        self.shutdown_event = threading.Event()
        self.stages: List[PipelineStage] = []
        
        # Worker pools
        self.detection_executor = None
        self.recognition_executor = None
        
        # Hardware components
        self.camera_handler = None
        self.gpio_handler = None
        
        # Initialize components
        self._initialize_hardware()
        self._initialize_pipeline_stages()
        self._setup_signal_handlers()
        
        logger.info("Pipeline orchestrator initialized")
    
    def _initialize_hardware(self) -> None:
        """Initialize hardware components with fallback to mocks."""
        try:
            self.camera_handler = CameraHandler.create()
            self.gpio_handler = GPIOHandler()
            logger.info("Hardware components initialized")
        except Exception as e:
            logger.error(f"Hardware initialization failed: {e}")
            raise
    
    def _initialize_pipeline_stages(self) -> None:
        """Initialize all pipeline stages with proper dependencies."""
        try:
            # Stage 1: Frame Capture (triggered by GPIO events)
            frame_capture = FrameCaptureWorker(
                camera_handler=self.camera_handler,
                message_bus=self.message_bus,
                config=self.config.frame_capture
            )
            self.stages.append(PipelineStage(
                name="frame_capture",
                worker=frame_capture,
                critical=True
            ))
            
            # Stage 2: Motion Detection (optional for performance)
            if self.config.motion_detection.enabled:
                motion_detector = MotionDetectionWorker(
                    message_bus=self.message_bus,
                    config=self.config.motion_detection
                )
                self.stages.append(PipelineStage(
                    name="motion_detection",
                    worker=motion_detector,
                    enabled=self.config.motion_detection.enabled
                ))
            
            # Stage 3: Face Detection (multi-process worker pool)
            face_detector = FaceDetectionWorker(
                message_bus=self.message_bus,
                config=self.config.face_detection
            )
            self.stages.append(PipelineStage(
                name="face_detection",
                worker=face_detector,
                critical=True
            ))
            
            # Stage 4: Face Recognition (multi-process worker pool)
            face_recognizer = FaceRecognitionWorker(
                message_bus=self.message_bus,
                config=self.config.face_recognition
            )
            self.stages.append(PipelineStage(
                name="face_recognition",
                worker=face_recognizer,
                critical=True
            ))
            
            # Stage 5: Event Processing and Enrichment
            event_processor = EventProcessor(
                message_bus=self.message_bus,
                event_database=self.event_database,
                config=self.config.event_processing
            )
            self.stages.append(PipelineStage(
                name="event_processing",
                worker=event_processor,
                critical=True
            ))
            
            logger.info(f"Initialized {len(self.stages)} pipeline stages")
            
        except Exception as e:
            logger.error(f"Pipeline stage initialization failed: {e}")
            raise
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start(self) -> None:
        """Start the complete pipeline with all stages."""
        if self.running:
            logger.warning("Pipeline already running")
            return
        
        try:
            logger.info("Starting doorbell security pipeline...")
            
            # Initialize worker pools
            self._initialize_worker_pools()
            
            # Start message bus
            self.message_bus.start()
            
            # Start queue manager
            self.queue_manager.start()
            
            # Start event database
            self.event_database.start()
            
            # Start all pipeline stages
            self._start_pipeline_stages()
            
            # Setup GPIO trigger
            self._setup_gpio_trigger()
            
            # Set running state
            self.running = True
            
            logger.info("✅ Pipeline started successfully - system is monitoring")
            
            # Main monitoring loop
            self._monitoring_loop()
            
        except Exception as e:
            logger.error(f"Pipeline startup failed: {e}")
            self.shutdown()
            raise
    
    def _initialize_worker_pools(self) -> None:
        """Initialize worker pools for CPU-intensive tasks."""
        # Face detection worker pool (can use multiple processes for different detectors)
        detection_workers = self.config.face_detection.worker_count
        self.detection_executor = ProcessPoolExecutor(
            max_workers=detection_workers,
            mp_context=None  # Use default multiprocessing context
        )
        
        # Face recognition worker pool
        recognition_workers = self.config.face_recognition.worker_count
        self.recognition_executor = ProcessPoolExecutor(
            max_workers=recognition_workers,
            mp_context=None
        )
        
        logger.info(f"Worker pools initialized: {detection_workers} detection, "
                   f"{recognition_workers} recognition workers")
    
    def _start_pipeline_stages(self) -> None:
        """Start all enabled pipeline stages in threads."""
        for stage in self.stages:
            if not stage.enabled:
                logger.info(f"Skipping disabled stage: {stage.name}")
                continue
            
            try:
                # Start stage worker in its own thread
                stage.thread = threading.Thread(
                    target=stage.worker.start,
                    name=f"Pipeline-{stage.name}",
                    daemon=False
                )
                stage.thread.start()
                
                logger.info(f"Started pipeline stage: {stage.name}")
                
                # Small delay to ensure proper startup order
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to start stage {stage.name}: {e}")
                if stage.critical:
                    raise
    
    def _setup_gpio_trigger(self) -> None:
        """Setup GPIO doorbell trigger to start the pipeline."""
        def on_doorbell_pressed(channel):
            """Handle doorbell press by triggering frame capture."""
            current_time = time.time()
            
            # Create doorbell event
            event = PipelineEvent(
                event_type=EventType.DOORBELL_PRESSED,
                data={'timestamp': current_time, 'channel': channel},
                timestamp=current_time,
                source='gpio_handler'
            )
            
            # Publish to frame capture stage
            self.message_bus.publish('doorbell_events', event)
            logger.info("🔔 Doorbell pressed - pipeline triggered")
        
        # Setup GPIO callback
        self.gpio_handler.setup_doorbell_button(on_doorbell_pressed)
        logger.info("GPIO doorbell trigger configured")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop for pipeline health and metrics."""
        last_health_check = 0
        health_check_interval = 30  # seconds
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Periodic health checks
                current_time = time.time()
                if current_time - last_health_check > health_check_interval:
                    self._health_check()
                    last_health_check = current_time
                
                # Check for dead stages
                self._check_stage_health()
                
                # Sleep to avoid busy waiting
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5.0)  # Longer sleep on error
    
    def _health_check(self) -> None:
        """Perform comprehensive system health check."""
        try:
            metrics = self.get_pipeline_metrics()
            
            # Log key metrics
            logger.info(f"Pipeline health: "
                       f"Queues: {metrics['queue_status']}, "
                       f"Workers: {metrics['worker_status']}, "
                       f"Events/min: {metrics.get('events_per_minute', 0)}")
            
            # Check for concerning metrics
            if metrics['queue_status'].get('backlog_warning', False):
                logger.warning("Queue backlog detected - consider scaling workers")
            
            if metrics['worker_status'].get('high_cpu', False):
                logger.warning("High CPU usage detected in workers")
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def _check_stage_health(self) -> None:
        """Check if all critical pipeline stages are running."""
        for stage in self.stages:
            if not stage.enabled or not stage.critical:
                continue
            
            if stage.thread and not stage.thread.is_alive():
                logger.error(f"Critical stage {stage.name} has died - initiating restart")
                try:
                    # Attempt to restart the stage
                    stage.thread = threading.Thread(
                        target=stage.worker.start,
                        name=f"Pipeline-{stage.name}-restart",
                        daemon=False
                    )
                    stage.thread.start()
                    logger.info(f"Successfully restarted stage: {stage.name}")
                except Exception as e:
                    logger.error(f"Failed to restart stage {stage.name}: {e}")
                    # If we can't restart critical stages, shutdown
                    self.shutdown()
    
    def shutdown(self) -> None:
        """Gracefully shutdown the entire pipeline."""
        if not self.running:
            return
        
        logger.info("Initiating pipeline shutdown...")
        self.running = False
        self.shutdown_event.set()
        
        try:
            # Stop GPIO handler first
            if self.gpio_handler:
                self.gpio_handler.cleanup()
            
            # Stop pipeline stages in reverse order
            for stage in reversed(self.stages):
                if stage.thread and stage.thread.is_alive():
                    logger.info(f"Stopping stage: {stage.name}")
                    stage.worker.stop()
                    stage.thread.join(timeout=5.0)
                    if stage.thread.is_alive():
                        logger.warning(f"Stage {stage.name} did not stop gracefully")
            
            # Shutdown worker pools
            if self.detection_executor:
                self.detection_executor.shutdown(wait=True, timeout=10.0)
            if self.recognition_executor:
                self.recognition_executor.shutdown(wait=True, timeout=10.0)
            
            # Stop infrastructure services
            self.event_database.stop()
            self.queue_manager.stop()
            self.message_bus.stop()
            
            # Cleanup camera
            if self.camera_handler:
                self.camera_handler.cleanup()
            
            logger.info("Pipeline shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline performance metrics."""
        try:
            return {
                'pipeline_status': 'running' if self.running else 'stopped',
                'uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0,
                'stages': {
                    stage.name: {
                        'enabled': stage.enabled,
                        'running': stage.thread.is_alive() if stage.thread else False,
                        'critical': stage.critical
                    }
                    for stage in self.stages
                },
                'queue_status': self.queue_manager.get_status(),
                'worker_status': self._get_worker_status(),
                'message_bus_stats': self.message_bus.get_stats(),
                'events_per_minute': self.event_database.get_event_rate()
            }
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {'error': str(e)}
    
    def _get_worker_status(self) -> Dict[str, Any]:
        """Get worker pool status and performance metrics."""
        return {
            'detection_workers': {
                'count': self.config.face_detection.worker_count,
                'active': 'unknown',  # Would need process monitoring
                'queue_size': 'unknown'
            },
            'recognition_workers': {
                'count': self.config.face_recognition.worker_count,
                'active': 'unknown',
                'queue_size': 'unknown'
            }
        }


def main():
    """Main entry point for the pipeline orchestrator."""
    try:
        # Setup logging
        setup_logging()
        
        # Create and start pipeline
        orchestrator = PipelineOrchestrator()
        orchestrator.start_time = time.time()
        orchestrator.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Pipeline orchestrator failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())