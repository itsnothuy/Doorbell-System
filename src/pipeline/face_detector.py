#!/usr/bin/env python3
"""
Face Detection Worker Pool

Multi-process face detection worker with strategy pattern, load balancing,
and performance optimization for real-time face detection.
"""

import time
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import PriorityQueue, Empty
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from src.pipeline.base_worker import PipelineWorker
from src.communication.message_bus import MessageBus, Message
from src.communication.events import (
    PipelineEvent, EventType, FrameEvent, FaceDetection, BoundingBox
)
from src.detectors.detector_factory import DetectorFactory
from src.detectors.detection_result import DetectionResult, DetectionMetrics, FaceDetectionResult

logger = logging.getLogger(__name__)


class DetectionJob:
    """Detection job with priority and metadata."""
    
    def __init__(self, priority: int, frame_event: FrameEvent, timestamp: float):
        self.priority = priority
        self.frame_event = frame_event
        self.timestamp = timestamp
        self.job_id = f"{frame_event.event_id}_{int(timestamp * 1000)}"
    
    def __lt__(self, other):
        """Compare jobs by priority for queue ordering."""
        return self.priority < other.priority


class FaceDetectionWorker(PipelineWorker):
    """Multi-process face detection worker pool."""
    
    def __init__(self, message_bus: MessageBus, config: Dict[str, Any]):
        """
        Initialize face detection worker pool.
        
        Args:
            message_bus: Message bus for communication
            config: Worker configuration
        """
        super().__init__(message_bus, config)
        
        # Configuration
        self.worker_count = config.get('worker_count', 2)
        self.detector_type = config.get('detector_type', 'cpu')
        self.max_queue_size = config.get('max_queue_size', 100)
        self.job_timeout = config.get('job_timeout', 30.0)
        
        # Worker pool and job management
        self.worker_pool: Optional[ProcessPoolExecutor] = None
        self.pending_jobs: Dict[str, Dict[str, Any]] = {}
        self.job_queue = PriorityQueue(maxsize=self.max_queue_size)
        
        # Performance monitoring
        self.detection_count = 0
        self.detection_errors = 0
        self.total_detection_time = 0.0
        self.worker_metrics = {}
        
        logger.info(
            f"Initialized {self.worker_id} with {self.worker_count} workers, "
            f"detector: {self.detector_type}"
        )
    
    def _setup_subscriptions(self) -> None:
        """Setup message bus subscriptions."""
        self.message_bus.subscribe('frame_captured', self.handle_frame_event, self.worker_id)
        self.message_bus.subscribe('worker_health_check', self.handle_health_check, self.worker_id)
        self.message_bus.subscribe('system_shutdown', self.handle_shutdown, self.worker_id)
        logger.debug(f"{self.worker_id} subscriptions configured")
    
    def _initialize_worker(self) -> None:
        """Initialize worker pool and detector strategy."""
        try:
            # Validate detector availability
            detector_class = DetectorFactory.get_detector_class(self.detector_type)
            if not detector_class.is_available():
                logger.warning(
                    f"{self.detector_type} detector not available, falling back to CPU"
                )
                self.detector_type = 'cpu'
            
            # Initialize worker pool with spawn context for better isolation
            self.worker_pool = ProcessPoolExecutor(
                max_workers=self.worker_count,
                mp_context=mp.get_context('spawn')
            )
            
            # Test detector creation in main process
            test_detector = DetectorFactory.create(self.detector_type, self.config)
            health_result = test_detector.health_check()
            
            if health_result.get('status') != 'healthy':
                raise RuntimeError(f"Detector health check failed: {health_result}")
            
            logger.info(f"{self.worker_id} initialized with {self.detector_type} detector")
            
        except Exception as e:
            logger.error(f"{self.worker_id} initialization failed: {e}")
            raise
    
    def handle_frame_event(self, message: Message) -> None:
        """
        Handle frame capture event and schedule detection.
        
        Args:
            message: Message containing FrameEvent
        """
        frame_event = message.data
        
        try:
            # Determine job priority (doorbell events get higher priority)
            metadata = getattr(frame_event, 'data', {})
            priority = 1 if 'doorbell' in metadata.get('source', '') else 2
            
            # Create detection job
            detection_job = DetectionJob(
                priority=priority,
                frame_event=frame_event,
                timestamp=time.time()
            )
            
            # Add to queue (non-blocking with queue size limit)
            try:
                self.job_queue.put(detection_job, block=False)
                logger.debug(f"Queued detection job {detection_job.job_id}")
            except:
                logger.warning(
                    f"Detection queue full, dropping frame {frame_event.event_id}"
                )
                return
            
            # Process job queue
            self._process_job_queue()
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Frame event handling failed: {e}")
            self._handle_detection_error(e, frame_event)
    
    def _process_job_queue(self) -> None:
        """Process jobs from queue using worker pool."""
        while not self.job_queue.empty() and len(self.pending_jobs) < self.worker_count * 2:
            try:
                # Get next job
                detection_job = self.job_queue.get(block=False)
                
                # Check job age (drop old jobs)
                job_age = time.time() - detection_job.timestamp
                if job_age > self.job_timeout:
                    logger.warning(
                        f"Dropping expired job {detection_job.job_id} (age: {job_age:.2f}s)"
                    )
                    continue
                
                # Submit to worker pool
                future = self.worker_pool.submit(
                    detect_faces_worker,
                    detection_job.frame_event.frame_data,
                    self.detector_type,
                    self.config,
                    detection_job.job_id
                )
                
                # Track pending job
                self.pending_jobs[detection_job.job_id] = {
                    'future': future,
                    'job': detection_job,
                    'submit_time': time.time()
                }
                
                # Add completion callback
                future.add_done_callback(
                    lambda f, job_id=detection_job.job_id: 
                        self._handle_detection_complete(f, job_id)
                )
                
                logger.debug(f"Submitted detection job {detection_job.job_id} to worker pool")
                
            except Empty:
                break
            except Exception as e:
                logger.error(f"Job processing failed: {e}")
    
    def _handle_detection_complete(self, future, job_id: str) -> None:
        """
        Handle completed detection job.
        
        Args:
            future: Completed future object
            job_id: Job identifier
        """
        try:
            # Get job info
            job_info = self.pending_jobs.pop(job_id, None)
            if not job_info:
                logger.warning(f"Completed job {job_id} not found in pending jobs")
                return
            
            # Get results
            detection_result = future.result(timeout=1.0)
            
            # Update metrics
            processing_time = time.time() - job_info['submit_time']
            self.detection_count += 1
            self.total_detection_time += processing_time
            
            # Convert detection results to FaceDetection objects for event
            face_detections = []
            for face in detection_result.faces:
                top, right, bottom, left = face.bounding_box
                bbox = BoundingBox(
                    x=left,
                    y=top,
                    width=right - left,
                    height=bottom - top,
                    confidence=face.confidence
                )
                face_detection = FaceDetection(
                    bounding_box=bbox,
                    landmarks=face.landmarks,
                    confidence=face.confidence,
                    quality_score=face.quality_score
                )
                face_detections.append(face_detection)
            
            # Create face detection event
            from src.communication.events import FaceDetectionEvent
            
            face_detection_event = FaceDetectionEvent(
                event_type=EventType.FACES_DETECTED if face_detections else EventType.NO_FACES_DETECTED,
                faces=face_detections,
                detection_time=processing_time,
                detector_type=self.detector_type,
                frame_event=job_info['job'].frame_event
            )
            
            # Add metadata
            face_detection_event.data.update({
                'frame_event_id': job_info['job'].frame_event.event_id,
                'processing_time_ms': processing_time * 1000,
                'face_count': len(face_detections),
                'confidence_scores': [face.confidence for face in detection_result.faces],
                'detection_timestamp': time.time(),
                'detector_type': self.detector_type,
                'metrics': detection_result.metrics.to_dict()
            })
            
            # Publish detection result
            self.message_bus.publish('faces_detected', face_detection_event)
            
            logger.debug(
                f"Detection completed for {job_id}: {len(detection_result.faces)} faces found"
            )
            self.processed_count += 1
            
        except Exception as e:
            self.detection_errors += 1
            logger.error(f"Detection completion handling failed for {job_id}: {e}")
            
            # Publish error event
            if job_info:
                self._handle_detection_error(e, job_info['job'].frame_event)
    
    def _handle_detection_error(self, error: Exception, frame_event: FrameEvent) -> None:
        """
        Handle detection errors and publish error events.
        
        Args:
            error: Exception that occurred
            frame_event: Frame event being processed
        """
        error_event = PipelineEvent(
            event_type=EventType.COMPONENT_ERROR,
            data={
                'component': self.worker_id,
                'error': str(error),
                'error_type': type(error).__name__,
                'frame_event_id': frame_event.event_id,
                'detector_type': self.detector_type,
                'worker_metrics': self.get_metrics()
            },
            source=self.worker_id
        )
        
        self.message_bus.publish('detection_errors', error_event)
    
    def handle_health_check(self, message: Message) -> None:
        """
        Handle health check requests.
        
        Args:
            message: Health check message
        """
        uptime = time.time() - self.start_time if self.start_time else 0
        
        health_status = {
            'worker_id': self.worker_id,
            'detector_type': self.detector_type,
            'worker_count': self.worker_count,
            'queue_size': self.job_queue.qsize(),
            'pending_jobs': len(self.pending_jobs),
            'detection_rate': self.detection_count / max(1, uptime),
            'error_rate': self.detection_errors / max(1, self.detection_count),
            'avg_processing_time': self.total_detection_time / max(1, self.detection_count)
        }
        
        health_event = PipelineEvent(
            event_type=EventType.HEALTH_CHECK,
            data=health_status,
            source=self.worker_id
        )
        
        self.message_bus.publish('worker_health_responses', health_event)
    
    def _cleanup_worker(self) -> None:
        """Cleanup worker pool and resources."""
        try:
            # Cancel pending jobs
            for job_id, job_info in list(self.pending_jobs.items()):
                try:
                    job_info['future'].cancel()
                except:
                    pass
            
            # Shutdown worker pool
            if self.worker_pool:
                self.worker_pool.shutdown(wait=True, cancel_futures=True)
            
            # Clear job queue
            while not self.job_queue.empty():
                try:
                    self.job_queue.get(block=False)
                except Empty:
                    break
            
            logger.info(f"{self.worker_id} cleanup completed")
            
        except Exception as e:
            logger.error(f"{self.worker_id} cleanup failed: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get worker performance metrics."""
        base_metrics = super().get_metrics()
        
        uptime = time.time() - self.start_time if self.start_time else 1
        
        detection_metrics = {
            'detection_count': self.detection_count,
            'detection_errors': self.detection_errors,
            'avg_detection_time': self.total_detection_time / max(1, self.detection_count),
            'detection_rate': self.detection_count / max(1, uptime),
            'error_rate': self.detection_errors / max(1, self.detection_count) if self.detection_count > 0 else 0,
            'queue_size': self.job_queue.qsize(),
            'pending_jobs': len(self.pending_jobs),
            'worker_count': self.worker_count,
            'detector_type': self.detector_type
        }
        
        return {**base_metrics, **detection_metrics}


def detect_faces_worker(
    frame_data: np.ndarray, 
    detector_type: str, 
    config: Dict[str, Any], 
    job_id: str
) -> DetectionResult:
    """
    Worker function for face detection (runs in separate process).
    
    Args:
        frame_data: Frame image data
        detector_type: Type of detector to use
        config: Detector configuration
        job_id: Job identifier
        
    Returns:
        DetectionResult with faces and metrics
    """
    try:
        # Create detector instance in worker process
        detector = DetectorFactory.create(detector_type, config)
        
        # Perform detection
        faces, metrics = detector.detect_faces(frame_data)
        
        return DetectionResult(
            job_id=job_id,
            faces=faces,
            metrics=metrics,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Face detection worker failed for {job_id}: {e}")
        return DetectionResult(
            job_id=job_id,
            faces=[],
            metrics=DetectionMetrics(),
            success=False,
            error=str(e)
        )
