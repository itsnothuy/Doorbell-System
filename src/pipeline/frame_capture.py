#!/usr/bin/env python3
"""
Frame Capture Worker with Ring Buffer

High-performance frame capture worker implementing continuous capture with
ring buffer, GPIO event integration, and multi-threaded processing.
"""

import time
import threading
import logging
from collections import deque
from typing import Dict, List, Any, Optional

try:
    import numpy as np
except ImportError:
    # Mock numpy for environments without it
    class np:
        @staticmethod
        def ndarray():
            pass

from src.pipeline.base_worker import PipelineWorker
from src.communication.message_bus import MessageBus, Message
from src.communication.events import (
    PipelineEvent, EventType, FrameEvent, DoorbellEvent,
    create_frame_event
)
from src.hardware.camera_handler import CameraHandler

logger = logging.getLogger(__name__)


class FrameCaptureWorker(PipelineWorker):
    """Frame capture worker with ring buffer and event-driven capture."""
    
    def __init__(self, camera_handler: CameraHandler, message_bus: MessageBus, config):
        # Store camera handler before calling parent __init__
        self.camera_handler = camera_handler
        
        # Handle both dict and config object
        if hasattr(config, 'buffer_size'):
            # It's a config object
            buffer_size = getattr(config, 'buffer_size', 30)
            capture_fps = getattr(config, 'capture_fps', 30)
            burst_count = getattr(config, 'burst_count', 5)
            burst_interval = getattr(config, 'burst_interval', 0.2)
        else:
            # It's a dictionary
            buffer_size = config.get('buffer_size', 30)
            capture_fps = config.get('capture_fps', 30)
            burst_count = config.get('burst_count', 5)
            burst_interval = config.get('burst_interval', 0.2)
        
        self.ring_buffer = deque(maxlen=buffer_size)
        
        # Threading components
        self.capture_thread: Optional[threading.Thread] = None
        self.capture_lock = threading.RLock()
        
        # Configuration
        self.capture_fps = capture_fps
        self.burst_count = burst_count
        self.burst_interval = burst_interval
        
        # Metrics
        self.frames_captured = 0
        self.capture_errors = 0
        self.last_capture_time: Optional[float] = None
        
        # Call parent __init__ after setting up our attributes
        super().__init__(message_bus, config)
        
        logger.info(f"Initialized {self.worker_id} with buffer size {self.ring_buffer.maxlen}")
    
    def _setup_subscriptions(self) -> None:
        """Setup message bus subscriptions."""
        self.message_bus.subscribe('doorbell_pressed', self.handle_doorbell_event, self.worker_id)
        self.message_bus.subscribe('system_shutdown', self.handle_shutdown, self.worker_id)
        logger.debug(f"{self.worker_id} subscriptions configured")
    
    def _initialize_worker(self) -> None:
        """Initialize camera and start continuous capture."""
        try:
            # Initialize camera if not already done
            if not self.camera_handler.is_initialized:
                if not self.camera_handler.initialize():
                    raise RuntimeError("Camera initialization failed")
            
            # Start continuous capture thread
            self.capture_thread = threading.Thread(
                target=self._continuous_capture_loop,
                name=f"{self.worker_id}_capture",
                daemon=True
            )
            self.capture_thread.start()
            
            logger.info(f"{self.worker_id} initialized successfully")
            
        except Exception as e:
            logger.error(f"{self.worker_id} initialization failed: {e}")
            raise
    
    def _continuous_capture_loop(self) -> None:
        """Continuous frame capture loop for ring buffer."""
        frame_interval = 1.0 / self.capture_fps
        
        while self.running and not self.shutdown_event.is_set():
            try:
                start_time = time.time()
                
                # Capture frame
                frame = self.camera_handler.capture_frame()
                if frame is not None:
                    self._add_frame_to_buffer(frame, {'source': 'continuous'})
                    self.frames_captured += 1
                    self.last_capture_time = time.time()
                
                # Maintain frame rate
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                self.capture_errors += 1
                logger.error(f"Continuous capture error: {e}")
                time.sleep(0.1)  # Brief pause on error
    
    def handle_doorbell_event(self, message: Message) -> None:
        """Handle doorbell press event and capture frame burst."""
        try:
            doorbell_event = message.data
            event_id = doorbell_event.event_id if hasattr(doorbell_event, 'event_id') else str(time.time())
            logger.info(f"Processing doorbell event: {event_id}")
            
            with self.capture_lock:
                # Capture burst of frames
                frames = self._capture_burst()
                
                # Publish frame events
                for i, frame_data in enumerate(frames):
                    frame_event = FrameEvent(
                        event_type=EventType.FRAME_CAPTURED,
                        frame_data=frame_data['frame'],
                        resolution=tuple(frame_data['frame'].shape[:2]),
                        capture_time=frame_data['timestamp'],
                        source='frame_capture'
                    )
                    
                    # Add metadata
                    frame_event.data.update({
                        'source': 'doorbell_burst',
                        'doorbell_event_id': event_id,
                        'burst_sequence': i,
                        'total_frames': len(frames)
                    })
                    
                    self.message_bus.publish('frame_captured', frame_event)
                
                logger.info(f"Published {len(frames)} frames for doorbell event {event_id}")
                self.processed_count += 1
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Doorbell event handling failed: {e}")
            self._handle_capture_error(e, message.data)
    
    def _capture_burst(self) -> List[Dict[str, Any]]:
        """Capture burst of frames for doorbell event."""
        frames = []
        
        try:
            for i in range(self.burst_count):
                frame = self.camera_handler.capture_frame()
                if frame is not None:
                    frame_data = {
                        'frame': frame,
                        'timestamp': time.time(),
                        'sequence': i
                    }
                    frames.append(frame_data)
                    
                    # Add to ring buffer as well
                    self._add_frame_to_buffer(frame, {'source': 'burst', 'sequence': i})
                
                # Wait between captures (except for last frame)
                if i < self.burst_count - 1:
                    time.sleep(self.burst_interval)
                    
        except Exception as e:
            logger.error(f"Burst capture failed: {e}")
            raise
        
        return frames
    
    def _add_frame_to_buffer(self, frame: Any, metadata: Dict[str, Any]) -> None:
        """Add frame to ring buffer with thread safety."""
        try:
            frame_entry = {
                'frame': frame.copy(),  # Deep copy to avoid reference issues
                'timestamp': time.time(),
                'metadata': metadata
            }
            
            with self.capture_lock:
                self.ring_buffer.append(frame_entry)
                
        except Exception as e:
            logger.error(f"Buffer add failed: {e}")
    
    def get_latest_frames(self, count: int) -> List[Dict[str, Any]]:
        """Get latest N frames from ring buffer."""
        with self.capture_lock:
            if count <= len(self.ring_buffer):
                return list(self.ring_buffer)[-count:]
            else:
                return list(self.ring_buffer)
    
    def _handle_capture_error(self, error: Exception, event_data: Any) -> None:
        """Handle capture errors and publish error events."""
        error_event = PipelineEvent(
            event_type=EventType.COMPONENT_ERROR,
            data={
                'component': self.worker_id,
                'error': str(error),
                'error_type': type(error).__name__,
                'original_event': event_data.event_id if hasattr(event_data, 'event_id') else 'unknown',
                'capture_metrics': self.get_metrics()
            },
            source=self.worker_id
        )
        
        self.message_bus.publish('capture_errors', error_event)
    
    def _cleanup_worker(self) -> None:
        """Cleanup worker resources."""
        try:
            # Signal shutdown
            self.shutdown_event.set()
            
            # Wait for capture thread
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)
            
            # Cleanup camera
            if self.camera_handler:
                self.camera_handler.cleanup()
            
            # Clear ring buffer
            with self.capture_lock:
                self.ring_buffer.clear()
            
            logger.info(f"{self.worker_id} cleanup completed")
            
        except Exception as e:
            logger.error(f"{self.worker_id} cleanup failed: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get worker performance metrics."""
        base_metrics = super().get_metrics()
        
        capture_metrics = {
            'frames_captured': self.frames_captured,
            'capture_errors': self.capture_errors,
            'buffer_size': len(self.ring_buffer),
            'buffer_capacity': self.ring_buffer.maxlen,
            'last_capture_time': self.last_capture_time,
            'capture_fps_configured': self.capture_fps,
            'camera_status': self.camera_handler.get_status() if self.camera_handler else 'unknown'
        }
        
        return {**base_metrics, **capture_metrics}
