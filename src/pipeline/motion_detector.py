#!/usr/bin/env python3
"""
Motion Detection Worker

Implements motion detection as an optional performance optimization stage in the pipeline.
Uses background subtraction and motion region analysis to filter frames before face detection.
"""

import time
import logging
from typing import Dict, Any, Optional, List, Tuple

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    np = None

from src.pipeline.base_worker import PipelineWorker
from src.communication.message_bus import MessageBus, Message
from src.communication.events import (
    EventType, FrameEvent, MotionResult, MotionHistory
)
from config.motion_config import MotionConfig

logger = logging.getLogger(__name__)


class MotionDetector(PipelineWorker):
    """Motion detection worker for pipeline optimization."""
    
    def __init__(self, message_bus: MessageBus, config: MotionConfig):
        """
        Initialize motion detector.
        
        Args:
            message_bus: Message bus for inter-component communication
            config: Motion detection configuration
        """
        # Check if OpenCV is available
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) is required for motion detection but not available")
        
        # Store motion config before calling parent init
        self.motion_config = config
        
        # Convert motion config to worker config format
        worker_config = {
            'enabled': config.enabled,
            'queue_size': config.queue_size,
            'timeout': config.timeout
        }
        
        # Initialize parent worker
        super().__init__(message_bus, worker_config)
        
        # Background subtraction model
        self.bg_subtractor: Optional[cv2.BackgroundSubtractor] = None
        self.background_model: Optional[np.ndarray] = None
        
        # Motion history tracking
        self.motion_history = MotionHistory()
        
        # Performance tracking
        self.frames_processed = 0
        self.frames_forwarded = 0
        self.motion_events = 0
        self.frame_skip_counter = 0
        self.last_forwarded_time = time.time()
        
        logger.info(f"Initialized {self.worker_id} with config: enabled={config.enabled}, "
                   f"threshold={config.motion_threshold}, subtractor={config.bg_subtractor_type}")
    
    def _setup_subscriptions(self) -> None:
        """Setup message bus subscriptions."""
        self.message_bus.subscribe('frame_captured', self._handle_frame_event, self.worker_id)
        self.message_bus.subscribe('pipeline_control', self._handle_control_event, self.worker_id)
        self.message_bus.subscribe('system_shutdown', self.handle_shutdown, self.worker_id)
        logger.debug(f"{self.worker_id} subscriptions configured")
    
    def _initialize_worker(self) -> None:
        """Initialize background subtractor and worker resources."""
        try:
            self.bg_subtractor = self._create_background_subtractor()
            logger.info(f"{self.worker_id} background subtractor initialized: {self.motion_config.bg_subtractor_type}")
        except Exception as e:
            logger.error(f"{self.worker_id} initialization failed: {e}")
            raise
    
    def _create_background_subtractor(self) -> cv2.BackgroundSubtractor:
        """
        Create background subtractor based on configuration.
        
        Returns:
            cv2.BackgroundSubtractor: Configured background subtractor
        """
        if self.motion_config.bg_subtractor_type == "MOG2":
            return cv2.createBackgroundSubtractorMOG2(
                history=self.motion_config.bg_history,
                varThreshold=self.motion_config.bg_var_threshold,
                detectShadows=True
            )
        elif self.motion_config.bg_subtractor_type == "KNN":
            return cv2.createBackgroundSubtractorKNN(
                history=self.motion_config.bg_history,
                dist2Threshold=400.0,
                detectShadows=True
            )
        else:
            raise ValueError(f"Unsupported background subtractor: {self.motion_config.bg_subtractor_type}")
    
    def _handle_frame_event(self, message: Message) -> None:
        """
        Process incoming frame from capture worker.
        
        Args:
            message: Message containing frame event
        """
        try:
            frame_event = message.data
            
            # Skip frames if configured
            if self.motion_config.skip_frame_count > 0:
                self.frame_skip_counter = (self.frame_skip_counter + 1) % (self.motion_config.skip_frame_count + 1)
                if self.frame_skip_counter != 0:
                    return
            
            # Extract frame data
            frame = self._extract_frame_data(frame_event)
            if frame is None:
                logger.warning("No frame data available in frame event")
                self._forward_frame_fallback(frame_event)
                return
            
            # Detect motion
            motion_result = self.detect_motion(frame)
            self.frames_processed += 1
            
            # Update motion history
            self._update_motion_history(motion_result)
            
            # Decide whether to forward frame
            should_forward = self._should_forward_frame(motion_result)
            
            if should_forward:
                self._forward_frame_with_motion_data(frame_event, motion_result)
                self.frames_forwarded += 1
                
                if motion_result.motion_detected:
                    self.motion_events += 1
            
            # Publish motion statistics periodically
            if self.frames_processed % 100 == 0:
                self._publish_motion_stats()
            
            self.processed_count += 1
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Motion detection failed: {e}", exc_info=True)
            
            # Forward frame anyway to prevent pipeline stall
            self._forward_frame_fallback(frame_event)
    
    def _handle_control_event(self, message: Message) -> None:
        """
        Handle pipeline control events.
        
        Args:
            message: Control message
        """
        control_data = message.data
        command = control_data.get('command')
        
        if command == 'reset_background':
            logger.info("Resetting background model")
            self.bg_subtractor = self._create_background_subtractor()
        elif command == 'get_stats':
            stats = self.get_metrics()
            logger.info(f"Motion detector stats: {stats}")
    
    def _extract_frame_data(self, frame_event: FrameEvent) -> Optional[np.ndarray]:
        """
        Extract frame data from frame event.
        
        Args:
            frame_event: Frame event containing frame data
            
        Returns:
            numpy.ndarray: Frame data or None if unavailable
        """
        # Try to get frame data from various possible locations
        if hasattr(frame_event, 'frame_data') and frame_event.frame_data is not None:
            return frame_event.frame_data
        
        if hasattr(frame_event, 'data') and isinstance(frame_event.data, dict):
            if 'frame' in frame_event.data:
                return frame_event.data['frame']
            if 'frame_data' in frame_event.data:
                return frame_event.data['frame_data']
        
        return None
    
    def detect_motion(self, frame: np.ndarray) -> MotionResult:
        """
        Detect motion in frame using background subtraction.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            MotionResult: Motion detection analysis result
        """
        start_time = time.time()
        
        try:
            # Preprocess frame
            processed_frame = self._preprocess_frame(frame)
            
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(
                processed_frame,
                learningRate=self.motion_config.bg_learning_rate
            )
            
            # Analyze foreground mask
            motion_result = self._analyze_motion(fg_mask)
            
            return motion_result
            
        except Exception as e:
            logger.error(f"Motion detection error: {e}")
            # Return no motion on error
            return MotionResult(
                motion_detected=False,
                motion_score=0.0,
                motion_regions=[],
                contour_count=0,
                largest_contour_area=0,
                motion_center=None,
                frame_timestamp=time.time(),
                processing_time=time.time() - start_time
            )
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for motion detection.
        
        Args:
            frame: Input frame
            
        Returns:
            numpy.ndarray: Preprocessed frame
        """
        # Apply ROI if enabled
        if self.motion_config.roi_enabled and self.motion_config.roi_coordinates:
            x, y, w, h = self.motion_config.roi_coordinates
            frame = frame[y:y+h, x:x+w]
        
        # Resize for performance
        if self.motion_config.frame_resize_factor != 1.0:
            height, width = frame.shape[:2]
            new_height = int(height * self.motion_config.frame_resize_factor)
            new_width = int(width * self.motion_config.frame_resize_factor)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        frame = cv2.GaussianBlur(frame, self.motion_config.gaussian_blur_kernel, 0)
        
        return frame
    
    def _analyze_motion(self, fg_mask: np.ndarray) -> MotionResult:
        """
        Analyze foreground mask for motion characteristics.
        
        Args:
            fg_mask: Foreground mask from background subtraction
            
        Returns:
            MotionResult: Motion analysis result
        """
        start_time = time.time()
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        valid_contours = [c for c in contours if cv2.contourArea(c) >= self.motion_config.min_contour_area]
        
        # Calculate motion metrics
        motion_regions = []
        total_motion_area = 0
        largest_contour_area = 0
        motion_center = None
        
        if valid_contours:
            for contour in valid_contours:
                x, y, w, h = cv2.boundingRect(contour)
                motion_regions.append((x, y, w, h))
                
                area = cv2.contourArea(contour)
                total_motion_area += area
                largest_contour_area = max(largest_contour_area, area)
            
            # Calculate center of motion
            moments = cv2.moments(fg_mask)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                motion_center = (cx, cy)
        
        # Calculate motion score as percentage of frame with motion
        frame_area = fg_mask.shape[0] * fg_mask.shape[1]
        motion_score = (total_motion_area / frame_area) * 100 if frame_area > 0 else 0.0
        
        # Determine if motion is significant
        motion_detected = (
            motion_score >= self.motion_config.motion_threshold and
            len(valid_contours) > 0 and
            largest_contour_area >= self.motion_config.min_contour_area
        )
        
        processing_time = time.time() - start_time
        
        return MotionResult(
            motion_detected=motion_detected,
            motion_score=motion_score,
            motion_regions=motion_regions,
            contour_count=len(valid_contours),
            largest_contour_area=largest_contour_area,
            motion_center=motion_center,
            frame_timestamp=time.time(),
            processing_time=processing_time
        )
    
    def _should_forward_frame(self, motion_result: MotionResult) -> bool:
        """
        Determine if frame should be forwarded to face detection.
        
        Args:
            motion_result: Motion detection result
            
        Returns:
            bool: True if frame should be forwarded
        """
        # Always forward if motion detected
        if motion_result.motion_detected:
            self.last_forwarded_time = time.time()
            return True
        
        # Forward periodically even without motion (heartbeat)
        current_time = time.time()
        if current_time - self.last_forwarded_time > self.motion_config.max_static_duration:
            self.last_forwarded_time = current_time
            logger.debug("Forwarding frame due to max static duration reached")
            return True
        
        # Forward based on motion trend analysis
        if self._is_motion_trend_increasing():
            logger.debug("Forwarding frame due to increasing motion trend")
            return True
        
        # Forward if we're in a transition period
        if self._is_motion_transitioning():
            logger.debug("Forwarding frame due to motion transition")
            return True
        
        return False
    
    def _is_motion_trend_increasing(self) -> bool:
        """
        Check if motion trend is increasing.
        
        Returns:
            bool: True if motion trend is increasing
        """
        if len(self.motion_history.recent_scores) < 3:
            return False
        
        trend = self.motion_history.calculate_trend()
        return trend == "increasing"
    
    def _is_motion_transitioning(self) -> bool:
        """
        Check if motion is in transition (starting or ending).
        
        Returns:
            bool: True if in transition period
        """
        if len(self.motion_history.recent_scores) < 2:
            return False
        
        # Check if motion recently started or stopped
        recent_scores = self.motion_history.recent_scores[-3:]
        
        # Starting: low scores followed by higher scores
        if len(recent_scores) >= 2:
            if recent_scores[-2] < self.motion_config.motion_threshold / 2 and \
               recent_scores[-1] > self.motion_config.motion_threshold / 2:
                return True
        
        # Ending: higher scores followed by lower scores
        if len(recent_scores) >= 3:
            if recent_scores[-3] > self.motion_config.motion_threshold and \
               recent_scores[-1] < self.motion_config.motion_threshold / 2:
                return True
        
        return False
    
    def _update_motion_history(self, motion_result: MotionResult) -> None:
        """
        Update motion history for trend analysis.
        
        Args:
            motion_result: Motion detection result
        """
        self.motion_history.add_score(
            score=motion_result.motion_score,
            timestamp=motion_result.frame_timestamp,
            is_motion=motion_result.motion_detected
        )
        
        # Trim history to configured size
        self.motion_history.trim_history(self.motion_config.motion_history_size)
        
        # Update trend direction
        self.motion_history.trend_direction = self.motion_history.calculate_trend()
    
    def _forward_frame_with_motion_data(self, frame_event: FrameEvent, motion_result: MotionResult) -> None:
        """
        Forward frame to face detection with motion data.
        
        Args:
            frame_event: Original frame event
            motion_result: Motion detection result
        """
        # Enhance frame event with motion data
        enhanced_data = frame_event.data.copy() if hasattr(frame_event, 'data') else {}
        enhanced_data.update({
            'motion_data': motion_result.to_dict(),
            'processing_stage': 'motion_analyzed',
            'motion_detected': motion_result.motion_detected,
            'motion_score': motion_result.motion_score
        })
        
        # Update frame event data
        frame_event.data = enhanced_data
        
        # Publish enhanced frame event
        self.message_bus.publish('motion_analyzed', frame_event)
        
        logger.debug(f"Forwarded frame with motion data: detected={motion_result.motion_detected}, "
                    f"score={motion_result.motion_score:.2f}")
    
    def _forward_frame_fallback(self, frame_event: FrameEvent) -> None:
        """
        Forward frame without motion analysis (fallback on error).
        
        Args:
            frame_event: Frame event to forward
        """
        # Add minimal motion data to indicate processing attempted
        fallback_data = frame_event.data.copy() if hasattr(frame_event, 'data') else {}
        fallback_data.update({
            'motion_data': None,
            'processing_stage': 'motion_detection_failed',
            'motion_detected': False,
            'motion_score': 0.0
        })
        
        frame_event.data = fallback_data
        
        # Publish frame event
        self.message_bus.publish('motion_analyzed', frame_event)
        
        logger.debug("Forwarded frame with fallback (motion detection failed)")
    
    def _publish_motion_stats(self) -> None:
        """Publish motion detection statistics."""
        stats = {
            'frames_processed': self.frames_processed,
            'frames_forwarded': self.frames_forwarded,
            'motion_events': self.motion_events,
            'forward_ratio': self.frames_forwarded / max(1, self.frames_processed),
            'motion_ratio': self.motion_events / max(1, self.frames_processed),
            'recent_motion_scores': self.motion_history.recent_scores[-10:] if self.motion_history.recent_scores else [],
            'trend_direction': self.motion_history.trend_direction
        }
        
        logger.info(f"Motion detector stats: {stats}")
    
    def _cleanup_worker(self) -> None:
        """Cleanup worker resources."""
        try:
            # Release background subtractor
            if self.bg_subtractor is not None:
                self.bg_subtractor = None
            
            # Clear motion history
            self.motion_history = MotionHistory()
            
            logger.info(f"{self.worker_id} cleanup completed")
            
        except Exception as e:
            logger.error(f"{self.worker_id} cleanup failed: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get worker performance metrics."""
        base_metrics = super().get_metrics()
        
        motion_metrics = {
            'frames_processed': self.frames_processed,
            'frames_forwarded': self.frames_forwarded,
            'motion_events': self.motion_events,
            'forward_ratio': self.frames_forwarded / max(1, self.frames_processed),
            'motion_event_ratio': self.motion_events / max(1, self.frames_processed),
            'avg_motion_score': sum(self.motion_history.recent_scores) / len(self.motion_history.recent_scores) if self.motion_history.recent_scores else 0.0,
            'motion_trend': self.motion_history.trend_direction,
            'static_duration': self.motion_history.static_duration,
            'config': {
                'threshold': self.motion_config.motion_threshold,
                'min_contour_area': self.motion_config.min_contour_area,
                'subtractor_type': self.motion_config.bg_subtractor_type,
                'frame_resize_factor': self.motion_config.frame_resize_factor
            }
        }
        
        return {**base_metrics, **motion_metrics}


# Alias for backward compatibility with orchestrator
MotionDetectionWorker = MotionDetector
