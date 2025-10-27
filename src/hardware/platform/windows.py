"""
Windows specific hardware implementations

Provides camera implementation for Windows using OpenCV VideoCapture with DirectShow backend.
"""

import logging
from typing import Optional
import numpy as np

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from src.hardware.base_hardware import (
    CameraHandler,
    CameraInfo,
    CameraSettings
)

logger = logging.getLogger(__name__)


class WindowsCameraHandler(CameraHandler):
    """Windows specific camera implementation using OpenCV."""
    
    def __init__(self, config: dict):
        """
        Initialize Windows camera handler.
        
        Args:
            config: Configuration dictionary with camera settings
        """
        if not OPENCV_AVAILABLE:
            raise ImportError("OpenCV not available")
        
        self.config = config
        self.capture = None
        self.stream_active = False
        self.is_initialized = False
        
        # Camera settings
        self.settings = CameraSettings(
            resolution=config.get('resolution', (1280, 720)),
            fps=config.get('fps', 15.0),
            brightness=config.get('brightness', 50.0),
            contrast=config.get('contrast', 50.0),
            rotation=config.get('rotation', 0)
        )
        
        # Camera index to try
        self.camera_index = config.get('camera_index', 0)
        self.max_index_to_try = config.get('max_index_to_try', 5)
        
        # Windows-specific: try DirectShow backend
        self.backend = cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else cv2.CAP_ANY
        
        logger.info("Windows camera handler created")
    
    def initialize(self) -> bool:
        """Initialize Windows camera using OpenCV."""
        try:
            logger.info("Initializing Windows camera...")
            
            # Try different camera indices to find a working camera
            for i in range(self.max_index_to_try):
                logger.debug(f"Trying camera index {i} with DirectShow backend...")
                self.capture = cv2.VideoCapture(i, self.backend)
                
                if not self.capture.isOpened():
                    self.capture.release()
                    continue
                
                # Configure camera settings
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.resolution[0])
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.resolution[1])
                self.capture.set(cv2.CAP_PROP_FPS, self.settings.fps)
                
                # Set brightness and contrast if supported
                if self.settings.brightness:
                    self.capture.set(cv2.CAP_PROP_BRIGHTNESS, self.settings.brightness / 100.0)
                if self.settings.contrast:
                    self.capture.set(cv2.CAP_PROP_CONTRAST, self.settings.contrast / 100.0)
                
                # Test capture
                ret, _ = self.capture.read()
                if not ret:
                    self.capture.release()
                    continue
                
                # Found working camera
                self.camera_index = i
                self.is_initialized = True
                logger.info(f"Windows camera initialized successfully at index {i}")
                return True
            
            logger.error("No working camera found on Windows")
            return False
            
        except Exception as e:
            logger.error(f"Windows camera initialization failed: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture frame using OpenCV on Windows."""
        if not self.is_initialized or not self.capture:
            return None
        
        try:
            ret, frame = self.capture.read()
            if not ret:
                logger.warning("Windows camera failed to capture frame")
                return None
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return rgb_frame
            
        except Exception as e:
            logger.error(f"Windows camera frame capture failed: {e}")
            return None
    
    def start_stream(self) -> bool:
        """Start Windows camera stream."""
        if not self.is_initialized:
            return False
        
        self.stream_active = True
        logger.info("Windows camera stream started")
        return True
    
    def stop_stream(self) -> None:
        """Stop Windows camera stream."""
        self.stream_active = False
        logger.info("Windows camera stream stopped")
    
    def get_camera_info(self) -> CameraInfo:
        """Get Windows camera information."""
        actual_width = 0
        actual_height = 0
        actual_fps = 0.0
        
        if self.capture:
            try:
                actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
            except:
                pass
        
        return CameraInfo(
            name=f"Windows Camera (index {self.camera_index})",
            resolution=(actual_width, actual_height) if actual_width > 0 else self.settings.resolution,
            fps=actual_fps if actual_fps > 0 else self.settings.fps,
            backend="opencv-dshow" if self.backend == cv2.CAP_DSHOW else "opencv",
            available=self.is_initialized
        )
    
    def set_camera_settings(self, settings: CameraSettings) -> bool:
        """Update Windows camera settings."""
        try:
            self.settings = settings
            
            if self.is_initialized and self.capture:
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.resolution[0])
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.resolution[1])
                self.capture.set(cv2.CAP_PROP_FPS, settings.fps)
                self.capture.set(cv2.CAP_PROP_BRIGHTNESS, settings.brightness / 100.0)
                self.capture.set(cv2.CAP_PROP_CONTRAST, settings.contrast / 100.0)
            
            logger.info(f"Windows camera settings updated: {settings}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update Windows camera settings: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if Windows camera is available."""
        return self.is_initialized and self.capture is not None and self.capture.isOpened()
    
    def cleanup(self) -> None:
        """Cleanup Windows camera resources."""
        try:
            if self.capture:
                self.capture.release()
                self.capture = None
            
            self.is_initialized = False
            self.stream_active = False
            logger.info("Windows camera cleanup completed")
            
        except Exception as e:
            logger.error(f"Windows camera cleanup failed: {e}")


__all__ = ['WindowsCameraHandler']
