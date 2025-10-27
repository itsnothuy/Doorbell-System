"""
Mock camera implementation for testing and development

Provides a realistic mock camera that generates synthetic frames with face-like
patterns for testing without actual camera hardware.
"""

import logging
import time
import random
from typing import Optional, List
from pathlib import Path
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


class MockCameraHandler(CameraHandler):
    """Mock camera implementation for testing without actual hardware."""
    
    def __init__(self, config: dict):
        """
        Initialize mock camera handler.
        
        Args:
            config: Configuration dictionary with camera settings
        """
        self.config = config
        self.is_initialized = False
        self.stream_active = False
        
        # Camera settings
        self.settings = CameraSettings(
            resolution=config.get('resolution', (640, 480)),
            fps=config.get('fps', 15.0),
            brightness=config.get('brightness', 50.0),
            contrast=config.get('contrast', 50.0),
            rotation=config.get('rotation', 0)
        )
        
        # Mock data generation
        self.frame_counter = 0
        self.test_images = []
        self.base_image = None
        
        logger.info("Mock camera handler created")
    
    def initialize(self) -> bool:
        """Initialize mock camera."""
        try:
            logger.info("Initializing mock camera...")
            
            # Load test images if available
            self.test_images = self._load_test_images()
            
            # Generate base synthetic image
            self.base_image = self._create_synthetic_image()
            
            self.is_initialized = True
            logger.info("Mock camera initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Mock camera initialization failed: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Generate mock frame data."""
        if not self.is_initialized:
            logger.warning("Mock camera not initialized")
            return None
        
        try:
            # Cycle through test images or generate synthetic frames
            if self.test_images and len(self.test_images) > 0:
                frame = self.test_images[self.frame_counter % len(self.test_images)].copy()
            else:
                frame = self._generate_synthetic_frame()
            
            self.frame_counter += 1
            return frame
            
        except Exception as e:
            logger.error(f"Mock frame capture failed: {e}")
            return None
    
    def start_stream(self) -> bool:
        """Start mock camera stream."""
        if not self.is_initialized:
            return False
        
        self.stream_active = True
        logger.info("Mock camera stream started")
        return True
    
    def stop_stream(self) -> None:
        """Stop mock camera stream."""
        self.stream_active = False
        logger.info("Mock camera stream stopped")
    
    def get_camera_info(self) -> CameraInfo:
        """Get mock camera information."""
        return CameraInfo(
            name="Mock Camera",
            resolution=self.settings.resolution,
            fps=self.settings.fps,
            backend="mock",
            available=self.is_initialized
        )
    
    def set_camera_settings(self, settings: CameraSettings) -> bool:
        """Update mock camera settings."""
        try:
            self.settings = settings
            logger.info(f"Mock camera settings updated: {settings}")
            return True
        except Exception as e:
            logger.error(f"Failed to update mock camera settings: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if mock camera is available."""
        return self.is_initialized
    
    def cleanup(self) -> None:
        """Cleanup mock camera resources."""
        self.is_initialized = False
        self.stream_active = False
        self.test_images = []
        self.base_image = None
        logger.info("Mock camera cleanup completed")
    
    def _create_synthetic_image(self) -> np.ndarray:
        """Create a synthetic test image with face-like features."""
        height, width = self.settings.resolution[1], self.settings.resolution[0]
        
        # Create a gradient background
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add a gradient background
        for y in range(height):
            image[y, :, 0] = int(50 + (y / height) * 50)  # Red channel
            image[y, :, 1] = int(100 + (y / height) * 50)  # Green channel
            image[y, :, 2] = int(150 + (y / height) * 50)  # Blue channel
        
        if OPENCV_AVAILABLE:
            # Add some geometric shapes to simulate a face
            center_x, center_y = width // 2, height // 2
            
            # Face outline (circle)
            cv2.circle(image, (center_x, center_y), min(width, height) // 6, (200, 180, 160), -1)
            
            # Eyes
            eye_y = center_y - height // 12
            cv2.circle(image, (center_x - width // 12, eye_y), width // 40, (50, 50, 50), -1)
            cv2.circle(image, (center_x + width // 12, eye_y), width // 40, (50, 50, 50), -1)
            
            # Nose
            nose_y = center_y
            cv2.circle(image, (center_x, nose_y), width // 60, (150, 120, 100), -1)
            
            # Mouth
            mouth_y = center_y + height // 15
            cv2.ellipse(image, (center_x, mouth_y), (width // 20, height // 40), 
                       0, 0, 180, (100, 50, 50), -1)
        
        return image
    
    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generate synthetic frame with variations."""
        if self.base_image is None:
            self.base_image = self._create_synthetic_image()
        
        # Add some random noise to simulate different lighting conditions
        noise = np.random.randint(-20, 20, self.base_image.shape, dtype=np.int16)
        noisy_image = np.clip(self.base_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Occasionally add some random movement simulation
        if OPENCV_AVAILABLE and np.random.random() < 0.3:
            # Slight shift to simulate person movement
            shift_x = np.random.randint(-10, 10)
            shift_y = np.random.randint(-5, 5)
            
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            noisy_image = cv2.warpAffine(
                noisy_image, M, 
                (noisy_image.shape[1], noisy_image.shape[0])
            )
        
        return noisy_image
    
    def _load_test_images(self) -> List[np.ndarray]:
        """Load test images for mock frames."""
        test_images = []
        test_image_dir = Path("tests/data/test_images")
        
        if test_image_dir.exists() and OPENCV_AVAILABLE:
            for image_path in test_image_dir.glob("*.jpg"):
                try:
                    image = cv2.imread(str(image_path))
                    if image is not None:
                        # Convert BGR to RGB
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        # Resize to match configured resolution
                        image = cv2.resize(image, self.settings.resolution)
                        test_images.append(image)
                except Exception as e:
                    logger.warning(f"Failed to load test image {image_path}: {e}")
        
        if test_images:
            logger.info(f"Loaded {len(test_images)} test images for mock camera")
        
        return test_images


class MockCamera:
    """
    Simplified mock camera for backward compatibility.
    
    This class provides a simpler interface for testing and can be used
    with the existing camera handler infrastructure.
    """
    
    def __init__(self):
        """Initialize simplified mock camera."""
        from config.settings import Settings
        self.settings = Settings()
        self.base_image = self._create_test_image()
    
    def _create_test_image(self) -> np.ndarray:
        """Create a test image with face-like features."""
        width, height = self.settings.CAMERA_RESOLUTION
        
        # Create a gradient background
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add a gradient background
        for y in range(height):
            image[y, :, 0] = int(50 + (y / height) * 50)  # Red channel
            image[y, :, 1] = int(100 + (y / height) * 50)  # Green channel
            image[y, :, 2] = int(150 + (y / height) * 50)  # Blue channel
        
        if OPENCV_AVAILABLE:
            # Add some geometric shapes to simulate a face
            center_x, center_y = width // 2, height // 2
            
            # Face outline (circle)
            cv2.circle(image, (center_x, center_y), min(width, height) // 6, (200, 180, 160), -1)
            
            # Eyes
            eye_y = center_y - height // 12
            cv2.circle(image, (center_x - width // 12, eye_y), width // 40, (50, 50, 50), -1)
            cv2.circle(image, (center_x + width // 12, eye_y), width // 40, (50, 50, 50), -1)
            
            # Nose
            nose_y = center_y
            cv2.circle(image, (center_x, nose_y), width // 60, (150, 120, 100), -1)
            
            # Mouth
            mouth_y = center_y + height // 15
            cv2.ellipse(image, (center_x, mouth_y), (width // 20, height // 40), 
                       0, 0, 180, (100, 50, 50), -1)
        
        return image
    
    def capture(self) -> np.ndarray:
        """Simulate capturing an image."""
        # Add some random noise to simulate different lighting conditions
        noise = np.random.randint(-20, 20, self.base_image.shape, dtype=np.int16)
        noisy_image = np.clip(self.base_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Occasionally add some random movement simulation
        if OPENCV_AVAILABLE and np.random.random() < 0.3:
            # Slight shift to simulate person movement
            shift_x = np.random.randint(-10, 10)
            shift_y = np.random.randint(-5, 5)
            
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            noisy_image = cv2.warpAffine(
                noisy_image, M,
                (noisy_image.shape[1], noisy_image.shape[0])
            )
        
        return noisy_image


__all__ = ['MockCameraHandler', 'MockCamera']
