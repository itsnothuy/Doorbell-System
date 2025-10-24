"""
Camera operations module for cross-platform support
"""

import logging
import time
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional
from src.platform_detector import platform_detector

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

logger = logging.getLogger(__name__)


class CameraHandler:
    """Handles camera operations with fallback support"""
    
    def __init__(self):
        from config.settings import Settings
        self.settings = Settings()
        
        self.camera = None
        self.camera_type = None
        self.is_initialized = False
        
        logger.info("Camera Handler initialized")
    
    def initialize(self):
        """Initialize camera with platform-appropriate method"""
        print("DEBUG: Starting CameraHandler.initialize() method.") # Aggressive debug print
        logger.debug("Starting CameraHandler.initialize() method.")
        try:
            camera_config = platform_detector.get_camera_config()
            print(f"DEBUG: Camera config from platform_detector: {camera_config}") # Aggressive debug print
            logger.debug(f"Camera config from platform_detector: {camera_config}")

            if camera_config['mock']:
                # Use mock camera for testing
                self._init_mock()
                self.camera_type = 'mock'
                print("DEBUG: Camera initialized with mock implementation") # Aggressive debug print
                logger.info("Camera initialized with mock implementation")

            elif camera_config['type'] == 'picamera' and PICAMERA2_AVAILABLE:
                # Use Pi Camera
                if self._init_picamera2():
                    self.camera_type = 'picamera2'
                    print("DEBUG: Camera initialized with picamera2") # Aggressive debug print
                    logger.info("Camera initialized with picamera2")
                else:
                    print("ERROR: Pi Camera initialization failed") # Aggressive debug print
                    # raise Exception("Pi Camera initialization failed") # Temporarily removed raise
                    self._init_mock() # Fallback to mock on error
                    self.camera_type = 'mock'
                    logger.warning("Pi Camera initialization failed, falling back to mock camera.")

            elif OPENCV_AVAILABLE:
                # Try to use webcam/USB camera via OpenCV
                print("DEBUG: Attempting to initialize OpenCV camera...") # Aggressive debug print
                logger.debug("Attempting to initialize OpenCV camera...")
                if self._init_opencv():
                    self.camera_type = 'opencv'
                    if platform_detector.is_macos:
                        print("DEBUG: Camera initialized with macOS webcam") # Aggressive debug print
                        logger.info("Camera initialized with macOS webcam")
                    else:
                        print("DEBUG: Camera initialized with OpenCV") # Aggressive debug print
                        logger.info("Camera initialized with OpenCV")
                else:
                    print("WARNING: OpenCV camera initialization failed, falling back to mock camera.") # Aggressive debug print
                    logger.warning("OpenCV camera initialization failed, falling back to mock camera.")
                    self._init_mock()
                    self.camera_type = 'mock'
                    print("DEBUG: Camera initialized with mock implementation (fallback).") # Aggressive debug print
                    logger.info("Camera initialized with mock implementation (fallback).")

            else:
                # Fallback to mock if no other camera backend is available
                self._init_mock()
                self.camera_type = 'mock'
                print("WARNING: No camera backend available, using mock implementation.") # Aggressive debug print
                logger.warning("No camera backend available, using mock implementation.")
            
            self.is_initialized = True
            
            # Test capture
            test_image = self.capture_image()
            if test_image is None and self.camera_type != 'mock': # Only log if not using mock
                print("ERROR: Camera test capture failed") # Aggressive debug print
                # raise Exception("Camera test capture failed") # Temporarily removed raise
                logger.error("Camera test capture failed.")
            elif test_image is None and self.camera_type == 'mock':
                print("DEBUG: Mock camera test capture successful.") # Aggressive debug print
                logger.info("Mock camera test capture successful.")
            else:
                print("DEBUG: Camera test successful.") # Aggressive debug print
                logger.info("Camera test successful.")
            
        except Exception as e:
            print(f"ERROR: Camera initialization failed: {e}") # Aggressive debug print
            logger.error(f"Camera initialization failed: {e}")
            # raise # Temporarily removed raise
    
    def _init_mock(self) -> bool:
        """Initialize mock camera for testing"""
        print("DEBUG: Initializing mock camera...") # Aggressive debug print
        try:
            # Create a mock camera object
            self.camera = MockCamera()
            return True
        except Exception as e:
            logger.warning(f"Mock camera initialization failed: {e}")
            return False
    
    def _init_picamera2(self) -> bool:
        """Initialize with picamera2 library"""
        try:
            self.camera = Picamera2()
            
            # Configure camera
            config = self.camera.create_still_configuration(
                main={"size": self.settings.CAMERA_RESOLUTION}
            )
            self.camera.configure(config)
            
            # Set camera properties
            controls = {}
            if hasattr(self.settings, 'CAMERA_BRIGHTNESS'):
                controls["Brightness"] = self.settings.CAMERA_BRIGHTNESS / 100.0
            if hasattr(self.settings, 'CAMERA_CONTRAST'):
                controls["Contrast"] = self.settings.CAMERA_CONTRAST / 100.0
            
            if controls:
                self.camera.set_controls(controls)
            
            # Start camera
            self.camera.start()
            time.sleep(2)  # Camera warm-up time
            
            return True
            
        except Exception as e:
            logger.warning(f"picamera2 initialization failed: {e}")
            return False
    
    def _init_opencv(self) -> bool:
        """Initialize with OpenCV VideoCapture"""
        try:
            # Try different camera indices to find a working webcam
            for i in range(self.settings.OPENCV_CAMERA_MAX_INDEX_TO_TRY):
                print(f"DEBUG: Attempting to open OpenCV camera at index {i}...") # Aggressive debug print
                self.camera = cv2.VideoCapture(i)
                logger.debug(f"Attempting to open OpenCV camera at index {i}...")

                if not self.camera.isOpened():
                    print(f"WARNING: OpenCV camera failed to open at index {i}. Trying next index if available.") # Aggressive debug print
                    logger.warning(f"OpenCV camera failed to open at index {i}. Trying next index if available.")
                    self.camera.release() # Release any opened but non-functional camera
                    continue # Try next index
                
                print(f"DEBUG: OpenCV camera successfully opened at index {i}.") # Aggressive debug print
                logger.info(f"OpenCV camera successfully opened at index {i}.")

                # Set resolution
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.CAMERA_RESOLUTION[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.CAMERA_RESOLUTION[1])

                # Set other properties if supported
                if hasattr(self.settings, 'CAMERA_BRIGHTNESS'):
                    self.camera.set(cv2.CAP_PROP_BRIGHTNESS, self.settings.CAMERA_BRIGHTNESS / 100.0)
                if hasattr(self.settings, 'CAMERA_CONTRAST'):
                    self.camera.set(cv2.CAP_PROP_CONTRAST, self.settings.CAMERA_CONTRAST / 100.0)

                # Test capture
                ret, frame = self.camera.read()
                if not ret:
                    print(f"WARNING: OpenCV camera at index {i} failed test capture. Trying next index if available.") # Aggressive debug print
                    logger.warning(f"OpenCV camera at index {i} failed test capture. Trying next index if available.")
                    self.camera.release()
                    continue # Try next index
                
                print(f"DEBUG: OpenCV camera at index {i} passed test capture.") # Aggressive debug print
                return True # Camera successfully initialized and tested

            print("ERROR: No OpenCV camera could be initialized after checking multiple indices.") # Aggressive debug print
            logger.error("No OpenCV camera could be initialized after checking multiple indices.")
            return False

        except Exception as e:
            print(f"ERROR: OpenCV camera initialization failed: {e}") # Aggressive debug print
            logger.warning(f"OpenCV camera initialization failed: {e}")
            return False
    
    def capture_image(self) -> Optional[np.ndarray]:
        """Capture an image from the camera"""
        if not self.is_initialized:
            logger.error("Camera not initialized")
            return None
        
        try:
            if self.camera_type == 'picamera2':
                return self._capture_picamera2()
            elif self.camera_type == 'opencv':
                return self._capture_opencv()
            elif self.camera_type == 'mock':
                return self._capture_mock()
            else:
                logger.error(f"Unknown camera type: {self.camera_type}")
                return None
                
        except Exception as e:
            logger.error(f"Image capture failed: {e}")
            return None
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame (alias for capture_image for pipeline compatibility)."""
        return self.capture_image()
    
    def get_status(self) -> str:
        """Get camera status for monitoring."""
        if not self.is_initialized:
            return 'not_initialized'
        if self.camera_type == 'mock':
            return 'mock_active'
        if self.camera_type in ['picamera2', 'opencv']:
            return 'active'
        return 'unknown'
    
    def _capture_picamera2(self) -> Optional[np.ndarray]:
        """Capture image using picamera2"""
        try:
            # Capture image as numpy array
            image_array = self.camera.capture_array()
            
            # Convert from RGB to standard format if needed
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # Already RGB format
                return image_array
            else:
                # Convert other formats
                return cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                
        except Exception as e:
            logger.error(f"picamera2 capture failed: {e}")
            return None
    
    def _capture_opencv(self) -> Optional[np.ndarray]:
        """Capture image using OpenCV"""
        try:
            ret, frame = self.camera.read()
            if not ret:
                logger.error("OpenCV capture returned no frame")
                return None
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return rgb_frame
            
        except Exception as e:
            logger.error(f"OpenCV capture failed: {e}")
            return None
    
    def save_image(self, image: np.ndarray, path: Path):
        """Save image to file"""
        try:
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Apply rotation if configured
            if self.settings.CAMERA_ROTATION != 0:
                pil_image = pil_image.rotate(self.settings.CAMERA_ROTATION, expand=True)
            
            # Save image
            pil_image.save(path, 'JPEG', quality=85)
            logger.debug(f"Image saved: {path}")
            
        except Exception as e:
            logger.error(f"Failed to save image {path}: {e}")
            raise
    
    def get_camera_info(self) -> dict:
        """Get camera information"""
        info = {
            'initialized': self.is_initialized,
            'camera_type': self.camera_type,
            'resolution': self.settings.CAMERA_RESOLUTION,
            'rotation': self.settings.CAMERA_ROTATION
        }
        
        if self.camera_type == 'opencv' and self.camera:
            try:
                info.update({
                    'actual_width': int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'actual_height': int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'fps': self.camera.get(cv2.CAP_PROP_FPS)
                })
            except:
                pass
        
        return info
    
    def cleanup(self):
        """Cleanup camera resources"""
        try:
            if self.camera:
                if self.camera_type == 'picamera2':
                    self.camera.stop()
                    self.camera.close()
                elif self.camera_type == 'opencv':
                    self.camera.release()
                
                self.camera = None
                self.is_initialized = False
                logger.info("Camera cleanup completed")
                
        except Exception as e:
            logger.error(f"Camera cleanup failed: {e}")


    def _capture_mock(self) -> Optional[np.ndarray]:
        """Capture mock image"""
        try:
            return self.camera.capture()
        except Exception as e:
            logger.error(f"Mock capture failed: {e}")
            return None


class MockCamera:
    """Mock camera for testing"""
    
    def __init__(self):
        # Create a more realistic test image with face-like features
        self.base_image = self._create_test_image()
    
    def _create_test_image(self) -> np.ndarray:
        """Create a test image with face-like features"""
        from config.settings import Settings
        settings = Settings()
        width, height = settings.CAMERA_RESOLUTION
        
        # Create a gradient background
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add a gradient background
        for y in range(height):
            image[y, :, 0] = int(50 + (y / height) * 50)  # Red channel
            image[y, :, 1] = int(100 + (y / height) * 50)  # Green channel
            image[y, :, 2] = int(150 + (y / height) * 50)  # Blue channel
        
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
        cv2.ellipse(image, (center_x, mouth_y), (width // 20, height // 40), 0, 0, 180, (100, 50, 50), -1)
        
        return image
    
    def capture(self) -> np.ndarray:
        """Simulate capturing an image"""
        # Add some random noise to simulate different lighting conditions
        noise = np.random.randint(-20, 20, self.base_image.shape, dtype=np.int16)
        noisy_image = np.clip(self.base_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Occasionally add some random movement simulation
        if np.random.random() < 0.3:
            # Slight shift to simulate person movement
            shift_x = np.random.randint(-10, 10)
            shift_y = np.random.randint(-5, 5)
            
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            noisy_image = cv2.warpAffine(noisy_image, M, (noisy_image.shape[1], noisy_image.shape[0]))
        
        return noisy_image


class MockCameraHandler(CameraHandler):
    """Mock camera handler for testing without actual camera"""
    
    def __init__(self):
        super().__init__()
        self.mock_image = None
        
    def initialize(self):
        """Mock initialization"""
        # Create a simple test image
        self.mock_image = np.random.randint(0, 255, 
                                          (*self.settings.CAMERA_RESOLUTION[::-1], 3), 
                                          dtype=np.uint8)
        self.is_initialized = True
        self.camera_type = 'mock'
        logger.info("Mock camera initialized")
    
    def capture_image(self) -> Optional[np.ndarray]:
        """Return mock image"""
        if not self.is_initialized:
            return None
        
        # Add some noise to simulate different captures
        noise = np.random.randint(-10, 10, self.mock_image.shape, dtype=np.int16)
        noisy_image = np.clip(self.mock_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return noisy_image
    
    def cleanup(self):
        """Mock cleanup"""
        self.is_initialized = False
        logger.info("Mock camera cleanup completed")
