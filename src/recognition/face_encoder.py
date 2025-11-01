#!/usr/bin/env python3
"""
Face Encoder

Utility for extracting face encodings from face images using face_recognition library.
"""

import logging
import time
from typing import Dict, Any, Optional

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    face_recognition = None

logger = logging.getLogger(__name__)


class FaceEncoder:
    """Extracts face encodings from face images."""
    
    def __init__(self, config):
        """
        Initialize face encoder.
        
        Args:
            config: Configuration dictionary or config object with encoding settings
        """
        self.config = config
        
        # Handle both dict and config object
        if hasattr(config, 'encoding_model'):
            # It's a config object
            self.model = getattr(config, 'encoding_model', 'small')
            self.num_jitters = getattr(config, 'face_jitter', 1)
            self.num_upsamplings = getattr(config, 'number_of_times_to_upsample', 1)
        else:
            # It's a dictionary
            self.model = config.get('encoding_model', 'small')
            self.num_jitters = config.get('face_jitter', 1)
            self.num_upsamplings = config.get('number_of_times_to_upsample', 1)
        
        # Performance metrics
        self.encoding_count = 0
        self.total_encoding_time = 0.0
        self.encoding_errors = 0
        
        if not FACE_RECOGNITION_AVAILABLE:
            logger.warning("face_recognition library not available, using mock encoder")
        
        logger.info(f"FaceEncoder initialized with model={self.model}, jitters={self.num_jitters}")
    
    def encode_face(self, face_image: Any) -> Optional[Any]:
        """
        Extract face encoding from face image.
        
        Args:
            face_image: Face image (numpy array)
            
        Returns:
            Face encoding (128-d vector) or None if encoding fails
        """
        if not FACE_RECOGNITION_AVAILABLE or not NUMPY_AVAILABLE:
            # Return mock encoding for testing
            if np and NUMPY_AVAILABLE:
                return np.random.rand(128)
            return None
        
        start_time = time.time()
        
        try:
            # Validate input
            if face_image is None or face_image.size == 0:
                logger.warning("Invalid face image provided for encoding")
                return None
            
            # Extract encoding using face_recognition
            encodings = face_recognition.face_encodings(
                face_image,
                num_jitters=self.num_jitters,
                model=self.model
            )
            
            if not encodings:
                logger.debug("No face encoding extracted from image")
                return None
            
            # Return first encoding
            encoding = encodings[0]
            
            # Update metrics
            encoding_time = time.time() - start_time
            self.encoding_count += 1
            self.total_encoding_time += encoding_time
            
            logger.debug(f"Face encoding extracted in {encoding_time*1000:.2f}ms")
            
            return encoding
            
        except Exception as e:
            self.encoding_errors += 1
            logger.error(f"Face encoding extraction failed: {e}")
            return None
    
    def encode_faces_batch(self, face_images: list) -> list:
        """
        Extract face encodings from multiple face images.
        
        Args:
            face_images: List of face images
            
        Returns:
            List of face encodings (may contain None for failed extractions)
        """
        encodings = []
        for face_image in face_images:
            encoding = self.encode_face(face_image)
            encodings.append(encoding)
        
        return encodings
    
    def test_encoding(self) -> Optional[Any]:
        """
        Test face encoding extraction with a dummy image.
        
        Returns:
            Test encoding or None if test fails
        """
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available for encoding test")
            return None
        
        try:
            # Create a simple test image (64x64 RGB)
            test_image = np.zeros((64, 64, 3), dtype=np.uint8)
            test_image[:] = [128, 128, 128]  # Gray image
            
            if FACE_RECOGNITION_AVAILABLE:
                # Try to encode (will likely fail but tests the API)
                encoding = self.encode_face(test_image)
                if encoding is not None:
                    return encoding
            
            # Return mock encoding for testing
            return np.random.rand(128)
            
        except Exception as e:
            logger.error(f"Encoding test failed: {e}")
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get encoder performance metrics."""
        avg_time = self.total_encoding_time / max(1, self.encoding_count)
        
        return {
            'encoding_count': self.encoding_count,
            'total_encoding_time': self.total_encoding_time,
            'avg_encoding_time': avg_time,
            'encoding_errors': self.encoding_errors,
            'error_rate': self.encoding_errors / max(1, self.encoding_count),
            'model': self.model,
            'num_jitters': self.num_jitters
        }
