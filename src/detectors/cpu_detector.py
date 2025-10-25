#!/usr/bin/env python3
"""
CPU-based Face Detector

Face detection implementation using face_recognition library (dlib HOG/CNN)
optimized for CPU execution on Raspberry Pi and general systems.
"""

import logging
from typing import Dict, Any, List

import numpy as np

from src.detectors.base_detector import BaseDetector, DetectorType, ModelType, FaceDetectionResult

logger = logging.getLogger(__name__)


class CPUDetector(BaseDetector):
    """
    CPU-based face detector using face_recognition library.
    
    This detector uses dlib's HOG (Histogram of Oriented Gradients) or CNN
    face detection models, both of which run efficiently on CPU.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CPU detector.
        
        Args:
            config: Detector configuration including model type and parameters
        """
        # Store model config before super init
        self._model_name = config.get('model', 'hog')  # 'hog' or 'cnn'
        self._upsample_times = config.get('number_of_times_to_upsample', 1)
        
        # Initialize base detector
        super().__init__(config)
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if face_recognition library is available."""
        try:
            import face_recognition
            return True
        except ImportError:
            logger.warning("face_recognition library not available")
            return False
    
    def _get_detector_type(self) -> DetectorType:
        """Return CPU detector type."""
        return DetectorType.CPU
    
    def _initialize_model(self) -> None:
        """Initialize face_recognition detector."""
        try:
            import face_recognition
            self._face_recognition = face_recognition
            
            # Validate model type
            if self._model_name not in ['hog', 'cnn']:
                logger.warning(f"Unknown model type '{self._model_name}', defaulting to 'hog'")
                self._model_name = 'hog'
            
            # Update model type in config
            if self._model_name == 'hog':
                self.model_type = ModelType.HOG
            elif self._model_name == 'cnn':
                self.model_type = ModelType.CNN
            
            logger.info(f"Initialized CPU detector with {self._model_name} model (upsample: {self._upsample_times})")
            
        except ImportError as e:
            logger.error(f"Failed to import face_recognition: {e}")
            raise RuntimeError("face_recognition library not available") from e
    
    def _run_inference(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """
        Run face detection using face_recognition library.
        
        Args:
            image: Input image in RGB format (numpy array)
            
        Returns:
            List of face detection results
        """
        try:
            # Detect face locations using face_recognition
            # Returns list of (top, right, bottom, left) tuples
            face_locations = self._face_recognition.face_locations(
                image,
                number_of_times_to_upsample=self._upsample_times,
                model=self._model_name
            )
            
            # Convert to FaceDetectionResult objects
            detections = []
            for location in face_locations:
                top, right, bottom, left = location
                
                # Calculate confidence (face_recognition doesn't provide confidence scores)
                # For now, use a default high confidence for detected faces
                confidence = 0.95 if self._model_name == 'cnn' else 0.90
                
                detection = FaceDetectionResult(
                    bounding_box=(top, right, bottom, left),
                    confidence=confidence,
                    landmarks=None,  # Could extract landmarks separately if needed
                    quality_score=0.0  # Will be calculated in postprocessing
                )
                detections.append(detection)
            
            logger.debug(f"CPU detector found {len(detections)} faces")
            return detections
            
        except Exception as e:
            logger.error(f"CPU face detection failed: {e}")
            return []
    
    def detect_faces_with_landmarks(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """
        Detect faces and extract facial landmarks.
        
        Args:
            image: Input image in RGB format
            
        Returns:
            List of face detections with landmark information
        """
        try:
            # First detect face locations
            face_locations = self._face_recognition.face_locations(
                image,
                number_of_times_to_upsample=self._upsample_times,
                model=self._model_name
            )
            
            # Then extract landmarks
            face_landmarks_list = self._face_recognition.face_landmarks(image, face_locations)
            
            # Combine locations and landmarks
            detections = []
            for location, landmarks in zip(face_locations, face_landmarks_list):
                top, right, bottom, left = location
                
                # Convert landmarks to our format
                landmarks_dict = {
                    'chin': landmarks.get('chin', []),
                    'left_eyebrow': landmarks.get('left_eyebrow', []),
                    'right_eyebrow': landmarks.get('right_eyebrow', []),
                    'nose_bridge': landmarks.get('nose_bridge', []),
                    'nose_tip': landmarks.get('nose_tip', []),
                    'left_eye': landmarks.get('left_eye', []),
                    'right_eye': landmarks.get('right_eye', []),
                    'top_lip': landmarks.get('top_lip', []),
                    'bottom_lip': landmarks.get('bottom_lip', [])
                }
                
                confidence = 0.95 if self._model_name == 'cnn' else 0.90
                
                detection = FaceDetectionResult(
                    bounding_box=(top, right, bottom, left),
                    confidence=confidence,
                    landmarks=landmarks_dict,
                    quality_score=0.0
                )
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Face detection with landmarks failed: {e}")
            return []
    
    def cleanup(self) -> None:
        """Cleanup detector resources."""
        super().cleanup()
        self._face_recognition = None
