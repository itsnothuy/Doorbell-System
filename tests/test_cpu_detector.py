#!/usr/bin/env python3
"""
Test suite for CPU Detector

Tests for CPU-based face detection implementation.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Try to import numpy, create mock if not available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Simple mock for testing without numpy
    class np:
        uint8 = int
        
        @staticmethod
        def zeros(shape, dtype=None):
            class MockArray:
                def __init__(self, shape):
                    self.shape = shape
                    self.nbytes = shape[0] * shape[1] * (shape[2] if len(shape) > 2 else 1)
            return MockArray(shape)

from src.detectors.cpu_detector import CPUDetector
from src.detectors.base_detector import DetectorType, ModelType


class TestCPUDetector(unittest.TestCase):
    """Test suite for CPU detector."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'model': 'hog',
            'number_of_times_to_upsample': 1,
            'confidence_threshold': 0.5,
            'min_face_size': (30, 30),
            'max_face_size': (1000, 1000)
        }
    
    @patch('src.detectors.cpu_detector.logger')
    def test_is_available_with_library(self, mock_logger):
        """Test availability check when face_recognition is available."""
        # Mock face_recognition import
        with patch.dict('sys.modules', {'face_recognition': Mock()}):
            available = CPUDetector.is_available()
            self.assertTrue(available)
    
    @patch('src.detectors.cpu_detector.logger')
    def test_is_available_without_library(self, mock_logger):
        """Test availability check when face_recognition is not available."""
        # Mock ImportError on face_recognition import
        with patch('builtins.__import__', side_effect=ImportError('face_recognition not found')):
            available = CPUDetector.is_available()
            self.assertFalse(available)
    
    @patch('src.detectors.cpu_detector.CPUDetector._initialize_model')
    def test_detector_type(self, mock_init):
        """Test that detector type is CPU."""
        detector = CPUDetector(self.config)
        self.assertEqual(detector.detector_type, DetectorType.CPU)
    
    @patch('src.detectors.cpu_detector.CPUDetector._initialize_model')
    def test_model_type_hog(self, mock_init):
        """Test HOG model type configuration."""
        config = self.config.copy()
        config['model'] = 'hog'
        
        detector = CPUDetector(config)
        self.assertEqual(detector._model_name, 'hog')
    
    @patch('src.detectors.cpu_detector.CPUDetector._initialize_model')
    def test_model_type_cnn(self, mock_init):
        """Test CNN model type configuration."""
        config = self.config.copy()
        config['model'] = 'cnn'
        
        detector = CPUDetector(config)
        self.assertEqual(detector._model_name, 'cnn')
    
    @patch('src.detectors.cpu_detector.CPUDetector._initialize_model')
    def test_upsample_configuration(self, mock_init):
        """Test upsample times configuration."""
        config = self.config.copy()
        config['number_of_times_to_upsample'] = 2
        
        detector = CPUDetector(config)
        self.assertEqual(detector._upsample_times, 2)
    
    def test_initialize_model_with_face_recognition(self):
        """Test model initialization with face_recognition library."""
        mock_face_recognition = Mock()
        
        with patch.dict('sys.modules', {'face_recognition': mock_face_recognition}):
            detector = CPUDetector(self.config)
            
            # Should have stored face_recognition module
            self.assertIsNotNone(detector._face_recognition)
            self.assertEqual(detector._face_recognition, mock_face_recognition)
    
    def test_initialize_model_without_face_recognition(self):
        """Test model initialization fails without face_recognition."""
        with patch('builtins.__import__', side_effect=ImportError('face_recognition not found')):
            with self.assertRaises(RuntimeError):
                detector = CPUDetector(self.config)
    
    @patch('src.detectors.cpu_detector.CPUDetector._initialize_model')
    def test_run_inference_with_faces(self, mock_init):
        """Test face detection inference with faces found."""
        detector = CPUDetector(self.config)
        
        # Mock face_recognition module
        mock_face_recognition = Mock()
        mock_face_recognition.face_locations.return_value = [
            (10, 90, 90, 10),  # top, right, bottom, left
            (100, 190, 190, 100)
        ]
        detector._face_recognition = mock_face_recognition
        
        # Create test image
        if NUMPY_AVAILABLE:
            test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        else:
            test_image = np.zeros((200, 200, 3))
        
        # Run inference
        faces = detector._run_inference(test_image)
        
        # Should find 2 faces
        self.assertEqual(len(faces), 2)
        
        # Check first face
        self.assertEqual(faces[0].bounding_box, (10, 90, 90, 10))
        self.assertGreater(faces[0].confidence, 0.0)
        
        # Verify face_locations was called with correct parameters
        mock_face_recognition.face_locations.assert_called_once_with(
            test_image,
            number_of_times_to_upsample=1,
            model='hog'
        )
    
    @patch('src.detectors.cpu_detector.CPUDetector._initialize_model')
    def test_run_inference_no_faces(self, mock_init):
        """Test face detection inference with no faces found."""
        detector = CPUDetector(self.config)
        
        # Mock face_recognition module
        mock_face_recognition = Mock()
        mock_face_recognition.face_locations.return_value = []
        detector._face_recognition = mock_face_recognition
        
        # Create test image
        if NUMPY_AVAILABLE:
            test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        else:
            test_image = np.zeros((200, 200, 3))
        
        # Run inference
        faces = detector._run_inference(test_image)
        
        # Should find no faces
        self.assertEqual(len(faces), 0)
    
    @patch('src.detectors.cpu_detector.CPUDetector._initialize_model')
    def test_run_inference_error_handling(self, mock_init):
        """Test error handling in inference."""
        detector = CPUDetector(self.config)
        
        # Mock face_recognition module to raise error
        mock_face_recognition = Mock()
        mock_face_recognition.face_locations.side_effect = Exception("Detection failed")
        detector._face_recognition = mock_face_recognition
        
        # Create test image
        if NUMPY_AVAILABLE:
            test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        else:
            test_image = np.zeros((200, 200, 3))
        
        # Run inference - should handle error gracefully
        faces = detector._run_inference(test_image)
        
        # Should return empty list on error
        self.assertEqual(len(faces), 0)
    
    @patch('src.detectors.cpu_detector.CPUDetector._initialize_model')
    def test_detect_faces_with_landmarks(self, mock_init):
        """Test face detection with landmarks."""
        detector = CPUDetector(self.config)
        
        # Mock face_recognition module
        mock_face_recognition = Mock()
        mock_face_recognition.face_locations.return_value = [(10, 90, 90, 10)]
        mock_face_recognition.face_landmarks.return_value = [
            {
                'chin': [(10, 90), (20, 90)],
                'left_eye': [(30, 40), (35, 40)],
                'right_eye': [(50, 40), (55, 40)]
            }
        ]
        detector._face_recognition = mock_face_recognition
        
        # Create test image
        if NUMPY_AVAILABLE:
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        else:
            test_image = np.zeros((100, 100, 3))
        
        # Run detection with landmarks
        faces = detector.detect_faces_with_landmarks(test_image)
        
        # Should find 1 face with landmarks
        self.assertEqual(len(faces), 1)
        self.assertIsNotNone(faces[0].landmarks)
        self.assertIn('chin', faces[0].landmarks)
        self.assertIn('left_eye', faces[0].landmarks)
    
    @patch('src.detectors.cpu_detector.CPUDetector._initialize_model')
    def test_confidence_score_hog(self, mock_init):
        """Test confidence score for HOG model."""
        config = self.config.copy()
        config['model'] = 'hog'
        detector = CPUDetector(config)
        
        # Mock face_recognition
        mock_face_recognition = Mock()
        mock_face_recognition.face_locations.return_value = [(10, 90, 90, 10)]
        detector._face_recognition = mock_face_recognition
        
        # Run inference
        if NUMPY_AVAILABLE:
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        else:
            test_image = np.zeros((100, 100, 3))
        
        faces = detector._run_inference(test_image)
        
        # HOG should have confidence around 0.90
        self.assertAlmostEqual(faces[0].confidence, 0.90, places=2)
    
    @patch('src.detectors.cpu_detector.CPUDetector._initialize_model')
    def test_confidence_score_cnn(self, mock_init):
        """Test confidence score for CNN model."""
        config = self.config.copy()
        config['model'] = 'cnn'
        detector = CPUDetector(config)
        
        # Mock face_recognition
        mock_face_recognition = Mock()
        mock_face_recognition.face_locations.return_value = [(10, 90, 90, 10)]
        detector._face_recognition = mock_face_recognition
        
        # Run inference
        if NUMPY_AVAILABLE:
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        else:
            test_image = np.zeros((100, 100, 3))
        
        faces = detector._run_inference(test_image)
        
        # CNN should have confidence around 0.95
        self.assertAlmostEqual(faces[0].confidence, 0.95, places=2)
    
    @patch('src.detectors.cpu_detector.CPUDetector._initialize_model')
    def test_cleanup(self, mock_init):
        """Test detector cleanup."""
        detector = CPUDetector(self.config)
        detector._face_recognition = Mock()
        
        # Call cleanup
        detector.cleanup()
        
        # face_recognition should be cleared
        self.assertIsNone(detector._face_recognition)


if __name__ == '__main__':
    unittest.main()
