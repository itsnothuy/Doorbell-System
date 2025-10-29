#!/usr/bin/env python3
"""
Test suite for EdgeTPU Detector

Tests for Coral EdgeTPU face detection implementation.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Try to import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class np:
        uint8 = int
        float32 = float
        
        @staticmethod
        def zeros(shape, dtype=None):
            class MockArray:
                def __init__(self, shape):
                    self.shape = shape
                    self.size = shape[0] * shape[1] * (shape[2] if len(shape) > 2 else 1)
            return MockArray(shape)
        
        @staticmethod
        def mean(arr):
            return 0.0

from src.detectors.edgetpu_detector import EdgeTPUDetector
from src.detectors.base_detector import DetectorType


class TestEdgeTPUDetector(unittest.TestCase):
    """Test suite for EdgeTPU detector."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'model': 'mobilenet_face',
            'confidence_threshold': 0.6,
            'min_face_size': (30, 30),
            'max_face_size': (1000, 1000)
        }
    
    @patch('src.detectors.edgetpu_detector.logger')
    def test_is_available_with_edgetpu(self, mock_logger):
        """Test availability check when EdgeTPU is available."""
        # Mock pycoral with available device
        mock_pycoral = Mock()
        mock_utils = Mock()
        mock_edgetpu = Mock()
        mock_edgetpu.list_edge_tpus.return_value = [{'type': 'usb', 'path': '/dev/bus/usb/001/002'}]
        mock_utils.edgetpu = mock_edgetpu
        mock_pycoral.utils = mock_utils
        
        with patch.dict('sys.modules', {'pycoral': mock_pycoral, 'pycoral.utils': mock_utils, 'pycoral.utils.edgetpu': mock_edgetpu}):
            available = EdgeTPUDetector.is_available()
            self.assertTrue(available)
    
    @patch('src.detectors.edgetpu_detector.logger')
    def test_is_available_without_edgetpu_device(self, mock_logger):
        """Test availability check when no EdgeTPU device is found."""
        # Mock pycoral with no devices
        mock_pycoral = Mock()
        mock_utils = Mock()
        mock_edgetpu = Mock()
        mock_edgetpu.list_edge_tpus.return_value = []
        mock_utils.edgetpu = mock_edgetpu
        mock_pycoral.utils = mock_utils
        
        with patch.dict('sys.modules', {'pycoral': mock_pycoral, 'pycoral.utils': mock_utils, 'pycoral.utils.edgetpu': mock_edgetpu}):
            available = EdgeTPUDetector.is_available()
            self.assertFalse(available)
    
    @patch('src.detectors.edgetpu_detector.logger')
    def test_is_available_without_pycoral(self, mock_logger):
        """Test availability check when pycoral is not installed."""
        with patch('builtins.__import__', side_effect=ImportError('pycoral not found')):
            available = EdgeTPUDetector.is_available()
            self.assertFalse(available)
    
    @patch('src.detectors.edgetpu_detector.EdgeTPUDetector._initialize_model')
    def test_detector_type(self, mock_init):
        """Test that detector type is EdgeTPU."""
        detector = EdgeTPUDetector(self.config)
        self.assertEqual(detector.detector_type, DetectorType.EDGETPU)
    
    @patch('src.detectors.edgetpu_detector.EdgeTPUDetector._initialize_model')
    def test_model_configuration(self, mock_init):
        """Test model configuration."""
        config = self.config.copy()
        config['model'] = 'mobilenet_face'
        
        detector = EdgeTPUDetector(config)
        self.assertEqual(detector.model_name, 'mobilenet_face')
        self.assertIn('mobilenet_face', detector.EDGETPU_MODELS)
    
    @patch('src.detectors.edgetpu_detector.EdgeTPUDetector._initialize_model')
    def test_device_path_configuration(self, mock_init):
        """Test device path configuration."""
        config = self.config.copy()
        config['device_path'] = '/dev/bus/usb/001/002'
        
        detector = EdgeTPUDetector(config)
        self.assertEqual(detector.device_path, '/dev/bus/usb/001/002')
    
    @patch('src.detectors.edgetpu_detector.EdgeTPUDetector._initialize_model')
    def test_monitoring_configuration(self, mock_init):
        """Test monitoring configuration."""
        config = self.config.copy()
        config['enable_monitoring'] = True
        config['temperature_limit'] = 90.0
        
        detector = EdgeTPUDetector(config)
        self.assertTrue(detector.enable_monitoring)
        self.assertEqual(detector.temperature_limit, 90.0)
    
    @patch('src.detectors.edgetpu_detector.EdgeTPUDetector._initialize_model')
    def test_unknown_model_defaults(self, mock_init):
        """Test that unknown model defaults to mobilenet_face."""
        config = self.config.copy()
        config['model'] = 'unknown_model'
        
        detector = EdgeTPUDetector(config)
        self.assertEqual(detector.model_name, 'mobilenet_face')
    
    def test_initialize_model_with_pycoral(self):
        """Test model initialization with pycoral."""
        # Mock pycoral and dependencies
        mock_interpreter = Mock()
        mock_input_details = {'index': 0, 'shape': [1, 224, 224, 3]}
        mock_output_details = [{'index': 1, 'name': 'output'}]
        mock_interpreter.get_input_details.return_value = [mock_input_details]
        mock_interpreter.get_output_details.return_value = mock_output_details
        mock_interpreter.allocate_tensors.return_value = None
        
        mock_make_interpreter = Mock(return_value=mock_interpreter)
        
        # Mock model manager
        mock_model_manager = Mock()
        mock_model_manager.get_model.return_value = Path('/tmp/model.tflite')
        
        mock_pycoral = Mock()
        mock_utils = Mock()
        mock_edgetpu = Mock()
        mock_edgetpu.make_interpreter = mock_make_interpreter
        mock_utils.edgetpu = mock_edgetpu
        mock_pycoral.utils = mock_utils
        
        with patch.dict('sys.modules', {'pycoral': mock_pycoral, 'pycoral.utils': mock_utils, 'pycoral.utils.edgetpu': mock_edgetpu}):
            with patch('src.detectors.edgetpu_detector.ModelManager', return_value=mock_model_manager):
                with patch.object(EdgeTPUDetector, '_warmup_model'):
                    detector = EdgeTPUDetector(self.config)
                    
                    # Verify interpreter was created
                    self.assertIsNotNone(detector.interpreter)
                    self.assertIsNotNone(detector.input_details)
                    self.assertIsNotNone(detector.output_details)
    
    def test_initialize_model_without_pycoral(self):
        """Test model initialization fails without pycoral."""
        with patch('builtins.__import__', side_effect=ImportError('pycoral not found')):
            with self.assertRaises(RuntimeError):
                detector = EdgeTPUDetector(self.config)
    
    @patch('src.detectors.edgetpu_detector.EdgeTPUDetector._initialize_model')
    def test_run_inference(self, mock_init):
        """Test inference execution."""
        detector = EdgeTPUDetector(self.config)
        
        # Mock interpreter
        mock_interpreter = Mock()
        detector.interpreter = mock_interpreter
        detector.input_details = {'index': 0}
        detector.output_details = [{'index': 1}]
        
        # Create test image
        if NUMPY_AVAILABLE:
            test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            test_image = np.zeros((224, 224, 3))
        
        # Run inference
        with patch.object(detector, '_preprocess_image_for_edgetpu', return_value=test_image):
            with patch.object(detector, '_run_tflite_inference', return_value=[]):
                with patch.object(detector, '_postprocess_edgetpu_outputs', return_value=[]):
                    faces = detector._run_inference(test_image)
                    
                    # Should return results (empty in this mock)
                    self.assertIsInstance(faces, list)
    
    @patch('src.detectors.edgetpu_detector.EdgeTPUDetector._initialize_model')
    def test_preprocessing_quantized(self, mock_init):
        """Test image preprocessing for quantized model."""
        detector = EdgeTPUDetector(self.config)
        
        if not NUMPY_AVAILABLE:
            self.skipTest("NumPy not available")
        
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock cv2
        mock_cv2 = Mock()
        mock_cv2.resize.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        
        with patch.dict('sys.modules', {'cv2': mock_cv2}):
            processed = detector._preprocess_image_for_edgetpu(test_image)
            
            # Should have batch dimension
            self.assertEqual(len(processed.shape), 4)
            self.assertEqual(processed.shape[0], 1)
    
    @patch('src.detectors.edgetpu_detector.EdgeTPUDetector._initialize_model')
    def test_tflite_inference(self, mock_init):
        """Test TensorFlow Lite inference."""
        detector = EdgeTPUDetector(self.config)
        
        # Mock interpreter
        mock_interpreter = Mock()
        mock_interpreter.get_tensor.return_value = np.array([]) if NUMPY_AVAILABLE else []
        detector.interpreter = mock_interpreter
        detector.input_details = {'index': 0}
        detector.output_details = [{'index': 1}, {'index': 2}]
        
        if NUMPY_AVAILABLE:
            test_input = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        else:
            test_input = np.zeros((1, 224, 224, 3))
        
        # Run inference
        outputs = detector._run_tflite_inference(test_input)
        
        # Should return outputs
        self.assertIsInstance(outputs, list)
        mock_interpreter.invoke.assert_called_once()
    
    @patch('src.detectors.edgetpu_detector.EdgeTPUDetector._initialize_model')
    def test_performance_metrics(self, mock_init):
        """Test performance metrics collection."""
        detector = EdgeTPUDetector(self.config)
        detector.inference_times = [0.005, 0.008, 0.006]
        detector.throttle_events = 2
        detector.temperature_readings = [60.0, 65.0, 70.0]
        
        metrics = detector.get_performance_metrics()
        
        # Should include EdgeTPU-specific metrics
        self.assertIn('device_path', metrics)
        self.assertIn('model', metrics)
        self.assertIn('backend', metrics)
        self.assertIn('quantized', metrics)
        self.assertIn('throttle_events', metrics)
        self.assertEqual(metrics['backend'], 'tflite')
        self.assertEqual(metrics['throttle_events'], 2)
    
    @patch('src.detectors.edgetpu_detector.EdgeTPUDetector._initialize_model')
    def test_cleanup(self, mock_init):
        """Test detector cleanup."""
        detector = EdgeTPUDetector(self.config)
        detector.interpreter = Mock()
        
        # Call cleanup
        detector.cleanup()
        
        # Interpreter should be cleared
        self.assertIsNone(detector.interpreter)
    
    @patch('src.detectors.edgetpu_detector.EdgeTPUDetector._initialize_model')
    def test_temperature_monitoring(self, mock_init):
        """Test temperature monitoring."""
        config = self.config.copy()
        config['enable_monitoring'] = True
        
        detector = EdgeTPUDetector(config)
        
        # Should not raise error
        detector._check_temperature()
    
    @patch('src.detectors.edgetpu_detector.EdgeTPUDetector._initialize_model')
    def test_error_handling_in_inference(self, mock_init):
        """Test error handling during inference."""
        detector = EdgeTPUDetector(self.config)
        
        # Mock interpreter to raise error
        mock_interpreter = Mock()
        mock_interpreter.invoke.side_effect = Exception("Inference failed")
        detector.interpreter = mock_interpreter
        detector.input_details = {'index': 0}
        detector.output_details = [{'index': 1}]
        
        # Create test image
        if NUMPY_AVAILABLE:
            test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            test_image = np.zeros((224, 224, 3))
        
        # Run inference - should handle error gracefully
        with patch.object(detector, '_preprocess_image_for_edgetpu', return_value=test_image):
            faces = detector._run_inference(test_image)
            
            # Should return empty list on error
            self.assertEqual(len(faces), 0)


if __name__ == '__main__':
    unittest.main()
