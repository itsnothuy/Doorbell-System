#!/usr/bin/env python3
"""
Test suite for GPU Detector

Tests for GPU-accelerated face detection implementation.
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
        float32 = float
        
        @staticmethod
        def zeros(shape, dtype=None):
            class MockArray:
                def __init__(self, shape):
                    self.shape = shape
                    self.size = shape[0] * shape[1] * (shape[2] if len(shape) > 2 else 1)
            return MockArray(shape)
        
        @staticmethod
        def random():
            class Random:
                @staticmethod
                def rand(*shape):
                    class MockArray:
                        def __init__(self, shape):
                            self.shape = shape
                        
                        def astype(self, dtype):
                            return self
                    return MockArray(shape)
            return Random()
        
        @staticmethod
        def mean(arr):
            return 0.0

from src.detectors.gpu_detector import GPUDetector
from src.detectors.base_detector import DetectorType, ModelType


class TestGPUDetector(unittest.TestCase):
    """Test suite for GPU detector."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'model': 'retinaface',
            'device': 'cuda:0',
            'batch_size': 4,
            'confidence_threshold': 0.5,
            'min_face_size': (30, 30),
            'max_face_size': (1000, 1000)
        }
    
    @patch('src.detectors.gpu_detector.logger')
    def test_is_available_with_cuda(self, mock_logger):
        """Test availability check when CUDA is available."""
        # Mock onnxruntime with CUDA support
        mock_ort = Mock()
        mock_ort.get_available_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        with patch.dict('sys.modules', {'onnxruntime': mock_ort}):
            available = GPUDetector.is_available()
            self.assertTrue(available)
    
    @patch('src.detectors.gpu_detector.logger')
    def test_is_available_without_cuda(self, mock_logger):
        """Test availability check when CUDA is not available."""
        # Mock onnxruntime without CUDA support
        mock_ort = Mock()
        mock_ort.get_available_providers.return_value = ['CPUExecutionProvider']
        
        with patch.dict('sys.modules', {'onnxruntime': mock_ort}):
            available = GPUDetector.is_available()
            self.assertFalse(available)
    
    @patch('src.detectors.gpu_detector.logger')
    def test_is_available_without_onnxruntime(self, mock_logger):
        """Test availability check when onnxruntime is not installed."""
        with patch('builtins.__import__', side_effect=ImportError('onnxruntime not found')):
            available = GPUDetector.is_available()
            self.assertFalse(available)
    
    @patch('src.detectors.gpu_detector.GPUDetector._initialize_model')
    def test_detector_type(self, mock_init):
        """Test that detector type is GPU."""
        detector = GPUDetector(self.config)
        self.assertEqual(detector.detector_type, DetectorType.GPU)
    
    @patch('src.detectors.gpu_detector.GPUDetector._initialize_model')
    def test_model_configuration(self, mock_init):
        """Test model configuration."""
        config = self.config.copy()
        config['model'] = 'retinaface'
        
        detector = GPUDetector(config)
        self.assertEqual(detector.model_name, 'retinaface')
        self.assertIn('retinaface', detector.SUPPORTED_MODELS)
    
    @patch('src.detectors.gpu_detector.GPUDetector._initialize_model')
    def test_device_configuration(self, mock_init):
        """Test device configuration."""
        config = self.config.copy()
        config['device'] = 'cuda:1'
        
        detector = GPUDetector(config)
        self.assertEqual(detector.device, 'cuda:1')
    
    @patch('src.detectors.gpu_detector.GPUDetector._initialize_model')
    def test_batch_size_configuration(self, mock_init):
        """Test batch size configuration."""
        config = self.config.copy()
        config['batch_size'] = 8
        
        detector = GPUDetector(config)
        self.assertEqual(detector.batch_size, 8)
    
    @patch('src.detectors.gpu_detector.GPUDetector._initialize_model')
    def test_unknown_model_defaults(self, mock_init):
        """Test that unknown model defaults to retinaface."""
        config = self.config.copy()
        config['model'] = 'unknown_model'
        
        detector = GPUDetector(config)
        self.assertEqual(detector.model_name, 'retinaface')
    
    def test_initialize_model_with_onnxruntime(self):
        """Test model initialization with onnxruntime."""
        # Mock onnxruntime and dependencies
        mock_ort = Mock()
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name='input')]
        mock_session.get_outputs.return_value = [Mock(name='output1'), Mock(name='output2')]
        mock_session.get_providers.return_value = ['CUDAExecutionProvider']
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = Mock()
        mock_ort.GraphOptimizationLevel = Mock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 'ORT_ENABLE_ALL'
        
        # Mock model manager
        mock_model_manager = Mock()
        mock_model_manager.get_model.return_value = Path('/tmp/model.onnx')
        
        with patch.dict('sys.modules', {'onnxruntime': mock_ort}):
            with patch('src.detectors.gpu_detector.ModelManager', return_value=mock_model_manager):
                with patch.object(GPUDetector, '_warmup_model'):
                    detector = GPUDetector(self.config)
                    
                    # Verify session was created
                    self.assertIsNotNone(detector.session)
                    self.assertIsNotNone(detector.input_name)
                    self.assertIsNotNone(detector.output_names)
    
    def test_initialize_model_without_onnxruntime(self):
        """Test model initialization fails without onnxruntime."""
        with patch('builtins.__import__', side_effect=ImportError('onnxruntime not found')):
            with self.assertRaises(RuntimeError):
                detector = GPUDetector(self.config)
    
    @patch('src.detectors.gpu_detector.GPUDetector._initialize_model')
    def test_run_inference(self, mock_init):
        """Test inference execution."""
        detector = GPUDetector(self.config)
        
        # Mock session
        mock_session = Mock()
        mock_session.run.return_value = [
            np.zeros((1, 10, 4)),  # boxes
            np.zeros((1, 10))      # scores
        ] if NUMPY_AVAILABLE else []
        detector.session = mock_session
        detector.input_name = 'input'
        detector.output_names = ['boxes', 'scores']
        
        # Create test image
        if NUMPY_AVAILABLE:
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            test_image = np.zeros((480, 640, 3))
        
        # Run inference
        with patch.object(detector, '_preprocess_image_for_model', return_value=test_image):
            with patch.object(detector, '_postprocess_outputs', return_value=[]):
                faces = detector._run_inference(test_image)
                
                # Should return results (empty in this mock)
                self.assertIsInstance(faces, list)
    
    @patch('src.detectors.gpu_detector.GPUDetector._initialize_model')
    def test_preprocessing(self, mock_init):
        """Test image preprocessing."""
        detector = GPUDetector(self.config)
        
        if not NUMPY_AVAILABLE:
            self.skipTest("NumPy not available")
        
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock cv2
        mock_cv2 = Mock()
        mock_cv2.resize.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
        
        with patch.dict('sys.modules', {'cv2': mock_cv2}):
            processed = detector._preprocess_image_for_model(test_image)
            
            # Should have correct shape (batch, channels, height, width)
            self.assertEqual(len(processed.shape), 4)
            self.assertEqual(processed.shape[0], 1)  # batch size
    
    @patch('src.detectors.gpu_detector.GPUDetector._initialize_model')
    def test_performance_metrics(self, mock_init):
        """Test performance metrics collection."""
        detector = GPUDetector(self.config)
        detector.inference_times = [0.01, 0.015, 0.012]
        
        metrics = detector.get_performance_metrics()
        
        # Should include GPU-specific metrics
        self.assertIn('device', metrics)
        self.assertIn('model', metrics)
        self.assertIn('backend', metrics)
        self.assertEqual(metrics['backend'], 'onnx')
        self.assertEqual(metrics['device'], 'cuda:0')
    
    @patch('src.detectors.gpu_detector.GPUDetector._initialize_model')
    def test_cleanup(self, mock_init):
        """Test detector cleanup."""
        detector = GPUDetector(self.config)
        detector.session = Mock()
        
        # Call cleanup
        detector.cleanup()
        
        # Session should be cleared
        self.assertIsNone(detector.session)
    
    @patch('src.detectors.gpu_detector.GPUDetector._initialize_model')
    def test_nms_application(self, mock_init):
        """Test Non-Maximum Suppression."""
        detector = GPUDetector(self.config)
        
        if not NUMPY_AVAILABLE:
            self.skipTest("NumPy not available")
        
        # Create test boxes and scores
        boxes = np.array([
            [0, 0, 100, 100],
            [10, 10, 110, 110],  # Overlapping with first
            [200, 200, 300, 300]  # Separate
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.85], dtype=np.float32)
        
        # Apply NMS
        keep_indices = detector._apply_nms(boxes, scores, threshold=0.5)
        
        # Should keep non-overlapping boxes
        self.assertIsInstance(keep_indices, list)
        self.assertGreater(len(keep_indices), 0)
    
    @patch('src.detectors.gpu_detector.GPUDetector._initialize_model')
    def test_error_handling_in_inference(self, mock_init):
        """Test error handling during inference."""
        detector = GPUDetector(self.config)
        
        # Mock session to raise error
        mock_session = Mock()
        mock_session.run.side_effect = Exception("Inference failed")
        detector.session = mock_session
        detector.input_name = 'input'
        detector.output_names = ['output']
        
        # Create test image
        if NUMPY_AVAILABLE:
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            test_image = np.zeros((480, 640, 3))
        
        # Run inference - should handle error gracefully
        with patch.object(detector, '_preprocess_image_for_model', return_value=test_image):
            faces = detector._run_inference(test_image)
            
            # Should return empty list on error
            self.assertEqual(len(faces), 0)


if __name__ == '__main__':
    unittest.main()
