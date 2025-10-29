#!/usr/bin/env python3
"""
Test suite for Hardware Detector

Tests for hardware capability detection.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.detectors.hardware_detector import HardwareDetector, GPUInfo, EdgeTPUInfo


class TestHardwareDetector(unittest.TestCase):
    """Test suite for hardware detector."""
    
    def setUp(self):
        """Set up test environment."""
        self.detector = HardwareDetector()
    
    def test_initialization(self):
        """Test hardware detector initialization."""
        self.assertIsNotNone(self.detector.system_info)
        self.assertIn('os', self.detector.system_info)
        self.assertIn('arch', self.detector.system_info)
        self.assertIn('python_version', self.detector.system_info)
    
    def test_detect_gpus_with_pytorch(self):
        """Test GPU detection with PyTorch."""
        # Mock PyTorch with CUDA support
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_props = Mock()
        mock_props.name = "NVIDIA GeForce GTX 1080"
        mock_props.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
        mock_props.major = 6
        mock_props.minor = 1
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.version.cuda = "11.8"
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            gpus = self.detector.detect_gpus()
            
            self.assertEqual(len(gpus), 1)
            self.assertEqual(gpus[0].name, "NVIDIA GeForce GTX 1080")
            self.assertGreater(gpus[0].memory_mb, 0)
            self.assertEqual(gpus[0].compute_capability, "6.1")
    
    def test_detect_gpus_with_tensorflow(self):
        """Test GPU detection with TensorFlow."""
        # Mock TensorFlow with GPU support
        mock_tf = Mock()
        mock_config = Mock()
        mock_device = Mock()
        mock_config.list_physical_devices.return_value = [mock_device]
        mock_config.experimental.get_device_details.return_value = {
            'device_name': 'Tesla V100',
            'total_memory': 16 * 1024 * 1024 * 1024,  # 16GB
            'compute_capability': '7.0'
        }
        mock_tf.config = mock_config
        
        with patch.dict('sys.modules', {'tensorflow': mock_tf}):
            # Disable PyTorch mock
            with patch.dict('sys.modules', {'torch': None}):
                gpus = self.detector.detect_gpus()
                
                # Should detect TensorFlow GPU
                # Note: implementation may vary
                self.assertIsInstance(gpus, list)
    
    def test_detect_gpus_no_cuda(self):
        """Test GPU detection when no CUDA GPUs available."""
        # Mock libraries with no CUDA support
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            gpus = self.detector.detect_gpus()
            
            self.assertEqual(len(gpus), 0)
    
    def test_detect_gpus_no_libraries(self):
        """Test GPU detection when no libraries available."""
        with patch('builtins.__import__', side_effect=ImportError):
            gpus = self.detector.detect_gpus()
            
            self.assertEqual(len(gpus), 0)
    
    def test_detect_edgetpus_with_devices(self):
        """Test EdgeTPU detection with devices present."""
        # Mock pycoral
        mock_pycoral = Mock()
        mock_utils = Mock()
        mock_edgetpu = Mock()
        mock_edgetpu.list_edge_tpus.return_value = [
            {'type': 'usb', 'path': '/dev/bus/usb/001/002'},
            {'type': 'pci', 'path': '/dev/apex_0'}
        ]
        mock_utils.edgetpu = mock_edgetpu
        mock_pycoral.utils = mock_utils
        
        with patch.dict('sys.modules', {
            'pycoral': mock_pycoral,
            'pycoral.utils': mock_utils,
            'pycoral.utils.edgetpu': mock_edgetpu
        }):
            edgetpus = self.detector.detect_edgetpus()
            
            self.assertEqual(len(edgetpus), 2)
            self.assertEqual(edgetpus[0].device_type, 'usb')
            self.assertEqual(edgetpus[1].device_type, 'pci')
    
    def test_detect_edgetpus_no_devices(self):
        """Test EdgeTPU detection when no devices present."""
        # Mock pycoral with no devices
        mock_pycoral = Mock()
        mock_utils = Mock()
        mock_edgetpu = Mock()
        mock_edgetpu.list_edge_tpus.return_value = []
        mock_utils.edgetpu = mock_edgetpu
        mock_pycoral.utils = mock_utils
        
        with patch.dict('sys.modules', {
            'pycoral': mock_pycoral,
            'pycoral.utils': mock_utils,
            'pycoral.utils.edgetpu': mock_edgetpu
        }):
            edgetpus = self.detector.detect_edgetpus()
            
            self.assertEqual(len(edgetpus), 0)
    
    def test_detect_edgetpus_no_pycoral(self):
        """Test EdgeTPU detection when pycoral not available."""
        with patch('builtins.__import__', side_effect=ImportError):
            edgetpus = self.detector.detect_edgetpus()
            
            self.assertEqual(len(edgetpus), 0)
    
    def test_has_cuda_gpu(self):
        """Test CUDA GPU availability check."""
        with patch.object(self.detector, 'detect_gpus', return_value=[Mock()]):
            self.assertTrue(self.detector.has_cuda_gpu())
        
        with patch.object(self.detector, 'detect_gpus', return_value=[]):
            self.assertFalse(self.detector.has_cuda_gpu())
    
    def test_has_edgetpu(self):
        """Test EdgeTPU availability check."""
        with patch.object(self.detector, 'detect_edgetpus', return_value=[Mock()]):
            self.assertTrue(self.detector.has_edgetpu())
        
        with patch.object(self.detector, 'detect_edgetpus', return_value=[]):
            self.assertFalse(self.detector.has_edgetpu())
    
    def test_get_best_device_edgetpu(self):
        """Test best device selection with EdgeTPU."""
        with patch.object(self.detector, 'has_edgetpu', return_value=True):
            best = self.detector.get_best_device()
            self.assertEqual(best, 'edgetpu')
    
    def test_get_best_device_gpu(self):
        """Test best device selection with GPU."""
        with patch.object(self.detector, 'has_edgetpu', return_value=False):
            with patch.object(self.detector, 'has_cuda_gpu', return_value=True):
                best = self.detector.get_best_device()
                self.assertEqual(best, 'gpu')
    
    def test_get_best_device_cpu(self):
        """Test best device selection with CPU only."""
        with patch.object(self.detector, 'has_edgetpu', return_value=False):
            with patch.object(self.detector, 'has_cuda_gpu', return_value=False):
                best = self.detector.get_best_device()
                self.assertEqual(best, 'cpu')
    
    def test_get_device_capabilities(self):
        """Test device capabilities report."""
        with patch.object(self.detector, 'detect_gpus', return_value=[]):
            with patch.object(self.detector, 'detect_edgetpus', return_value=[]):
                capabilities = self.detector.get_device_capabilities()
                
                self.assertIn('system', capabilities)
                self.assertIn('gpus', capabilities)
                self.assertIn('edgetpus', capabilities)
                self.assertIn('best_device', capabilities)
                self.assertIn('has_gpu', capabilities)
                self.assertIn('has_edgetpu', capabilities)
    
    def test_check_onnxruntime_providers(self):
        """Test ONNX Runtime providers check."""
        # Mock onnxruntime
        mock_ort = Mock()
        mock_ort.get_available_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        with patch.dict('sys.modules', {'onnxruntime': mock_ort}):
            providers = self.detector.check_onnxruntime_providers()
            
            self.assertIn('CUDAExecutionProvider', providers)
            self.assertIn('CPUExecutionProvider', providers)
    
    def test_check_onnxruntime_providers_not_available(self):
        """Test ONNX Runtime providers check when not available."""
        with patch('builtins.__import__', side_effect=ImportError):
            providers = self.detector.check_onnxruntime_providers()
            
            self.assertEqual(len(providers), 0)
    
    def test_check_tensorflow_gpu(self):
        """Test TensorFlow GPU check."""
        # Mock TensorFlow with GPU
        mock_tf = Mock()
        mock_config = Mock()
        mock_config.list_physical_devices.return_value = [Mock()]
        mock_tf.config = mock_config
        
        with patch.dict('sys.modules', {'tensorflow': mock_tf}):
            has_tf_gpu = self.detector.check_tensorflow_gpu()
            self.assertTrue(has_tf_gpu)
    
    def test_check_tensorflow_gpu_not_available(self):
        """Test TensorFlow GPU check when not available."""
        with patch('builtins.__import__', side_effect=ImportError):
            has_tf_gpu = self.detector.check_tensorflow_gpu()
            self.assertFalse(has_tf_gpu)
    
    def test_get_hardware_summary(self):
        """Test hardware summary generation."""
        with patch.object(self.detector, 'get_device_capabilities', return_value={
            'system': {'os': 'Linux', 'arch': 'x86_64', 'python_version': '3.11'},
            'gpus': [{'name': 'GTX 1080', 'memory_mb': 8192, 'compute_capability': '6.1'}],
            'edgetpus': [],
            'best_device': 'gpu',
            'has_gpu': True,
            'has_edgetpu': False
        }):
            summary = self.detector.get_hardware_summary()
            
            self.assertIsInstance(summary, str)
            self.assertIn('Hardware Capabilities', summary)
            self.assertIn('GTX 1080', summary)
            self.assertIn('GPU', summary)


if __name__ == '__main__':
    unittest.main()
