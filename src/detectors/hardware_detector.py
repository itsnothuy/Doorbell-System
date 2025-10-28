#!/usr/bin/env python3
"""
Hardware Detector - Capability Detection for Accelerators

Detects available hardware acceleration capabilities (GPU, EdgeTPU)
and provides system information for optimal detector selection.
"""

import logging
import platform
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU device information."""
    name: str
    memory_mb: int
    compute_capability: Optional[str] = None
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None


@dataclass
class EdgeTPUInfo:
    """EdgeTPU device information."""
    device_type: str  # 'usb' or 'pci'
    device_path: str
    temperature: Optional[float] = None


class HardwareDetector:
    """
    Detects available hardware acceleration capabilities.
    
    Features:
    - CUDA GPU detection
    - Coral EdgeTPU detection
    - System resource information
    - Hardware capability reporting
    """
    
    def __init__(self):
        """Initialize hardware detector."""
        self.system_info = self._get_system_info()
        logger.debug(f"System: {self.system_info['os']} {self.system_info['arch']}")
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get basic system information."""
        return {
            'os': platform.system(),
            'os_version': platform.version(),
            'arch': platform.machine(),
            'python_version': platform.python_version()
        }
    
    def detect_gpus(self) -> List[GPUInfo]:
        """
        Detect available NVIDIA GPUs with CUDA support.
        
        Returns:
            List of GPU information objects
        """
        gpus = []
        
        # Try PyTorch detection
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info = GPUInfo(
                        name=props.name,
                        memory_mb=props.total_memory // (1024 * 1024),
                        compute_capability=f"{props.major}.{props.minor}",
                        cuda_version=torch.version.cuda,
                        driver_version=None
                    )
                    gpus.append(gpu_info)
                    logger.info(f"Detected GPU {i}: {gpu_info.name} ({gpu_info.memory_mb}MB)")
                return gpus
        except ImportError:
            logger.debug("PyTorch not available for GPU detection")
        except Exception as e:
            logger.debug(f"PyTorch GPU detection failed: {e}")
        
        # Try TensorFlow detection
        try:
            import tensorflow as tf
            physical_devices = tf.config.list_physical_devices('GPU')
            for i, device in enumerate(physical_devices):
                # Get device details
                device_details = tf.config.experimental.get_device_details(device)
                gpu_info = GPUInfo(
                    name=device_details.get('device_name', f'GPU {i}'),
                    memory_mb=device_details.get('total_memory', 0) // (1024 * 1024),
                    compute_capability=device_details.get('compute_capability', None),
                    cuda_version=None,
                    driver_version=None
                )
                gpus.append(gpu_info)
                logger.info(f"Detected GPU {i}: {gpu_info.name}")
            return gpus
        except ImportError:
            logger.debug("TensorFlow not available for GPU detection")
        except Exception as e:
            logger.debug(f"TensorFlow GPU detection failed: {e}")
        
        # Try pynvml (NVIDIA Management Library)
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_info = GPUInfo(
                    name=name.decode() if isinstance(name, bytes) else name,
                    memory_mb=memory.total // (1024 * 1024),
                    driver_version=pynvml.nvmlSystemGetDriverVersion().decode()
                )
                gpus.append(gpu_info)
                logger.info(f"Detected GPU {i}: {gpu_info.name} ({gpu_info.memory_mb}MB)")
            
            pynvml.nvmlShutdown()
            return gpus
        except ImportError:
            logger.debug("pynvml not available for GPU detection")
        except Exception as e:
            logger.debug(f"pynvml GPU detection failed: {e}")
        
        logger.debug("No CUDA GPUs detected")
        return gpus
    
    def detect_edgetpus(self) -> List[EdgeTPUInfo]:
        """
        Detect available Coral EdgeTPU devices.
        
        Returns:
            List of EdgeTPU information objects
        """
        edgetpus = []
        
        try:
            from pycoral.utils.edgetpu import list_edge_tpus
            
            devices = list_edge_tpus()
            for device in devices:
                device_type = device.get('type', 'unknown')
                device_path = device.get('path', 'unknown')
                
                edgetpu_info = EdgeTPUInfo(
                    device_type=device_type,
                    device_path=device_path
                )
                edgetpus.append(edgetpu_info)
                logger.info(f"Detected EdgeTPU: {device_type} at {device_path}")
            
            return edgetpus
            
        except ImportError:
            logger.debug("pycoral not available for EdgeTPU detection")
        except Exception as e:
            logger.debug(f"EdgeTPU detection failed: {e}")
        
        logger.debug("No EdgeTPU devices detected")
        return edgetpus
    
    def has_cuda_gpu(self) -> bool:
        """
        Check if CUDA-capable GPU is available.
        
        Returns:
            True if CUDA GPU available
        """
        gpus = self.detect_gpus()
        return len(gpus) > 0
    
    def has_edgetpu(self) -> bool:
        """
        Check if EdgeTPU device is available.
        
        Returns:
            True if EdgeTPU available
        """
        edgetpus = self.detect_edgetpus()
        return len(edgetpus) > 0
    
    def get_best_device(self) -> str:
        """
        Determine the best available device for face detection.
        
        Priority: EdgeTPU > GPU > CPU
        
        Returns:
            Device type string ('edgetpu', 'gpu', or 'cpu')
        """
        if self.has_edgetpu():
            logger.info("Best device: EdgeTPU")
            return 'edgetpu'
        elif self.has_cuda_gpu():
            logger.info("Best device: GPU")
            return 'gpu'
        else:
            logger.info("Best device: CPU (no acceleration available)")
            return 'cpu'
    
    def get_device_capabilities(self) -> Dict[str, Any]:
        """
        Get comprehensive device capability report.
        
        Returns:
            Dictionary containing all hardware information
        """
        gpus = self.detect_gpus()
        edgetpus = self.detect_edgetpus()
        
        return {
            'system': self.system_info,
            'gpus': [
                {
                    'name': gpu.name,
                    'memory_mb': gpu.memory_mb,
                    'compute_capability': gpu.compute_capability,
                    'cuda_version': gpu.cuda_version,
                    'driver_version': gpu.driver_version
                }
                for gpu in gpus
            ],
            'edgetpus': [
                {
                    'type': tpu.device_type,
                    'path': tpu.device_path,
                    'temperature': tpu.temperature
                }
                for tpu in edgetpus
            ],
            'best_device': self.get_best_device(),
            'has_gpu': len(gpus) > 0,
            'has_edgetpu': len(edgetpus) > 0
        }
    
    def check_onnxruntime_providers(self) -> List[str]:
        """
        Check available ONNX Runtime execution providers.
        
        Returns:
            List of available provider names
        """
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            logger.debug(f"ONNX Runtime providers: {providers}")
            return providers
        except ImportError:
            logger.debug("ONNX Runtime not available")
            return []
        except Exception as e:
            logger.debug(f"Failed to get ONNX Runtime providers: {e}")
            return []
    
    def check_tensorflow_gpu(self) -> bool:
        """
        Check if TensorFlow GPU support is available.
        
        Returns:
            True if TensorFlow can access GPU
        """
        try:
            import tensorflow as tf
            physical_devices = tf.config.list_physical_devices('GPU')
            has_gpu = len(physical_devices) > 0
            logger.debug(f"TensorFlow GPU support: {has_gpu}")
            return has_gpu
        except ImportError:
            logger.debug("TensorFlow not available")
            return False
        except Exception as e:
            logger.debug(f"TensorFlow GPU check failed: {e}")
            return False
    
    def get_hardware_summary(self) -> str:
        """
        Get human-readable hardware summary.
        
        Returns:
            Multi-line string describing hardware capabilities
        """
        capabilities = self.get_device_capabilities()
        
        lines = [
            "Hardware Capabilities:",
            f"  System: {capabilities['system']['os']} {capabilities['system']['arch']}",
            f"  Python: {capabilities['system']['python_version']}",
            ""
        ]
        
        # GPU information
        if capabilities['has_gpu']:
            lines.append("  GPUs:")
            for gpu in capabilities['gpus']:
                lines.append(f"    - {gpu['name']} ({gpu['memory_mb']}MB)")
                if gpu['compute_capability']:
                    lines.append(f"      Compute: {gpu['compute_capability']}")
        else:
            lines.append("  GPUs: None detected")
        
        # EdgeTPU information
        if capabilities['has_edgetpu']:
            lines.append("  EdgeTPUs:")
            for tpu in capabilities['edgetpus']:
                lines.append(f"    - {tpu['type']} at {tpu['path']}")
        else:
            lines.append("  EdgeTPUs: None detected")
        
        lines.append("")
        lines.append(f"  Recommended device: {capabilities['best_device'].upper()}")
        
        return "\n".join(lines)


def print_hardware_info():
    """Convenience function to print hardware information."""
    detector = HardwareDetector()
    print(detector.get_hardware_summary())


if __name__ == '__main__':
    # Run hardware detection when executed directly
    print_hardware_info()
