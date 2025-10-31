"""
Platform detection and auto-configuration for different environments
"""

import os
import platform
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class PlatformDetector:
    """Detects platform and provides appropriate configurations"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.machine = platform.machine().lower()
        self.is_raspberry_pi = self._detect_raspberry_pi()
        self.is_apple_silicon = self._detect_apple_silicon()
        self.is_macos = self.system == 'darwin'
        self.is_linux = self.system == 'linux'
        self.is_windows = self.system == 'windows'
        self.is_development = os.getenv('DEVELOPMENT_MODE', 'false').lower() == 'true'
        
        logger.info(f"Platform detected: {self.system} {self.machine}")
        if self.is_raspberry_pi:
            logger.info("Running on Raspberry Pi")
        elif self.is_apple_silicon:
            logger.info("Running on macOS Apple Silicon")
        elif self.is_macos:
            logger.info("Running on macOS - Development mode")
        else:
            logger.info(f"Running on {self.system} - Mock mode")
    
    def _detect_raspberry_pi(self) -> bool:
        """Detect if running on Raspberry Pi"""
        try:
            # Check for Raspberry Pi specific files
            pi_files = [
                '/proc/device-tree/model',
                '/proc/cpuinfo'
            ]
            
            for file_path in pi_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read().lower()
                        if 'raspberry pi' in content:
                            return True
            
            # Check for GPIO directory
            if os.path.exists('/dev/gpiomem'):
                return True
                
        except Exception:
            pass
        
        return False
    
    def _detect_apple_silicon(self) -> bool:
        """Detect Apple Silicon (M1/M2/M3) processors"""
        if self.system != 'darwin':
            return False
        
        try:
            # Check for ARM64 architecture on macOS
            if 'arm64' in self.machine or 'aarch64' in self.machine:
                return True
            
            # Additional check using sysctl
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                cpu_brand = result.stdout.lower()
                if 'apple' in cpu_brand:
                    return True
        except Exception:
            pass
        
        return False
    
    def _detect_gpu_support(self) -> Dict[str, Any]:
        """Detect GPU support and capabilities"""
        gpu_info = {
            'has_gpu': False,
            'type': None,
            'cuda_available': False,
            'opencl_available': False
        }
        
        try:
            # Check for NVIDIA GPU (CUDA)
            if os.path.exists('/usr/local/cuda') or os.environ.get('CUDA_HOME'):
                gpu_info['has_gpu'] = True
                gpu_info['type'] = 'nvidia'
                gpu_info['cuda_available'] = True
            
            # Check for AMD GPU (ROCm)
            if os.path.exists('/opt/rocm'):
                gpu_info['has_gpu'] = True
                gpu_info['type'] = 'amd'
            
            # Check for Apple GPU (Metal)
            if self.is_apple_silicon:
                gpu_info['has_gpu'] = True
                gpu_info['type'] = 'apple'
        except Exception:
            pass
        
        return gpu_info
    
    def _detect_camera_support(self) -> bool:
        """Detect if camera hardware is available"""
        if self.is_raspberry_pi:
            # Check for Raspberry Pi camera
            return os.path.exists('/dev/video0') or os.path.exists('/dev/video1')
        elif self.is_linux:
            # Check for V4L2 devices
            return os.path.exists('/dev/video0')
        elif self.is_macos or self.is_windows:
            # Assume camera available in development
            return True
        return False
    
    def _detect_gpio_support(self) -> bool:
        """Detect if GPIO hardware is available"""
        return self.is_raspberry_pi and os.path.exists('/dev/gpiomem')
    
    def _get_memory_gb(self) -> float:
        """Get total system memory in GB"""
        try:
            if self.is_linux:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            kb = int(line.split()[1])
                            return round(kb / (1024 * 1024), 2)
            elif self.is_macos:
                result = subprocess.run(
                    ['sysctl', '-n', 'hw.memsize'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    bytes_mem = int(result.stdout.strip())
                    return round(bytes_mem / (1024 ** 3), 2)
        except Exception:
            pass
        
        return 0.0
    
    def get_platform_info(self) -> Dict[str, Any]:
        """Get comprehensive platform information"""
        return {
            "os": platform.system(),
            "os_version": platform.release(),
            "architecture": self.machine,
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "is_raspberry_pi": self.is_raspberry_pi,
            "is_apple_silicon": self.is_apple_silicon,
            "is_windows": self.is_windows,
            "is_linux": self.is_linux,
            "is_macos": self.is_macos,
            "has_gpu": self._detect_gpu_support()['has_gpu'],
            "gpu_info": self._detect_gpu_support(),
            "has_camera": self._detect_camera_support(),
            "has_gpio": self._detect_gpio_support(),
            "memory_gb": self._get_memory_gb(),
            "cpu_count": os.cpu_count() or 1,
            "is_development": self.is_development,
        }
    
    def get_recommended_installation_method(self) -> str:
        """Get platform-specific installation recommendations"""
        if self.is_raspberry_pi:
            return "apt_then_pip"  # Use system packages first
        elif self.is_apple_silicon:
            return "homebrew_then_pip"  # Use homebrew for system deps
        elif self.is_windows:
            return "conda_preferred"  # Conda handles Windows deps better
        else:
            return "pip_with_system_deps"  # Standard Linux approach
    
    def get_camera_config(self) -> dict:
        """Get camera configuration based on platform"""
        if self.is_raspberry_pi:
            return {
                'type': 'picamera',
                'use_picamera2': True,
                'device_id': 0,
                'resolution': (1280, 720),
                'mock': False
            }
        elif self.is_macos or self.is_development:
            return {
                'type': 'opencv',
                'use_picamera2': False,
                'device_id': 0,  # Default webcam
                'resolution': (1280, 720),
                'mock': False
            }
        else:
            return {
                'type': 'mock',
                'use_picamera2': False,
                'device_id': 0,
                'resolution': (640, 480),
                'mock': True
            }
    
    def get_gpio_config(self) -> dict:
        """Get GPIO configuration based on platform"""
        if self.is_raspberry_pi and not self.is_development:
            return {
                'use_real_gpio': True,
                'mock': False,
                'web_interface': False
            }
        else:
            return {
                'use_real_gpio': False,
                'mock': True,
                'web_interface': True  # Use web interface for testing
            }
    
    def get_deployment_config(self) -> dict:
        """Get deployment configuration"""
        if os.getenv('VERCEL'):
            return {
                'platform': 'vercel',
                'serverless': True,
                'static_files': True,
                'port': int(os.getenv('PORT', 3000))
            }
        elif os.getenv('RENDER'):
            return {
                'platform': 'render',
                'serverless': False,
                'static_files': True,
                'port': int(os.getenv('PORT', 10000))
            }
        elif os.getenv('RAILWAY'):
            return {
                'platform': 'railway',
                'serverless': False,
                'static_files': True,
                'port': int(os.getenv('PORT', 8080))
            }
        else:
            return {
                'platform': 'local',
                'serverless': False,
                'static_files': False,
                'port': int(os.getenv('PORT', 5000))
            }
    
    def should_use_mock(self, component: str) -> bool:
        """Determine if a component should use mock implementation"""
        if self.is_development:
            return True
        
        if component == 'gpio':
            return not self.is_raspberry_pi
        elif component == 'camera':
            return False  # Always try real camera first
        elif component == 'telegram':
            return False  # Always use real Telegram if configured
        
        return False
    
    def get_requirements_for_platform(self) -> list:
        """Get platform-specific requirements"""
        base_requirements = [
            'face_recognition>=1.3.0',
            'opencv-python>=4.8.0',
            'Pillow>=10.0.0',
            'numpy>=1.24.0',
            'requests>=2.31.0',
            'python-dateutil>=2.8.0',
            'pyyaml>=6.0.0',
            'python-telegram-bot>=20.6'
        ]
        
        if self.is_raspberry_pi:
            base_requirements.extend([
                'RPi.GPIO>=0.7.1',
                'picamera2>=0.3.12'
            ])
        
        if self.is_macos or self.is_development:
            base_requirements.extend([
                'flask>=2.3.0',
                'flask-cors>=4.0.0',
                'gunicorn>=21.0.0'
            ])
        
        return base_requirements


# Global platform detector instance
platform_detector = PlatformDetector()
