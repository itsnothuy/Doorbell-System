"""
Platform detection and auto-configuration for different environments
"""

import os
import platform
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PlatformDetector:
    """Detects platform and provides appropriate configurations"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.machine = platform.machine().lower()
        self.is_raspberry_pi = self._detect_raspberry_pi()
        self.is_macos = self.system == 'darwin'
        self.is_linux = self.system == 'linux'
        self.is_windows = self.system == 'windows'
        self.is_development = os.getenv('DEVELOPMENT_MODE', 'false').lower() == 'true'
        
        logger.info(f"Platform detected: {self.system} {self.machine}")
        if self.is_raspberry_pi:
            logger.info("Running on Raspberry Pi")
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
