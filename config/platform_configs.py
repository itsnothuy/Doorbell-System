#!/usr/bin/env python3
"""
Platform-specific configuration management.

This module provides platform-specific configurations for optimal performance
and compatibility across different target platforms.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class PlatformConfigurations:
    """Platform-specific configuration management."""
    
    # Platform-specific default configurations
    CONFIGS = {
        "raspberry_pi": {
            "memory_limit_mb": 1024,
            "worker_processes": 1,
            "camera_backend": "picamera2",
            "gpio_backend": "RPi.GPIO",
            "face_detection_model": "hog",  # CPU efficient
            "opencv_backend": "opencv-python-headless",
            "installation_method": "apt_first",
            "enable_swap": True,
            "low_memory_mode": True,
            "image_quality": 85,
            "max_concurrent_operations": 1,
            "cache_size_mb": 128,
            "enable_hardware_acceleration": True,
        },
        
        "apple_silicon": {
            "memory_limit_mb": 4096,
            "worker_processes": 4,
            "camera_backend": "opencv",
            "gpio_backend": "mock",
            "face_detection_model": "hog",  # dlib issues on M1
            "opencv_backend": "opencv-python",
            "installation_method": "homebrew_first",
            "environment_setup": "arm64_optimized",
            "enable_metal_acceleration": True,
            "image_quality": 95,
            "max_concurrent_operations": 4,
            "cache_size_mb": 512,
            "enable_hardware_acceleration": True,
        },
        
        "macos_intel": {
            "memory_limit_mb": 2048,
            "worker_processes": 2,
            "camera_backend": "opencv",
            "gpio_backend": "mock",
            "face_detection_model": "cnn",  # Can use CNN
            "opencv_backend": "opencv-python",
            "installation_method": "homebrew_first",
            "image_quality": 95,
            "max_concurrent_operations": 2,
            "cache_size_mb": 256,
            "enable_hardware_acceleration": True,
        },
        
        "ubuntu_x86": {
            "memory_limit_mb": 2048,
            "worker_processes": 2,
            "camera_backend": "opencv",
            "gpio_backend": "mock",
            "face_detection_model": "cnn",  # Can use CNN
            "opencv_backend": "opencv-python",
            "installation_method": "apt_then_pip",
            "image_quality": 95,
            "max_concurrent_operations": 2,
            "cache_size_mb": 256,
            "enable_hardware_acceleration": True,
        },
        
        "windows": {
            "memory_limit_mb": 2048,
            "worker_processes": 2,
            "camera_backend": "opencv",
            "gpio_backend": "mock",
            "face_detection_model": "hog",
            "opencv_backend": "opencv-python",
            "installation_method": "conda_preferred",
            "path_separator": "\\",
            "image_quality": 90,
            "max_concurrent_operations": 2,
            "cache_size_mb": 256,
            "enable_hardware_acceleration": False,
        },
        
        "docker": {
            "memory_limit_mb": 2048,
            "worker_processes": 2,
            "camera_backend": "opencv",
            "gpio_backend": "mock",
            "face_detection_model": "hog",
            "opencv_backend": "opencv-python-headless",
            "installation_method": "pip",
            "image_quality": 85,
            "max_concurrent_operations": 2,
            "cache_size_mb": 256,
            "enable_hardware_acceleration": False,
        },
    }
    
    @classmethod
    def get_config_for_platform(cls, platform_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific platform type.
        
        Args:
            platform_type: Platform identifier (e.g., 'raspberry_pi', 'apple_silicon')
            
        Returns:
            Platform-specific configuration dictionary
        """
        config = cls.CONFIGS.get(platform_type, cls.CONFIGS["ubuntu_x86"])
        logger.info(f"Loaded configuration for platform: {platform_type}")
        return config.copy()
    
    @classmethod
    def get_config(cls, platform_detector=None) -> Dict[str, Any]:
        """
        Get configuration for current platform.
        
        Args:
            platform_detector: Optional PlatformDetector instance
            
        Returns:
            Platform-specific configuration dictionary
        """
        if platform_detector is None:
            from src.platform_detector import platform_detector as detector
            platform_detector = detector
        
        # Determine platform type
        if platform_detector.is_raspberry_pi:
            platform_type = "raspberry_pi"
        elif platform_detector.is_apple_silicon:
            platform_type = "apple_silicon"
        elif platform_detector.is_macos:
            platform_type = "macos_intel"
        elif platform_detector.is_windows:
            platform_type = "windows"
        elif platform_detector.is_linux:
            platform_type = "ubuntu_x86"
        else:
            platform_type = "ubuntu_x86"  # Default fallback
        
        config = cls.get_config_for_platform(platform_type)
        
        # Add platform detection info
        config["platform_type"] = platform_type
        config["detected_os"] = platform_detector.system
        config["detected_architecture"] = platform_detector.machine
        
        # Adjust for development mode
        if platform_detector.is_development:
            config["gpio_backend"] = "mock"
            config["enable_debug_logging"] = True
        
        return config
    
    @classmethod
    def get_memory_config(cls, available_memory_mb: float) -> Dict[str, Any]:
        """
        Get memory-optimized configuration based on available memory.
        
        Args:
            available_memory_mb: Available memory in megabytes
            
        Returns:
            Memory-optimized configuration
        """
        config = {}
        
        if available_memory_mb < 1024:
            # Very low memory (< 1GB)
            config["memory_limit_mb"] = 512
            config["worker_processes"] = 1
            config["cache_size_mb"] = 64
            config["low_memory_mode"] = True
            config["image_quality"] = 75
            config["max_concurrent_operations"] = 1
        elif available_memory_mb < 2048:
            # Low memory (1-2GB)
            config["memory_limit_mb"] = 1024
            config["worker_processes"] = 1
            config["cache_size_mb"] = 128
            config["low_memory_mode"] = True
            config["image_quality"] = 85
            config["max_concurrent_operations"] = 1
        elif available_memory_mb < 4096:
            # Medium memory (2-4GB)
            config["memory_limit_mb"] = 2048
            config["worker_processes"] = 2
            config["cache_size_mb"] = 256
            config["low_memory_mode"] = False
            config["image_quality"] = 90
            config["max_concurrent_operations"] = 2
        else:
            # High memory (4GB+)
            config["memory_limit_mb"] = 4096
            config["worker_processes"] = 4
            config["cache_size_mb"] = 512
            config["low_memory_mode"] = False
            config["image_quality"] = 95
            config["max_concurrent_operations"] = 4
        
        return config
    
    @classmethod
    def get_optimized_config(cls, platform_detector=None) -> Dict[str, Any]:
        """
        Get optimized configuration based on platform and available resources.
        
        Args:
            platform_detector: Optional PlatformDetector instance
            
        Returns:
            Optimized configuration dictionary
        """
        if platform_detector is None:
            from src.platform_detector import platform_detector as detector
            platform_detector = detector
        
        # Start with platform-specific config
        config = cls.get_config(platform_detector)
        
        # Get platform info for optimization
        platform_info = platform_detector.get_platform_info()
        
        # Optimize based on available memory
        if platform_info.get("memory_gb", 0) > 0:
            memory_mb = platform_info["memory_gb"] * 1024
            memory_config = cls.get_memory_config(memory_mb)
            config.update(memory_config)
        
        # Optimize based on CPU count
        cpu_count = platform_info.get("cpu_count", 1)
        if cpu_count > config.get("worker_processes", 1):
            config["worker_processes"] = min(cpu_count, 4)
        
        # Optimize based on GPU availability
        if platform_info.get("has_gpu", False):
            gpu_info = platform_info.get("gpu_info", {})
            if gpu_info.get("cuda_available"):
                config["enable_gpu_acceleration"] = True
                config["gpu_backend"] = "cuda"
            elif gpu_info.get("type") == "apple":
                config["enable_gpu_acceleration"] = True
                config["gpu_backend"] = "metal"
        
        return config
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """
        Validate platform configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = [
            "memory_limit_mb",
            "worker_processes",
            "camera_backend",
            "gpio_backend",
            "face_detection_model",
            "opencv_backend",
        ]
        
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required configuration key: {key}")
                return False
        
        # Validate memory limit
        if config["memory_limit_mb"] < 256:
            logger.warning("Memory limit is very low (< 256MB)")
        
        # Validate worker processes
        if config["worker_processes"] < 1:
            logger.error("Worker processes must be at least 1")
            return False
        
        return True
    
    @classmethod
    def get_installation_dependencies(cls, platform_type: str) -> Dict[str, Any]:
        """
        Get platform-specific installation dependencies.
        
        Args:
            platform_type: Platform identifier
            
        Returns:
            Dictionary with system and Python dependencies
        """
        dependencies = {
            "raspberry_pi": {
                "system": [
                    "build-essential",
                    "cmake",
                    "libopenblas-dev",
                    "libatlas-base-dev",
                    "python3-dev",
                    "python3-pip",
                    "python3-numpy",
                    "python3-opencv",
                    "python3-pil",
                    "python3-rpi.gpio",
                    "python3-picamera2",
                ],
                "python": [
                    "face_recognition",
                    "opencv-python-headless",
                    "Pillow",
                    "numpy",
                ],
            },
            "apple_silicon": {
                "system": [
                    "cmake",
                    "pkg-config",
                    "dlib",
                    "opencv",
                ],
                "python": [
                    "face_recognition",
                    "opencv-python",
                    "Pillow",
                    "numpy",
                ],
            },
            "ubuntu_x86": {
                "system": [
                    "build-essential",
                    "cmake",
                    "libboost-all-dev",
                    "libdlib-dev",
                    "libopenblas-dev",
                    "python3-dev",
                ],
                "python": [
                    "face_recognition",
                    "opencv-python",
                    "Pillow",
                    "numpy",
                ],
            },
            "windows": {
                "system": [],  # Windows uses conda or pip
                "python": [
                    "face_recognition",
                    "opencv-python",
                    "Pillow",
                    "numpy",
                ],
            },
        }
        
        return dependencies.get(platform_type, dependencies["ubuntu_x86"])


# Singleton instance
_platform_configs = None


def get_platform_configs(platform_detector=None) -> Dict[str, Any]:
    """
    Get platform-specific configurations (singleton pattern).
    
    Args:
        platform_detector: Optional PlatformDetector instance
        
    Returns:
        Platform-specific configuration dictionary
    """
    global _platform_configs
    
    if _platform_configs is None:
        _platform_configs = PlatformConfigurations.get_optimized_config(platform_detector)
        
        # Log configuration
        logger.info("Platform configuration loaded:")
        for key, value in _platform_configs.items():
            logger.debug(f"  {key}: {value}")
    
    return _platform_configs


def reload_platform_configs(platform_detector=None) -> Dict[str, Any]:
    """
    Reload platform configurations (useful for testing).
    
    Args:
        platform_detector: Optional PlatformDetector instance
        
    Returns:
        Reloaded platform configuration dictionary
    """
    global _platform_configs
    _platform_configs = None
    return get_platform_configs(platform_detector)
