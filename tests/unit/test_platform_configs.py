#!/usr/bin/env python3
"""
Unit tests for platform_configs module.

Tests platform-specific configuration management.
"""

import pytest
from unittest.mock import Mock, patch

from config.platform_configs import (
    PlatformConfigurations,
    get_platform_configs,
    reload_platform_configs
)


class TestPlatformConfigurations:
    """Test PlatformConfigurations class."""

    def test_all_platform_configs_exist(self):
        """Test all expected platform configurations exist."""
        expected_platforms = [
            "raspberry_pi",
            "apple_silicon",
            "macos_intel",
            "ubuntu_x86",
            "windows",
            "docker"
        ]
        
        for platform in expected_platforms:
            assert platform in PlatformConfigurations.CONFIGS

    def test_config_structure(self):
        """Test each configuration has required keys."""
        required_keys = [
            "memory_limit_mb",
            "worker_processes",
            "camera_backend",
            "gpio_backend",
            "face_detection_model",
            "opencv_backend",
            "installation_method"
        ]
        
        for platform, config in PlatformConfigurations.CONFIGS.items():
            for key in required_keys:
                assert key in config, f"Platform {platform} missing key: {key}"

    def test_raspberry_pi_config(self):
        """Test Raspberry Pi configuration is optimized for low resources."""
        config = PlatformConfigurations.get_config_for_platform("raspberry_pi")
        
        assert config["memory_limit_mb"] <= 1024
        assert config["worker_processes"] == 1
        assert config["face_detection_model"] == "hog"
        assert config["low_memory_mode"] is True

    def test_apple_silicon_config(self):
        """Test Apple Silicon configuration uses appropriate settings."""
        config = PlatformConfigurations.get_config_for_platform("apple_silicon")
        
        assert config["memory_limit_mb"] >= 2048
        assert config["worker_processes"] >= 2
        assert config["installation_method"] == "homebrew_first"
        assert config["enable_metal_acceleration"] is True

    def test_windows_config(self):
        """Test Windows configuration has correct path separator."""
        config = PlatformConfigurations.get_config_for_platform("windows")
        
        assert config["path_separator"] == "\\"
        assert config["installation_method"] == "conda_preferred"

    def test_get_config_returns_copy(self):
        """Test get_config_for_platform returns a copy."""
        config1 = PlatformConfigurations.get_config_for_platform("raspberry_pi")
        config2 = PlatformConfigurations.get_config_for_platform("raspberry_pi")
        
        # Modify one config
        config1["test_key"] = "test_value"
        
        # Other config should not be affected
        assert "test_key" not in config2


class TestGetConfigWithPlatformDetector:
    """Test get_config with platform detector."""

    @patch('src.platform_detector.platform_detector')
    def test_raspberry_pi_detected(self, mock_detector):
        """Test Raspberry Pi platform is detected correctly."""
        mock_detector.is_raspberry_pi = True
        mock_detector.is_apple_silicon = False
        mock_detector.is_macos = False
        mock_detector.is_windows = False
        mock_detector.is_linux = True
        mock_detector.is_development = False
        mock_detector.system = "linux"
        mock_detector.machine = "armv7l"
        
        config = PlatformConfigurations.get_config(mock_detector)
        
        assert config["platform_type"] == "raspberry_pi"
        assert config["gpio_backend"] == "RPi.GPIO"

    @patch('src.platform_detector.platform_detector')
    def test_apple_silicon_detected(self, mock_detector):
        """Test Apple Silicon platform is detected correctly."""
        mock_detector.is_raspberry_pi = False
        mock_detector.is_apple_silicon = True
        mock_detector.is_macos = True
        mock_detector.is_windows = False
        mock_detector.is_linux = False
        mock_detector.is_development = False
        mock_detector.system = "darwin"
        mock_detector.machine = "arm64"
        
        config = PlatformConfigurations.get_config(mock_detector)
        
        assert config["platform_type"] == "apple_silicon"
        assert config["enable_metal_acceleration"] is True

    @patch('src.platform_detector.platform_detector')
    def test_development_mode_overrides(self, mock_detector):
        """Test development mode overrides GPIO backend."""
        mock_detector.is_raspberry_pi = True
        mock_detector.is_apple_silicon = False
        mock_detector.is_macos = False
        mock_detector.is_windows = False
        mock_detector.is_linux = True
        mock_detector.is_development = True
        mock_detector.system = "linux"
        mock_detector.machine = "armv7l"
        
        config = PlatformConfigurations.get_config(mock_detector)
        
        # Development mode should force mock GPIO
        assert config["gpio_backend"] == "mock"
        assert config["enable_debug_logging"] is True

    @patch('src.platform_detector.platform_detector')
    def test_windows_platform_detected(self, mock_detector):
        """Test Windows platform is detected correctly."""
        mock_detector.is_raspberry_pi = False
        mock_detector.is_apple_silicon = False
        mock_detector.is_macos = False
        mock_detector.is_windows = True
        mock_detector.is_linux = False
        mock_detector.is_development = False
        mock_detector.system = "windows"
        mock_detector.machine = "x86_64"
        
        config = PlatformConfigurations.get_config(mock_detector)
        
        assert config["platform_type"] == "windows"
        assert config["path_separator"] == "\\"


class TestMemoryConfig:
    """Test memory-based configuration optimization."""

    def test_very_low_memory_config(self):
        """Test configuration for very low memory (< 1GB)."""
        config = PlatformConfigurations.get_memory_config(512)
        
        assert config["memory_limit_mb"] == 512
        assert config["worker_processes"] == 1
        assert config["low_memory_mode"] is True
        assert config["max_concurrent_operations"] == 1

    def test_low_memory_config(self):
        """Test configuration for low memory (1-2GB)."""
        config = PlatformConfigurations.get_memory_config(1536)
        
        assert config["memory_limit_mb"] == 1024
        assert config["worker_processes"] == 1
        assert config["low_memory_mode"] is True

    def test_medium_memory_config(self):
        """Test configuration for medium memory (2-4GB)."""
        config = PlatformConfigurations.get_memory_config(3072)
        
        assert config["memory_limit_mb"] == 2048
        assert config["worker_processes"] == 2
        assert config["low_memory_mode"] is False

    def test_high_memory_config(self):
        """Test configuration for high memory (4GB+)."""
        config = PlatformConfigurations.get_memory_config(8192)
        
        assert config["memory_limit_mb"] == 4096
        assert config["worker_processes"] == 4
        assert config["low_memory_mode"] is False
        assert config["max_concurrent_operations"] == 4


class TestOptimizedConfig:
    """Test get_optimized_config method."""

    @patch('src.platform_detector.platform_detector')
    def test_optimized_config_with_memory(self, mock_detector):
        """Test optimized config considers memory."""
        mock_detector.is_raspberry_pi = False
        mock_detector.is_apple_silicon = False
        mock_detector.is_macos = False
        mock_detector.is_windows = False
        mock_detector.is_linux = True
        mock_detector.is_development = False
        mock_detector.system = "linux"
        mock_detector.machine = "x86_64"
        
        mock_detector.get_platform_info.return_value = {
            "memory_gb": 8.0,
            "cpu_count": 4,
            "has_gpu": False,
            "gpu_info": {}
        }
        
        config = PlatformConfigurations.get_optimized_config(mock_detector)
        
        # Should use high memory config
        assert config["memory_limit_mb"] >= 2048

    @patch('src.platform_detector.platform_detector')
    def test_optimized_config_with_gpu(self, mock_detector):
        """Test optimized config enables GPU if available."""
        mock_detector.is_raspberry_pi = False
        mock_detector.is_apple_silicon = False
        mock_detector.is_macos = False
        mock_detector.is_windows = False
        mock_detector.is_linux = True
        mock_detector.is_development = False
        mock_detector.system = "linux"
        mock_detector.machine = "x86_64"
        
        mock_detector.get_platform_info.return_value = {
            "memory_gb": 8.0,
            "cpu_count": 4,
            "has_gpu": True,
            "gpu_info": {"cuda_available": True, "type": "nvidia"}
        }
        
        config = PlatformConfigurations.get_optimized_config(mock_detector)
        
        # Should enable GPU acceleration
        assert config.get("enable_gpu_acceleration") is True
        assert config.get("gpu_backend") == "cuda"

    @patch('src.platform_detector.platform_detector')
    def test_optimized_config_cpu_scaling(self, mock_detector):
        """Test optimized config scales with CPU count."""
        mock_detector.is_raspberry_pi = False
        mock_detector.is_apple_silicon = False
        mock_detector.is_macos = False
        mock_detector.is_windows = False
        mock_detector.is_linux = True
        mock_detector.is_development = False
        mock_detector.system = "linux"
        mock_detector.machine = "x86_64"
        
        mock_detector.get_platform_info.return_value = {
            "memory_gb": 4.0,
            "cpu_count": 8,
            "has_gpu": False,
            "gpu_info": {}
        }
        
        config = PlatformConfigurations.get_optimized_config(mock_detector)
        
        # Should scale workers with CPU count (but capped at 4)
        assert config["worker_processes"] <= 4


class TestValidateConfig:
    """Test configuration validation."""

    def test_valid_config(self):
        """Test validation of valid configuration."""
        config = {
            "memory_limit_mb": 2048,
            "worker_processes": 2,
            "camera_backend": "opencv",
            "gpio_backend": "mock",
            "face_detection_model": "hog",
            "opencv_backend": "opencv-python"
        }
        
        assert PlatformConfigurations.validate_config(config) is True

    def test_missing_required_key(self):
        """Test validation fails with missing key."""
        config = {
            "memory_limit_mb": 2048,
            "worker_processes": 2,
            # Missing camera_backend
        }
        
        assert PlatformConfigurations.validate_config(config) is False

    def test_invalid_worker_count(self):
        """Test validation fails with invalid worker count."""
        config = {
            "memory_limit_mb": 2048,
            "worker_processes": 0,  # Invalid
            "camera_backend": "opencv",
            "gpio_backend": "mock",
            "face_detection_model": "hog",
            "opencv_backend": "opencv-python"
        }
        
        assert PlatformConfigurations.validate_config(config) is False


class TestInstallationDependencies:
    """Test get_installation_dependencies method."""

    def test_raspberry_pi_dependencies(self):
        """Test Raspberry Pi dependencies include system packages."""
        deps = PlatformConfigurations.get_installation_dependencies("raspberry_pi")
        
        assert "system" in deps
        assert "python" in deps
        assert "python3-rpi.gpio" in deps["system"]
        assert "python3-picamera2" in deps["system"]

    def test_apple_silicon_dependencies(self):
        """Test Apple Silicon dependencies use homebrew."""
        deps = PlatformConfigurations.get_installation_dependencies("apple_silicon")
        
        assert "system" in deps
        assert "python" in deps
        assert "cmake" in deps["system"]
        assert "dlib" in deps["system"]

    def test_windows_dependencies(self):
        """Test Windows dependencies are minimal."""
        deps = PlatformConfigurations.get_installation_dependencies("windows")
        
        assert "system" in deps
        assert "python" in deps
        assert len(deps["system"]) == 0  # No system packages for Windows


class TestSingletonPattern:
    """Test singleton pattern for platform configs."""

    def test_get_platform_configs_singleton(self):
        """Test get_platform_configs returns same instance."""
        # Reload to ensure clean state
        reload_platform_configs()
        
        config1 = get_platform_configs()
        config2 = get_platform_configs()
        
        # Should return the same dictionary object
        assert config1 is config2

    def test_reload_platform_configs(self):
        """Test reload_platform_configs creates new instance."""
        config1 = get_platform_configs()
        config2 = reload_platform_configs()
        
        # After reload, should get different object
        # (but with same content)
        assert isinstance(config1, dict)
        assert isinstance(config2, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
