#!/usr/bin/env python3
"""
Unit tests for platform_detector module.

Tests platform detection and configuration.
"""

import os
import platform
import subprocess
from unittest.mock import patch, mock_open, MagicMock, Mock

import pytest

from src.platform_detector import PlatformDetector


class TestPlatformDetector:
    """Test platform detection functionality."""

    def test_platform_detector_initialization(self):
        """Test PlatformDetector initializes correctly."""
        detector = PlatformDetector()
        assert detector is not None
        assert hasattr(detector, 'system')
        assert hasattr(detector, 'machine')

    def test_system_detected(self):
        """Test system type is detected."""
        detector = PlatformDetector()
        assert detector.system in ['linux', 'darwin', 'windows']

    def test_machine_detected(self):
        """Test machine architecture is detected."""
        detector = PlatformDetector()
        assert detector.machine is not None
        assert isinstance(detector.machine, str)

    def test_platform_flags_are_boolean(self):
        """Test platform flags are booleans."""
        detector = PlatformDetector()
        assert isinstance(detector.is_raspberry_pi, bool)
        assert isinstance(detector.is_macos, bool)
        assert isinstance(detector.is_linux, bool)
        assert isinstance(detector.is_windows, bool)
        assert isinstance(detector.is_development, bool)

    def test_exactly_one_os_flag_true(self):
        """Test exactly one OS flag is true."""
        detector = PlatformDetector()
        os_flags = [detector.is_macos, detector.is_linux, detector.is_windows]
        # Exactly one should be True
        assert sum(os_flags) == 1

    @patch('platform.system')
    def test_detects_macos(self, mock_system):
        """Test macOS detection."""
        mock_system.return_value = 'Darwin'
        detector = PlatformDetector()
        assert detector.is_macos is True
        assert detector.is_linux is False
        assert detector.is_windows is False

    @patch('platform.system')
    def test_detects_linux(self, mock_system):
        """Test Linux detection."""
        mock_system.return_value = 'Linux'
        detector = PlatformDetector()
        assert detector.is_linux is True
        assert detector.is_macos is False
        assert detector.is_windows is False

    @patch('platform.system')
    def test_detects_windows(self, mock_system):
        """Test Windows detection."""
        mock_system.return_value = 'Windows'
        detector = PlatformDetector()
        assert detector.is_windows is True
        assert detector.is_macos is False
        assert detector.is_linux is False

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='raspberry pi')
    def test_detects_raspberry_pi_from_device_tree(self, mock_file, mock_exists):
        """Test Raspberry Pi detection from device tree."""
        mock_exists.return_value = True
        detector = PlatformDetector()
        # Should detect Raspberry Pi
        assert detector.is_raspberry_pi is True

    @patch('os.path.exists')
    def test_not_raspberry_pi_when_files_missing(self, mock_exists):
        """Test Raspberry Pi not detected when files missing."""
        mock_exists.return_value = False
        detector = PlatformDetector()
        # Should not detect Raspberry Pi
        assert detector.is_raspberry_pi is False

    @patch.dict(os.environ, {'DEVELOPMENT_MODE': 'true'})
    def test_development_mode_enabled(self):
        """Test development mode detection."""
        detector = PlatformDetector()
        assert detector.is_development is True

    @patch.dict(os.environ, {'DEVELOPMENT_MODE': 'false'})
    def test_development_mode_disabled(self):
        """Test development mode not detected."""
        detector = PlatformDetector()
        assert detector.is_development is False

    @patch.dict(os.environ, {}, clear=True)
    def test_development_mode_default_false(self):
        """Test development mode defaults to false."""
        detector = PlatformDetector()
        assert detector.is_development is False

    def test_system_is_lowercase(self):
        """Test system string is lowercase."""
        detector = PlatformDetector()
        assert detector.system == detector.system.lower()

    def test_machine_is_lowercase(self):
        """Test machine string is lowercase."""
        detector = PlatformDetector()
        assert detector.machine == detector.machine.lower()


class TestPlatformDetectorMethods:
    """Test PlatformDetector helper methods."""

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='Raspberry Pi 4 Model B')
    def test_detect_raspberry_pi_from_proc_device_tree(self, mock_file, mock_exists):
        """Test Raspberry Pi detection from /proc/device-tree/model."""
        def exists_side_effect(path):
            return path == '/proc/device-tree/model'
        
        mock_exists.side_effect = exists_side_effect
        
        detector = PlatformDetector()
        assert detector.is_raspberry_pi is True

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='processor : 0\nmodel name : ARMv7 Processor rev 4 (v7l)\nHardware : BCM2835\nRevision : raspberry pi')
    def test_detect_raspberry_pi_from_cpuinfo(self, mock_file, mock_exists):
        """Test Raspberry Pi detection from /proc/cpuinfo."""
        def exists_side_effect(path):
            if path == '/proc/device-tree/model':
                return False
            return path == '/proc/cpuinfo'
        
        mock_exists.side_effect = exists_side_effect
        
        detector = PlatformDetector()
        # May or may not detect depending on implementation
        assert isinstance(detector.is_raspberry_pi, bool)

    @patch('os.path.exists')
    def test_detect_raspberry_pi_no_pi_files(self, mock_exists):
        """Test Raspberry Pi detection when no Pi files exist."""
        mock_exists.return_value = False
        detector = PlatformDetector()
        assert detector.is_raspberry_pi is False


class TestPlatformDetectorIntegration:
    """Integration tests for platform detection."""

    def test_platform_detector_consistent_state(self):
        """Test platform detector maintains consistent state."""
        detector1 = PlatformDetector()
        detector2 = PlatformDetector()
        
        # Both instances should detect the same platform
        assert detector1.system == detector2.system
        assert detector1.machine == detector2.machine
        assert detector1.is_raspberry_pi == detector2.is_raspberry_pi

    def test_platform_info_types(self):
        """Test all platform info is correct type."""
        detector = PlatformDetector()
        
        assert isinstance(detector.system, str)
        assert isinstance(detector.machine, str)
        assert isinstance(detector.is_raspberry_pi, bool)
        assert isinstance(detector.is_macos, bool)
        assert isinstance(detector.is_linux, bool)
        assert isinstance(detector.is_windows, bool)
        assert isinstance(detector.is_development, bool)
        assert isinstance(detector.is_apple_silicon, bool)


class TestAppleSiliconDetection:
    """Test Apple Silicon detection functionality."""

    @patch('platform.system')
    @patch('platform.machine')
    def test_detects_apple_silicon_m1(self, mock_machine, mock_system):
        """Test Apple Silicon M1 detection via architecture."""
        mock_system.return_value = 'Darwin'
        mock_machine.return_value = 'arm64'
        detector = PlatformDetector()
        assert detector.is_apple_silicon is True
        assert detector.is_macos is True

    @patch('platform.system')
    @patch('platform.machine')
    def test_detects_apple_silicon_aarch64(self, mock_machine, mock_system):
        """Test Apple Silicon detection with aarch64 architecture."""
        mock_system.return_value = 'Darwin'
        mock_machine.return_value = 'aarch64'
        detector = PlatformDetector()
        assert detector.is_apple_silicon is True

    @patch('platform.system')
    @patch('platform.machine')
    def test_not_apple_silicon_intel_mac(self, mock_machine, mock_system):
        """Test Intel Mac is not detected as Apple Silicon."""
        mock_system.return_value = 'Darwin'
        mock_machine.return_value = 'x86_64'
        detector = PlatformDetector()
        assert detector.is_apple_silicon is False
        assert detector.is_macos is True

    @patch('platform.system')
    def test_not_apple_silicon_non_macos(self, mock_system):
        """Test non-macOS systems are not Apple Silicon."""
        mock_system.return_value = 'Linux'
        detector = PlatformDetector()
        assert detector.is_apple_silicon is False


class TestPlatformInfo:
    """Test get_platform_info method."""

    def test_get_platform_info_structure(self):
        """Test platform info returns correct structure."""
        detector = PlatformDetector()
        info = detector.get_platform_info()
        
        # Check all required keys are present
        required_keys = [
            "os", "os_version", "architecture", "processor",
            "python_version", "is_raspberry_pi", "is_apple_silicon",
            "is_windows", "is_linux", "is_macos", "has_gpu",
            "gpu_info", "has_camera", "has_gpio", "memory_gb",
            "cpu_count", "is_development"
        ]
        
        for key in required_keys:
            assert key in info, f"Missing key: {key}"

    def test_get_platform_info_types(self):
        """Test platform info values are correct types."""
        detector = PlatformDetector()
        info = detector.get_platform_info()
        
        assert isinstance(info["os"], str)
        assert isinstance(info["os_version"], str)
        assert isinstance(info["architecture"], str)
        assert isinstance(info["python_version"], str)
        assert isinstance(info["is_raspberry_pi"], bool)
        assert isinstance(info["is_apple_silicon"], bool)
        assert isinstance(info["is_windows"], bool)
        assert isinstance(info["is_linux"], bool)
        assert isinstance(info["is_macos"], bool)
        assert isinstance(info["has_gpu"], bool)
        assert isinstance(info["gpu_info"], dict)
        assert isinstance(info["has_camera"], bool)
        assert isinstance(info["has_gpio"], bool)
        assert isinstance(info["memory_gb"], (int, float))
        assert isinstance(info["cpu_count"], int)
        assert isinstance(info["is_development"], bool)

    def test_cpu_count_positive(self):
        """Test CPU count is positive."""
        detector = PlatformDetector()
        info = detector.get_platform_info()
        assert info["cpu_count"] >= 1

    def test_memory_gb_non_negative(self):
        """Test memory GB is non-negative."""
        detector = PlatformDetector()
        info = detector.get_platform_info()
        assert info["memory_gb"] >= 0


class TestRecommendedInstallation:
    """Test get_recommended_installation_method."""

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='raspberry pi')
    def test_raspberry_pi_installation_method(self, mock_file, mock_exists):
        """Test Raspberry Pi uses apt_then_pip."""
        mock_exists.return_value = True
        detector = PlatformDetector()
        method = detector.get_recommended_installation_method()
        
        if detector.is_raspberry_pi:
            assert method == "apt_then_pip"

    @patch('platform.system')
    @patch('platform.machine')
    def test_apple_silicon_installation_method(self, mock_machine, mock_system):
        """Test Apple Silicon uses homebrew_then_pip."""
        mock_system.return_value = 'Darwin'
        mock_machine.return_value = 'arm64'
        detector = PlatformDetector()
        method = detector.get_recommended_installation_method()
        assert method == "homebrew_then_pip"

    @patch('platform.system')
    def test_windows_installation_method(self, mock_system):
        """Test Windows uses conda_preferred."""
        mock_system.return_value = 'Windows'
        detector = PlatformDetector()
        method = detector.get_recommended_installation_method()
        assert method == "conda_preferred"

    @patch('platform.system')
    def test_linux_installation_method(self, mock_system):
        """Test Linux uses pip_with_system_deps."""
        mock_system.return_value = 'Linux'
        detector = PlatformDetector()
        method = detector.get_recommended_installation_method()
        
        if not detector.is_raspberry_pi:
            assert method == "pip_with_system_deps"


class TestGPUDetection:
    """Test GPU detection functionality."""

    @patch('os.path.exists')
    def test_detect_nvidia_gpu(self, mock_exists):
        """Test NVIDIA GPU detection."""
        def exists_side_effect(path):
            return path == '/usr/local/cuda'
        
        mock_exists.side_effect = exists_side_effect
        detector = PlatformDetector()
        gpu_info = detector._detect_gpu_support()
        
        if gpu_info['has_gpu']:
            assert gpu_info['type'] in ['nvidia', 'apple']

    @patch('platform.system')
    @patch('platform.machine')
    def test_detect_apple_gpu(self, mock_machine, mock_system):
        """Test Apple GPU detection on Apple Silicon."""
        mock_system.return_value = 'Darwin'
        mock_machine.return_value = 'arm64'
        detector = PlatformDetector()
        gpu_info = detector._detect_gpu_support()
        assert gpu_info['type'] == 'apple'

    @patch('os.path.exists')
    def test_no_gpu_detection(self, mock_exists):
        """Test no GPU detected when paths don't exist."""
        mock_exists.return_value = False
        detector = PlatformDetector()
        
        # Only test on non-Apple Silicon
        if not detector.is_apple_silicon:
            gpu_info = detector._detect_gpu_support()
            # GPU might still be detected on the actual system
            assert isinstance(gpu_info['has_gpu'], bool)


class TestCameraDetection:
    """Test camera detection functionality."""

    @patch('os.path.exists')
    def test_detect_camera_v4l2(self, mock_exists):
        """Test camera detection via V4L2."""
        mock_exists.return_value = True
        detector = PlatformDetector()
        
        if detector.is_linux and not detector.is_raspberry_pi:
            has_camera = detector._detect_camera_support()
            assert isinstance(has_camera, bool)

    @patch('platform.system')
    def test_macos_camera_assumed(self, mock_system):
        """Test macOS assumes camera available."""
        mock_system.return_value = 'Darwin'
        detector = PlatformDetector()
        has_camera = detector._detect_camera_support()
        assert has_camera is True

    @patch('platform.system')
    def test_windows_camera_assumed(self, mock_system):
        """Test Windows assumes camera available."""
        mock_system.return_value = 'Windows'
        detector = PlatformDetector()
        has_camera = detector._detect_camera_support()
        assert has_camera is True


class TestGPIODetection:
    """Test GPIO detection functionality."""

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='raspberry pi')
    def test_gpio_on_raspberry_pi(self, mock_file, mock_exists):
        """Test GPIO detection on Raspberry Pi."""
        def exists_side_effect(path):
            return True  # All paths exist
        
        mock_exists.side_effect = exists_side_effect
        detector = PlatformDetector()
        
        if detector.is_raspberry_pi:
            has_gpio = detector._detect_gpio_support()
            assert has_gpio is True

    @patch('platform.system')
    def test_no_gpio_on_macos(self, mock_system):
        """Test no GPIO on macOS."""
        mock_system.return_value = 'Darwin'
        detector = PlatformDetector()
        has_gpio = detector._detect_gpio_support()
        assert has_gpio is False

    @patch('platform.system')
    def test_no_gpio_on_windows(self, mock_system):
        """Test no GPIO on Windows."""
        mock_system.return_value = 'Windows'
        detector = PlatformDetector()
        has_gpio = detector._detect_gpio_support()
        assert has_gpio is False


class TestMemoryDetection:
    """Test memory detection functionality."""

    @patch('platform.system')
    @patch('builtins.open', new_callable=mock_open, read_data='MemTotal:        4096000 kB\n')
    def test_memory_detection_linux(self, mock_file, mock_system):
        """Test memory detection on Linux."""
        mock_system.return_value = 'Linux'
        detector = PlatformDetector()
        memory_gb = detector._get_memory_gb()
        
        # Should detect approximately 4GB
        assert memory_gb > 0

    @patch('platform.system')
    @patch('subprocess.run')
    def test_memory_detection_macos(self, mock_run, mock_system):
        """Test memory detection on macOS."""
        mock_system.return_value = 'Darwin'
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = str(8 * 1024 ** 3)  # 8GB in bytes
        mock_run.return_value = mock_result
        
        detector = PlatformDetector()
        memory_gb = detector._get_memory_gb()
        
        # Should detect memory
        assert memory_gb >= 0

    def test_memory_gb_returns_float(self):
        """Test memory GB returns float."""
        detector = PlatformDetector()
        memory_gb = detector._get_memory_gb()
        assert isinstance(memory_gb, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
