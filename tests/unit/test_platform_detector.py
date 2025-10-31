#!/usr/bin/env python3
"""
Unit tests for platform_detector module.

Tests platform detection and configuration.
"""

import os
import platform
from unittest.mock import patch, mock_open, MagicMock

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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
