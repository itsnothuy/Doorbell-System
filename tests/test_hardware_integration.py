"""
Integration tests for Hardware Abstraction Layer with existing components

Tests the integration of the new HAL with existing camera and GPIO handlers
to ensure backward compatibility and seamless operation.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from src.camera_handler import CameraHandler as LegacyCameraHandler
from src.gpio_handler import GPIOHandler as LegacyGPIOHandler
from config.settings import Settings


class TestBackwardCompatibility:
    """Test backward compatibility of updated handlers."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock(spec=Settings)
        settings.CAMERA_RESOLUTION = (640, 480)
        settings.CAMERA_ROTATION = 0
        settings.CAMERA_BRIGHTNESS = 50.0
        settings.CAMERA_CONTRAST = 50.0
        settings.OPENCV_CAMERA_MAX_INDEX_TO_TRY = 5
        settings.DOORBELL_PIN = 18
        settings.STATUS_LED_PINS = {'red': 16, 'yellow': 20, 'green': 21}
        settings.DEBOUNCE_TIME = 2.0
        return settings
    
    def test_legacy_camera_handler_initialization(self, mock_settings):
        """Test legacy camera handler still works with HAL fallback."""
        with patch('src.camera_handler.Settings', return_value=mock_settings):
            camera = LegacyCameraHandler()
            
            assert camera is not None
            assert camera.camera is None  # Not initialized yet
    
    def test_legacy_camera_handler_mock_mode(self, mock_settings):
        """Test legacy camera handler in mock mode."""
        with patch('src.camera_handler.Settings', return_value=mock_settings):
            with patch('src.platform_detector.platform_detector.get_camera_config') as mock_config:
                mock_config.return_value = {'mock': True, 'type': 'mock'}
                
                camera = LegacyCameraHandler()
                camera.initialize()
                
                assert camera.is_initialized is True
                
                # Test capture
                frame = camera.capture_image()
                assert frame is not None
                assert isinstance(frame, np.ndarray)
    
    def test_legacy_gpio_handler_initialization(self, mock_settings):
        """Test legacy GPIO handler still works."""
        with patch('src.gpio_handler.Settings', return_value=mock_settings):
            with patch('src.platform_detector.platform_detector.get_gpio_config') as mock_config:
                mock_config.return_value = {
                    'use_real_gpio': False,
                    'mock': True,
                    'web_interface': True
                }
                
                gpio = LegacyGPIOHandler()
                
                assert gpio is not None
                assert gpio.initialized is True
    
    def test_legacy_gpio_handler_led_control(self, mock_settings):
        """Test legacy GPIO handler LED control."""
        with patch('src.gpio_handler.Settings', return_value=mock_settings):
            with patch('src.platform_detector.platform_detector.get_gpio_config') as mock_config:
                mock_config.return_value = {
                    'use_real_gpio': False,
                    'mock': True,
                    'web_interface': True
                }
                
                gpio = LegacyGPIOHandler()
                
                # Test LED control
                gpio.set_status_led('idle')
                assert gpio.led_states.get('green') is True
                
                gpio.set_status_led('alert')
                assert gpio.led_states.get('red') is True
    
    def test_legacy_gpio_handler_doorbell_simulation(self, mock_settings):
        """Test legacy GPIO handler doorbell simulation."""
        with patch('src.gpio_handler.Settings', return_value=mock_settings):
            with patch('src.platform_detector.platform_detector.get_gpio_config') as mock_config:
                mock_config.return_value = {
                    'use_real_gpio': False,
                    'mock': True,
                    'web_interface': True
                }
                
                gpio = LegacyGPIOHandler()
                
                # Setup callback
                callback = Mock()
                gpio.setup_doorbell_button(callback)
                
                # Simulate doorbell press
                gpio.simulate_doorbell_press()
                
                # Give callback time to execute
                import time
                time.sleep(0.2)
                
                # Callback should have been called
                assert callback.called


class TestHALIntegration:
    """Test HAL integration scenarios."""
    
    def test_hal_with_mock_hardware(self):
        """Test complete HAL setup with mock hardware."""
        from src.hardware import get_hal
        from config.hardware_config import HardwareConfig
        
        config = HardwareConfig(mock_mode=True)
        hal = get_hal(config.to_dict())
        
        # Get camera handler
        camera = hal.get_camera_handler()
        assert camera is not None
        assert camera.initialize() is True
        
        # Test frame capture
        frame = camera.capture_frame()
        assert frame is not None
        
        # Get GPIO handler
        gpio = hal.get_gpio_handler()
        assert gpio is not None
        assert gpio.initialize() is True
        
        # Test GPIO operations
        from src.hardware.base_hardware import GPIOMode
        assert gpio.setup_pin(18, GPIOMode.INPUT) is True
        
        # Cleanup
        hal.cleanup()
    
    def test_hal_health_monitoring(self):
        """Test HAL health monitoring."""
        from src.hardware import get_hal
        from config.hardware_config import HardwareConfig
        
        config = HardwareConfig(mock_mode=True)
        hal = get_hal(config.to_dict())
        
        # Initialize hardware
        camera = hal.get_camera_handler()
        camera.initialize()
        
        gpio = hal.get_gpio_handler()
        gpio.initialize()
        
        # Check health
        health = hal.health_check()
        
        assert health.healthy is True
        assert len(health.errors) == 0
        assert 'camera' in health.components
        assert 'gpio' in health.components
        
        # Cleanup
        hal.cleanup()
    
    def test_hal_configuration_override(self):
        """Test HAL with custom configuration."""
        from src.hardware import HardwareAbstractionLayer
        from config.hardware_config import CameraConfig, GPIOConfig
        
        custom_config = {
            'mock_mode': True,
            'camera_config': CameraConfig(
                resolution=(1920, 1080),
                fps=30.0
            ).to_dict(),
            'gpio_config': GPIOConfig(
                doorbell_pin=25
            ).to_dict()
        }
        
        hal = HardwareAbstractionLayer(custom_config)
        
        camera = hal.get_camera_handler()
        camera.initialize()
        
        # Verify custom config is applied
        info = camera.get_camera_info()
        assert info.resolution == (1920, 1080)
        
        hal.cleanup()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
