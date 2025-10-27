"""
Tests for Hardware Abstraction Layer

Tests the core HAL functionality including platform detection, hardware
initialization, mock implementations, and configuration management.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from src.hardware import (
    HardwareAbstractionLayer,
    HardwareType,
    HardwareStatus,
    get_hal
)
from src.hardware.base_hardware import (
    CameraHandler,
    GPIOHandler,
    SensorHandler,
    CameraInfo,
    CameraSettings,
    GPIOMode,
    GPIOEdge
)
from src.hardware.mock.mock_camera import MockCameraHandler
from src.hardware.mock.mock_gpio import MockGPIOHandler
from src.hardware.mock.mock_sensors import MockSensorHandler
from config.hardware_config import HardwareConfig, CameraConfig, GPIOConfig


class TestHardwareAbstractionLayer:
    """Test suite for Hardware Abstraction Layer core functionality."""
    
    def test_hal_initialization(self):
        """Test HAL initialization with default configuration."""
        config = {'mock_mode': True}
        hal = HardwareAbstractionLayer(config)
        
        assert hal is not None
        assert hal.mock_mode is True
        assert hal.platform_info is not None
    
    def test_hal_camera_handler_initialization(self):
        """Test camera handler initialization through HAL."""
        config = {'mock_mode': True}
        hal = HardwareAbstractionLayer(config)
        
        camera = hal.get_camera_handler()
        assert camera is not None
        assert isinstance(camera, CameraHandler)
    
    def test_hal_gpio_handler_initialization(self):
        """Test GPIO handler initialization through HAL."""
        config = {'mock_mode': True}
        hal = HardwareAbstractionLayer(config)
        
        gpio = hal.get_gpio_handler()
        assert gpio is not None
        assert isinstance(gpio, GPIOHandler)
    
    def test_hal_health_check(self):
        """Test HAL health check functionality."""
        config = {'mock_mode': True}
        hal = HardwareAbstractionLayer(config)
        
        # Initialize hardware
        camera = hal.get_camera_handler()
        camera.initialize()
        
        gpio = hal.get_gpio_handler()
        gpio.initialize()
        
        # Perform health check
        health = hal.health_check()
        
        assert health is not None
        assert health.healthy is True
        assert 'camera' in health.components
        assert 'gpio' in health.components
    
    def test_hal_mock_mode_switch(self):
        """Test switching HAL to mock mode."""
        config = {'mock_mode': False}
        hal = HardwareAbstractionLayer(config)
        
        # Switch to mock mode
        hal.switch_to_mock_mode()
        
        assert hal.mock_mode is True
        assert hal.camera_handler is not None
        assert hal.gpio_handler is not None
    
    def test_hal_cleanup(self):
        """Test HAL cleanup functionality."""
        config = {'mock_mode': True}
        hal = HardwareAbstractionLayer(config)
        
        # Initialize handlers
        camera = hal.get_camera_handler()
        camera.initialize()
        
        # Cleanup
        hal.cleanup()
        
        # Handlers should be cleaned up
        assert not camera.is_available()
    
    def test_get_hal_singleton(self):
        """Test global HAL singleton pattern."""
        config = {'mock_mode': True}
        
        # Get HAL instances
        hal1 = get_hal(config)
        hal2 = get_hal()
        
        # Should be the same instance
        assert hal1 is hal2


class TestMockCameraHandler:
    """Test suite for mock camera implementation."""
    
    def test_mock_camera_initialization(self):
        """Test mock camera initialization."""
        config = {'resolution': (640, 480), 'fps': 15.0}
        camera = MockCameraHandler(config)
        
        assert camera.initialize() is True
        assert camera.is_available() is True
    
    def test_mock_camera_frame_capture(self):
        """Test mock camera frame capture."""
        config = {'resolution': (640, 480), 'fps': 15.0}
        camera = MockCameraHandler(config)
        camera.initialize()
        
        frame = camera.capture_frame()
        
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape[2] == 3  # RGB channels
    
    def test_mock_camera_stream(self):
        """Test mock camera streaming."""
        config = {'resolution': (640, 480), 'fps': 15.0}
        camera = MockCameraHandler(config)
        camera.initialize()
        
        assert camera.start_stream() is True
        assert camera.stream_active is True
        
        camera.stop_stream()
        assert camera.stream_active is False
    
    def test_mock_camera_info(self):
        """Test mock camera info retrieval."""
        config = {'resolution': (640, 480), 'fps': 15.0}
        camera = MockCameraHandler(config)
        camera.initialize()
        
        info = camera.get_camera_info()
        
        assert isinstance(info, CameraInfo)
        assert info.backend == 'mock'
        assert info.available is True
    
    def test_mock_camera_settings(self):
        """Test mock camera settings update."""
        config = {'resolution': (640, 480), 'fps': 15.0}
        camera = MockCameraHandler(config)
        camera.initialize()
        
        new_settings = CameraSettings(
            resolution=(1280, 720),
            fps=30.0,
            brightness=60.0
        )
        
        assert camera.set_camera_settings(new_settings) is True
        assert camera.settings.resolution == (1280, 720)
    
    def test_mock_camera_cleanup(self):
        """Test mock camera cleanup."""
        config = {'resolution': (640, 480), 'fps': 15.0}
        camera = MockCameraHandler(config)
        camera.initialize()
        
        camera.cleanup()
        
        assert camera.is_available() is False


class TestMockGPIOHandler:
    """Test suite for mock GPIO implementation."""
    
    def test_mock_gpio_initialization(self):
        """Test mock GPIO initialization."""
        config = {'simulate_events': False}
        gpio = MockGPIOHandler(config)
        
        assert gpio.initialize() is True
        assert gpio.initialized is True
    
    def test_mock_gpio_pin_setup(self):
        """Test mock GPIO pin setup."""
        config = {'simulate_events': False}
        gpio = MockGPIOHandler(config)
        gpio.initialize()
        
        # Setup input pin
        assert gpio.setup_pin(18, GPIOMode.INPUT) is True
        
        # Setup output pin
        assert gpio.setup_pin(21, GPIOMode.OUTPUT, initial=False) is True
    
    def test_mock_gpio_pin_read_write(self):
        """Test mock GPIO pin read and write operations."""
        config = {'simulate_events': False}
        gpio = MockGPIOHandler(config)
        gpio.initialize()
        
        # Setup and write to output pin
        gpio.setup_pin(21, GPIOMode.OUTPUT)
        assert gpio.write_pin(21, True) is True
        
        # Read pin state
        state = gpio.read_pin(21)
        assert state is True
    
    def test_mock_gpio_interrupt_setup(self):
        """Test mock GPIO interrupt setup."""
        config = {'simulate_events': False}
        gpio = MockGPIOHandler(config)
        gpio.initialize()
        
        # Setup input pin
        gpio.setup_pin(18, GPIOMode.INPUT)
        
        # Setup interrupt
        callback = Mock()
        assert gpio.setup_interrupt(18, callback, GPIOEdge.FALLING) is True
    
    def test_mock_gpio_simulate_pin_change(self):
        """Test mock GPIO pin change simulation."""
        config = {'simulate_events': False}
        gpio = MockGPIOHandler(config)
        gpio.initialize()
        
        # Setup pin and interrupt
        gpio.setup_pin(18, GPIOMode.INPUT)
        callback = Mock()
        gpio.setup_interrupt(18, callback, GPIOEdge.FALLING)
        
        # Simulate pin change
        gpio.simulate_pin_change(18, True)
        time.sleep(0.1)  # Give callback time to execute
        gpio.simulate_pin_change(18, False)
        time.sleep(0.1)
        
        # Callback should have been called
        assert callback.called
    
    def test_mock_gpio_cleanup(self):
        """Test mock GPIO cleanup."""
        config = {'simulate_events': False}
        gpio = MockGPIOHandler(config)
        gpio.initialize()
        
        gpio.cleanup()
        
        assert gpio.initialized is False
        assert len(gpio.pin_states) == 0


class TestMockSensorHandler:
    """Test suite for mock sensor implementation."""
    
    def test_mock_sensor_initialization(self):
        """Test mock sensor initialization."""
        config = {}
        sensor = MockSensorHandler(config)
        
        assert sensor.is_available() is True
    
    def test_mock_sensor_temperature_read(self):
        """Test mock temperature sensor reading."""
        config = {'base_temperature': 22.0}
        sensor = MockSensorHandler(config)
        
        temp = sensor.read_temperature()
        
        assert temp is not None
        assert isinstance(temp, float)
        assert 18.0 <= temp <= 26.0  # Base ± variation
    
    def test_mock_sensor_humidity_read(self):
        """Test mock humidity sensor reading."""
        config = {'base_humidity': 45.0}
        sensor = MockSensorHandler(config)
        
        humidity = sensor.read_humidity()
        
        assert humidity is not None
        assert isinstance(humidity, float)
        assert 0.0 <= humidity <= 100.0
    
    def test_mock_sensor_motion_read(self):
        """Test mock motion sensor reading."""
        config = {}
        sensor = MockSensorHandler(config)
        
        motion = sensor.read_motion_sensor()
        
        assert motion is not None
        assert isinstance(motion, bool)
    
    def test_mock_sensor_status(self):
        """Test mock sensor status retrieval."""
        config = {}
        sensor = MockSensorHandler(config)
        
        status = sensor.get_sensor_status()
        
        assert 'temperature' in status
        assert 'humidity' in status
        assert 'motion' in status
        assert status['available'] is True
    
    def test_mock_sensor_manual_trigger(self):
        """Test manual motion sensor triggering."""
        config = {}
        sensor = MockSensorHandler(config)
        
        sensor.trigger_motion()
        assert sensor.motion_detected is True
        
        sensor.clear_motion()
        assert sensor.motion_detected is False


class TestHardwareConfig:
    """Test suite for hardware configuration."""
    
    def test_camera_config_creation(self):
        """Test camera config creation."""
        config = CameraConfig(
            resolution=(1280, 720),
            fps=30.0,
            brightness=60.0
        )
        
        assert config.resolution == (1280, 720)
        assert config.fps == 30.0
        assert config.brightness == 60.0
    
    def test_camera_config_from_dict(self):
        """Test camera config creation from dictionary."""
        config_dict = {
            'resolution': (1920, 1080),
            'fps': 60.0,
            'brightness': 70.0
        }
        
        config = CameraConfig.from_dict(config_dict)
        
        assert config.resolution == (1920, 1080)
        assert config.fps == 60.0
    
    def test_gpio_config_creation(self):
        """Test GPIO config creation."""
        config = GPIOConfig(
            gpio_mode='BCM',
            doorbell_pin=18,
            debounce_time=300.0
        )
        
        assert config.gpio_mode == 'BCM'
        assert config.doorbell_pin == 18
        assert config.debounce_time == 300.0
    
    def test_hardware_config_creation(self):
        """Test complete hardware config creation."""
        config = HardwareConfig(
            mock_mode=True,
            camera_config=CameraConfig(),
            gpio_config=GPIOConfig()
        )
        
        assert config.mock_mode is True
        assert config.camera_config is not None
        assert config.gpio_config is not None
    
    def test_hardware_config_from_dict(self):
        """Test hardware config creation from dictionary."""
        config_dict = {
            'mock_mode': True,
            'camera_config': {
                'resolution': (640, 480)
            },
            'gpio_config': {
                'doorbell_pin': 18
            }
        }
        
        config = HardwareConfig.from_dict(config_dict)
        
        assert config.mock_mode is True
        assert config.camera_config.resolution == (640, 480)
        assert config.gpio_config.doorbell_pin == 18
    
    def test_hardware_config_legacy_migration(self):
        """Test hardware config migration from legacy settings."""
        # Create mock legacy settings
        mock_settings = Mock()
        mock_settings.CAMERA_RESOLUTION = (1280, 720)
        mock_settings.CAMERA_ROTATION = 90
        mock_settings.CAMERA_BRIGHTNESS = 55.0
        mock_settings.CAMERA_CONTRAST = 60.0
        mock_settings.DOORBELL_PIN = 18
        mock_settings.STATUS_LED_PINS = {'red': 16, 'yellow': 20, 'green': 21}
        mock_settings.DEBOUNCE_TIME = 2.0
        
        config = HardwareConfig.from_legacy_config(mock_settings)
        
        assert config.camera_config.resolution == (1280, 720)
        assert config.camera_config.rotation == 90
        assert config.gpio_config.doorbell_pin == 18
    
    def test_hardware_config_platform_specific(self):
        """Test platform-specific configuration generation."""
        config = HardwareConfig()
        
        # Get Raspberry Pi specific config
        pi_config = config.to_platform_specific_config('raspberry_pi')
        assert pi_config['camera_config']['use_gpu_memory'] is True
        
        # Get macOS specific config
        macos_config = config.to_platform_specific_config('macos')
        assert macos_config['camera_config']['use_gpu_memory'] is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
