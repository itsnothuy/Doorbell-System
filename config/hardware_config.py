"""
Hardware configuration management

Provides configuration classes for all hardware components with
platform-specific settings and migration support.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Camera hardware configuration."""
    
    resolution: Tuple[int, int] = (1280, 720)
    fps: float = 15.0
    brightness: float = 50.0
    contrast: float = 50.0
    rotation: int = 0
    camera_index: int = 0
    max_index_to_try: int = 5
    
    # Pi-specific
    use_gpu_memory: bool = True
    camera_module_version: int = 2
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CameraConfig':
        """Create config from dictionary."""
        return cls(
            resolution=tuple(config_dict.get('resolution', (1280, 720))),
            fps=float(config_dict.get('fps', 15.0)),
            brightness=float(config_dict.get('brightness', 50.0)),
            contrast=float(config_dict.get('contrast', 50.0)),
            rotation=int(config_dict.get('rotation', 0)),
            camera_index=int(config_dict.get('camera_index', 0)),
            max_index_to_try=int(config_dict.get('max_index_to_try', 5)),
            use_gpu_memory=bool(config_dict.get('use_gpu_memory', True)),
            camera_module_version=int(config_dict.get('camera_module_version', 2))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'resolution': self.resolution,
            'fps': self.fps,
            'brightness': self.brightness,
            'contrast': self.contrast,
            'rotation': self.rotation,
            'camera_index': self.camera_index,
            'max_index_to_try': self.max_index_to_try,
            'use_gpu_memory': self.use_gpu_memory,
            'camera_module_version': self.camera_module_version
        }


@dataclass
class GPIOConfig:
    """GPIO hardware configuration."""
    
    gpio_mode: str = "BCM"  # BCM or BOARD
    doorbell_pin: int = 18
    status_led_pins: Dict[str, int] = field(default_factory=lambda: {
        'red': 16,
        'yellow': 20,
        'green': 21
    })
    debounce_time: float = 200.0  # milliseconds
    cleanup_on_exit: bool = True
    
    # Mock/simulation settings
    simulate_events: bool = False
    event_interval_range: Tuple[int, int] = (30, 120)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GPIOConfig':
        """Create config from dictionary."""
        return cls(
            gpio_mode=config_dict.get('gpio_mode', 'BCM'),
            doorbell_pin=int(config_dict.get('doorbell_pin', 18)),
            status_led_pins=config_dict.get('status_led_pins', {
                'red': 16,
                'yellow': 20,
                'green': 21
            }),
            debounce_time=float(config_dict.get('debounce_time', 200.0)),
            cleanup_on_exit=bool(config_dict.get('cleanup_on_exit', True)),
            simulate_events=bool(config_dict.get('simulate_events', False)),
            event_interval_range=tuple(config_dict.get('event_interval_range', (30, 120)))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'gpio_mode': self.gpio_mode,
            'doorbell_pin': self.doorbell_pin,
            'status_led_pins': self.status_led_pins,
            'debounce_time': self.debounce_time,
            'cleanup_on_exit': self.cleanup_on_exit,
            'simulate_events': self.simulate_events,
            'event_interval_range': self.event_interval_range
        }


@dataclass
class SensorConfig:
    """Environmental sensor configuration."""
    
    enabled: bool = False
    temperature_sensor: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False,
        'pin': 4,
        'type': 'DHT22'
    })
    humidity_sensor: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False,
        'pin': 4,
        'type': 'DHT22'
    })
    motion_sensor: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False,
        'pin': 17,
        'type': 'PIR'
    })
    
    # Mock/simulation settings
    base_temperature: float = 22.0
    base_humidity: float = 45.0
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SensorConfig':
        """Create config from dictionary."""
        return cls(
            enabled=bool(config_dict.get('enabled', False)),
            temperature_sensor=config_dict.get('temperature_sensor', {
                'enabled': False,
                'pin': 4,
                'type': 'DHT22'
            }),
            humidity_sensor=config_dict.get('humidity_sensor', {
                'enabled': False,
                'pin': 4,
                'type': 'DHT22'
            }),
            motion_sensor=config_dict.get('motion_sensor', {
                'enabled': False,
                'pin': 17,
                'type': 'PIR'
            }),
            base_temperature=float(config_dict.get('base_temperature', 22.0)),
            base_humidity=float(config_dict.get('base_humidity', 45.0))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'enabled': self.enabled,
            'temperature_sensor': self.temperature_sensor,
            'humidity_sensor': self.humidity_sensor,
            'motion_sensor': self.motion_sensor,
            'base_temperature': self.base_temperature,
            'base_humidity': self.base_humidity
        }


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    
    camera_buffer_size: int = 3
    gpio_debounce_time: float = 0.2
    health_check_interval: float = 60.0
    enable_hardware_acceleration: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PerformanceConfig':
        """Create config from dictionary."""
        return cls(
            camera_buffer_size=int(config_dict.get('camera_buffer_size', 3)),
            gpio_debounce_time=float(config_dict.get('gpio_debounce_time', 0.2)),
            health_check_interval=float(config_dict.get('health_check_interval', 60.0)),
            enable_hardware_acceleration=bool(config_dict.get('enable_hardware_acceleration', True))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'camera_buffer_size': self.camera_buffer_size,
            'gpio_debounce_time': self.gpio_debounce_time,
            'health_check_interval': self.health_check_interval,
            'enable_hardware_acceleration': self.enable_hardware_acceleration
        }


@dataclass
class HardwareConfig:
    """Comprehensive hardware configuration."""
    
    # Platform configuration
    platform_override: Optional[str] = None
    mock_mode: bool = False
    auto_detect_hardware: bool = True
    
    # Component configurations
    camera_config: CameraConfig = field(default_factory=CameraConfig)
    gpio_config: GPIOConfig = field(default_factory=GPIOConfig)
    sensor_config: SensorConfig = field(default_factory=SensorConfig)
    performance_config: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HardwareConfig':
        """
        Create hardware config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        
        Returns:
            HardwareConfig instance
        """
        return cls(
            platform_override=config_dict.get('platform_override'),
            mock_mode=bool(config_dict.get('mock_mode', False)),
            auto_detect_hardware=bool(config_dict.get('auto_detect_hardware', True)),
            camera_config=CameraConfig.from_dict(config_dict.get('camera_config', {})),
            gpio_config=GPIOConfig.from_dict(config_dict.get('gpio_config', {})),
            sensor_config=SensorConfig.from_dict(config_dict.get('sensor_config', {})),
            performance_config=PerformanceConfig.from_dict(config_dict.get('performance_config', {}))
        )
    
    @classmethod
    def from_legacy_config(cls, legacy_settings) -> 'HardwareConfig':
        """
        Create new config from legacy Settings object.
        
        Args:
            legacy_settings: Legacy Settings instance
        
        Returns:
            HardwareConfig instance
        """
        camera_config = CameraConfig(
            resolution=getattr(legacy_settings, 'CAMERA_RESOLUTION', (1280, 720)),
            rotation=getattr(legacy_settings, 'CAMERA_ROTATION', 0),
            brightness=getattr(legacy_settings, 'CAMERA_BRIGHTNESS', 50.0),
            contrast=getattr(legacy_settings, 'CAMERA_CONTRAST', 50.0),
            max_index_to_try=getattr(legacy_settings, 'OPENCV_CAMERA_MAX_INDEX_TO_TRY', 5)
        )
        
        gpio_config = GPIOConfig(
            doorbell_pin=getattr(legacy_settings, 'DOORBELL_PIN', 18),
            status_led_pins=getattr(legacy_settings, 'STATUS_LED_PINS', {
                'red': 16,
                'yellow': 20,
                'green': 21
            }),
            debounce_time=getattr(legacy_settings, 'DEBOUNCE_TIME', 2.0) * 1000  # Convert to ms
        )
        
        return cls(
            camera_config=camera_config,
            gpio_config=gpio_config,
            sensor_config=SensorConfig(),
            performance_config=PerformanceConfig()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'platform_override': self.platform_override,
            'mock_mode': self.mock_mode,
            'auto_detect_hardware': self.auto_detect_hardware,
            'camera_config': self.camera_config.to_dict(),
            'gpio_config': self.gpio_config.to_dict(),
            'sensor_config': self.sensor_config.to_dict(),
            'performance_config': self.performance_config.to_dict()
        }
    
    def to_platform_specific_config(self, platform: str) -> Dict[str, Any]:
        """
        Generate platform-specific configuration.
        
        Args:
            platform: Platform name ('raspberry_pi', 'macos', 'linux', 'windows')
        
        Returns:
            Platform-specific configuration dictionary
        """
        base_config = self.to_dict()
        
        if platform == 'raspberry_pi':
            # Enable Pi-specific features
            base_config['camera_config']['use_gpu_memory'] = True
            base_config['gpio_config']['simulate_events'] = False
        elif platform in ['macos', 'linux', 'windows']:
            # Disable Pi-specific features
            base_config['camera_config']['use_gpu_memory'] = False
            base_config['gpio_config']['simulate_events'] = True
        
        return base_config
    
    @classmethod
    def from_environment(cls) -> 'HardwareConfig':
        """
        Create hardware config from environment variables.
        
        Returns:
            HardwareConfig instance
        """
        mock_mode = os.getenv('HARDWARE_MOCK_MODE', 'false').lower() == 'true'
        
        camera_config = CameraConfig(
            resolution=tuple(map(int, os.getenv('CAMERA_RESOLUTION', '1280,720').split(','))),
            fps=float(os.getenv('CAMERA_FPS', '15.0')),
            brightness=float(os.getenv('CAMERA_BRIGHTNESS', '50.0')),
            contrast=float(os.getenv('CAMERA_CONTRAST', '50.0')),
            rotation=int(os.getenv('CAMERA_ROTATION', '0'))
        )
        
        gpio_config = GPIOConfig(
            gpio_mode=os.getenv('GPIO_MODE', 'BCM'),
            doorbell_pin=int(os.getenv('DOORBELL_PIN', '18')),
            debounce_time=float(os.getenv('DEBOUNCE_TIME', '200.0'))
        )
        
        return cls(
            mock_mode=mock_mode,
            camera_config=camera_config,
            gpio_config=gpio_config,
            sensor_config=SensorConfig(),
            performance_config=PerformanceConfig()
        )


# Default hardware configuration
DEFAULT_HARDWARE_CONFIG = HardwareConfig()


__all__ = [
    'HardwareConfig',
    'CameraConfig',
    'GPIOConfig',
    'SensorConfig',
    'PerformanceConfig',
    'DEFAULT_HARDWARE_CONFIG',
]
