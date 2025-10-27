"""
Hardware Abstraction Layer (HAL) for cross-platform hardware support

This module provides a unified interface for hardware components across different
platforms including Raspberry Pi, macOS, Linux, and Windows. It includes:
- Abstract hardware interfaces
- Platform-specific implementations
- Mock implementations for testing
- Runtime hardware detection and switching
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from src.platform_detector import platform_detector

logger = logging.getLogger(__name__)


class HardwareType(Enum):
    """Supported hardware types"""
    CAMERA = "camera"
    GPIO = "gpio"
    SENSOR = "sensor"


class HardwareStatus(Enum):
    """Hardware component status"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    MOCK = "mock"
    ERROR = "error"


@dataclass
class HardwareChange:
    """Represents a hardware state change"""
    hardware_type: HardwareType
    previous_status: HardwareStatus
    current_status: HardwareStatus
    timestamp: float
    details: Optional[Dict[str, Any]] = None


@dataclass
class HardwareHealthStatus:
    """Overall hardware health status"""
    healthy: bool
    components: Dict[str, HardwareStatus]
    errors: List[str]
    warnings: List[str]
    timestamp: float


class HardwareAbstractionLayer:
    """
    Central hardware abstraction layer coordinator.
    
    Manages all hardware components with platform detection, automatic fallback,
    and runtime hardware switching capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Hardware Abstraction Layer.
        
        Args:
            config: Optional hardware configuration dictionary
        """
        self.config = config or {}
        self.platform_info = {
            'system': platform_detector.system,
            'machine': platform_detector.machine,
            'is_raspberry_pi': platform_detector.is_raspberry_pi,
            'is_macos': platform_detector.is_macos,
            'is_linux': platform_detector.is_linux,
            'is_windows': platform_detector.is_windows,
        }
        
        # Hardware components
        self.camera_handler = None
        self.gpio_handler = None
        self.sensor_handler = None
        
        # Hardware state tracking
        self.hardware_available = {}
        self.mock_mode = self.config.get('mock_mode', False)
        self.auto_detect = self.config.get('auto_detect_hardware', True)
        
        logger.info(f"Hardware Abstraction Layer initialized for {self.platform_info['system']}")
        
        # Initialize hardware components
        if self.auto_detect:
            self._initialize_hardware()
    
    def _initialize_hardware(self) -> None:
        """Initialize all hardware components based on platform."""
        try:
            # Initialize camera
            camera_config = self.config.get('camera_config', {})
            self.camera_handler = self._get_camera_handler(camera_config)
            self.hardware_available[HardwareType.CAMERA] = (
                HardwareStatus.MOCK if self.mock_mode 
                else HardwareStatus.AVAILABLE if self.camera_handler 
                else HardwareStatus.UNAVAILABLE
            )
            
            # Initialize GPIO
            gpio_config = self.config.get('gpio_config', {})
            self.gpio_handler = self._get_gpio_handler(gpio_config)
            self.hardware_available[HardwareType.GPIO] = (
                HardwareStatus.MOCK if self.mock_mode or not platform_detector.is_raspberry_pi
                else HardwareStatus.AVAILABLE if self.gpio_handler
                else HardwareStatus.UNAVAILABLE
            )
            
            # Initialize sensors (if configured)
            sensor_config = self.config.get('sensor_config', {})
            if sensor_config.get('enabled', False):
                self.sensor_handler = self._get_sensor_handler(sensor_config)
                self.hardware_available[HardwareType.SENSOR] = (
                    HardwareStatus.MOCK if self.mock_mode
                    else HardwareStatus.AVAILABLE if self.sensor_handler
                    else HardwareStatus.UNAVAILABLE
                )
            
            logger.info(f"Hardware initialization complete: {self.hardware_available}")
            
        except Exception as e:
            logger.error(f"Hardware initialization failed: {e}")
            # Fallback to mock mode on initialization failure
            self.switch_to_mock_mode()
    
    def _get_camera_handler(self, config: Dict[str, Any]):
        """Get appropriate camera handler for current platform."""
        if self.mock_mode:
            from src.hardware.mock.mock_camera import MockCameraHandler
            return MockCameraHandler(config)
        
        if platform_detector.is_raspberry_pi:
            try:
                from src.hardware.platform.raspberry_pi import RaspberryPiCameraHandler
                return RaspberryPiCameraHandler(config)
            except ImportError:
                logger.warning("Raspberry Pi camera handler not available, using mock")
                from src.hardware.mock.mock_camera import MockCameraHandler
                return MockCameraHandler(config)
        
        elif platform_detector.is_macos:
            from src.hardware.platform.macos import MacOSCameraHandler
            return MacOSCameraHandler(config)
        
        elif platform_detector.is_linux:
            from src.hardware.platform.linux import LinuxCameraHandler
            return LinuxCameraHandler(config)
        
        elif platform_detector.is_windows:
            from src.hardware.platform.windows import WindowsCameraHandler
            return WindowsCameraHandler(config)
        
        else:
            # Fallback to mock for unknown platforms
            from src.hardware.mock.mock_camera import MockCameraHandler
            return MockCameraHandler(config)
    
    def _get_gpio_handler(self, config: Dict[str, Any]):
        """Get appropriate GPIO handler for current platform."""
        if self.mock_mode or not platform_detector.is_raspberry_pi:
            from src.hardware.mock.mock_gpio import MockGPIOHandler
            return MockGPIOHandler(config)
        
        try:
            from src.hardware.platform.raspberry_pi import RaspberryPiGPIOHandler
            return RaspberryPiGPIOHandler(config)
        except ImportError:
            logger.warning("Raspberry Pi GPIO handler not available, using mock")
            from src.hardware.mock.mock_gpio import MockGPIOHandler
            return MockGPIOHandler(config)
    
    def _get_sensor_handler(self, config: Dict[str, Any]):
        """Get appropriate sensor handler for current platform."""
        if self.mock_mode:
            from src.hardware.mock.mock_sensors import MockSensorHandler
            return MockSensorHandler(config)
        
        if platform_detector.is_raspberry_pi:
            try:
                from src.hardware.platform.raspberry_pi import RaspberryPiSensorHandler
                return RaspberryPiSensorHandler(config)
            except ImportError:
                logger.warning("Raspberry Pi sensor handler not available, using mock")
                from src.hardware.mock.mock_sensors import MockSensorHandler
                return MockSensorHandler(config)
        
        # Default to mock for non-Pi platforms
        from src.hardware.mock.mock_sensors import MockSensorHandler
        return MockSensorHandler(config)
    
    def get_camera_handler(self):
        """Get camera handler instance."""
        return self.camera_handler
    
    def get_gpio_handler(self):
        """Get GPIO handler instance."""
        return self.gpio_handler
    
    def get_sensor_handler(self):
        """Get sensor handler instance."""
        return self.sensor_handler
    
    def detect_hardware_changes(self) -> List[HardwareChange]:
        """
        Detect hardware changes at runtime.
        
        Returns:
            List of detected hardware changes
        """
        changes = []
        # TODO: Implement runtime hardware detection
        # This would involve checking for newly connected/disconnected hardware
        return changes
    
    def switch_to_mock_mode(self) -> None:
        """Switch all hardware to mock implementations."""
        logger.info("Switching to mock mode for all hardware")
        
        self.mock_mode = True
        
        # Cleanup existing handlers
        if self.camera_handler and hasattr(self.camera_handler, 'cleanup'):
            try:
                self.camera_handler.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up camera handler: {e}")
        
        if self.gpio_handler and hasattr(self.gpio_handler, 'cleanup'):
            try:
                self.gpio_handler.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up GPIO handler: {e}")
        
        # Initialize mock handlers
        self._initialize_hardware()
    
    def health_check(self) -> HardwareHealthStatus:
        """
        Perform comprehensive hardware health check.
        
        Returns:
            Hardware health status with component details
        """
        import time
        
        errors = []
        warnings = []
        components = {}
        
        # Check camera
        if self.camera_handler:
            try:
                if hasattr(self.camera_handler, 'is_available') and self.camera_handler.is_available():
                    components['camera'] = HardwareStatus.AVAILABLE
                elif hasattr(self.camera_handler, 'is_initialized') and self.camera_handler.is_initialized:
                    components['camera'] = HardwareStatus.AVAILABLE
                else:
                    components['camera'] = HardwareStatus.UNAVAILABLE
                    warnings.append("Camera not properly initialized")
            except Exception as e:
                components['camera'] = HardwareStatus.ERROR
                errors.append(f"Camera error: {str(e)}")
        else:
            components['camera'] = HardwareStatus.UNAVAILABLE
            warnings.append("Camera handler not initialized")
        
        # Check GPIO
        if self.gpio_handler:
            try:
                if hasattr(self.gpio_handler, 'initialized') and self.gpio_handler.initialized:
                    components['gpio'] = HardwareStatus.AVAILABLE
                else:
                    components['gpio'] = HardwareStatus.UNAVAILABLE
                    warnings.append("GPIO not properly initialized")
            except Exception as e:
                components['gpio'] = HardwareStatus.ERROR
                errors.append(f"GPIO error: {str(e)}")
        else:
            components['gpio'] = HardwareStatus.UNAVAILABLE
            warnings.append("GPIO handler not initialized")
        
        # Check sensors
        if self.sensor_handler:
            try:
                if hasattr(self.sensor_handler, 'is_available') and self.sensor_handler.is_available():
                    components['sensor'] = HardwareStatus.AVAILABLE
                else:
                    components['sensor'] = HardwareStatus.UNAVAILABLE
            except Exception as e:
                components['sensor'] = HardwareStatus.ERROR
                errors.append(f"Sensor error: {str(e)}")
        
        healthy = len(errors) == 0 and components.get('camera') != HardwareStatus.ERROR
        
        return HardwareHealthStatus(
            healthy=healthy,
            components=components,
            errors=errors,
            warnings=warnings,
            timestamp=time.time()
        )
    
    def cleanup(self) -> None:
        """Cleanup all hardware resources."""
        logger.info("Cleaning up Hardware Abstraction Layer")
        
        if self.camera_handler and hasattr(self.camera_handler, 'cleanup'):
            try:
                self.camera_handler.cleanup()
            except Exception as e:
                logger.error(f"Camera cleanup error: {e}")
        
        if self.gpio_handler and hasattr(self.gpio_handler, 'cleanup'):
            try:
                self.gpio_handler.cleanup()
            except Exception as e:
                logger.error(f"GPIO cleanup error: {e}")
        
        if self.sensor_handler and hasattr(self.sensor_handler, 'cleanup'):
            try:
                self.sensor_handler.cleanup()
            except Exception as e:
                logger.error(f"Sensor cleanup error: {e}")


# Global HAL instance (lazy initialization)
_hal_instance: Optional[HardwareAbstractionLayer] = None


def get_hal(config: Optional[Dict[str, Any]] = None) -> HardwareAbstractionLayer:
    """
    Get or create the global Hardware Abstraction Layer instance.
    
    Args:
        config: Optional configuration dictionary
    
    Returns:
        HardwareAbstractionLayer instance
    """
    global _hal_instance
    
    if _hal_instance is None:
        _hal_instance = HardwareAbstractionLayer(config)
    
    return _hal_instance


__all__ = [
    'HardwareAbstractionLayer',
    'HardwareType',
    'HardwareStatus',
    'HardwareChange',
    'HardwareHealthStatus',
    'get_hal',
]
