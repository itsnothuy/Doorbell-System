"""
Abstract base classes for hardware interfaces

Defines the contracts for all hardware components to ensure consistent
behavior across different platform implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class GPIOMode(Enum):
    """GPIO pin modes"""
    INPUT = "input"
    OUTPUT = "output"


class GPIOEdge(Enum):
    """GPIO interrupt edge detection"""
    RISING = "rising"
    FALLING = "falling"
    BOTH = "both"


@dataclass
class CameraInfo:
    """Camera hardware information"""
    name: str
    resolution: Tuple[int, int]
    fps: float
    backend: str
    available: bool


@dataclass
class CameraSettings:
    """Camera configuration settings"""
    resolution: Tuple[int, int] = (1280, 720)
    fps: float = 15.0
    brightness: float = 50.0
    contrast: float = 50.0
    rotation: int = 0


class CameraHandler(ABC):
    """Abstract camera handler interface for cross-platform camera support."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize camera hardware.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture single frame from camera.
        
        Returns:
            Numpy array containing the captured frame in RGB format,
            or None if capture failed
        """
        pass
    
    @abstractmethod
    def start_stream(self) -> bool:
        """
        Start continuous camera stream.
        
        Returns:
            True if stream started successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def stop_stream(self) -> None:
        """Stop camera stream."""
        pass
    
    @abstractmethod
    def get_camera_info(self) -> CameraInfo:
        """
        Get camera hardware information.
        
        Returns:
            CameraInfo object with camera details
        """
        pass
    
    @abstractmethod
    def set_camera_settings(self, settings: CameraSettings) -> bool:
        """
        Configure camera settings.
        
        Args:
            settings: CameraSettings object with desired configuration
        
        Returns:
            True if settings applied successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if camera is available.
        
        Returns:
            True if camera is available and functional, False otherwise
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup camera resources and release hardware."""
        pass


class GPIOHandler(ABC):
    """Abstract GPIO handler interface for cross-platform GPIO support."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize GPIO hardware.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def setup_pin(self, pin: int, mode: GPIOMode, **kwargs) -> bool:
        """
        Setup GPIO pin with specified mode.
        
        Args:
            pin: GPIO pin number
            mode: Pin mode (INPUT or OUTPUT)
            **kwargs: Additional platform-specific options like pull_up_down, initial
        
        Returns:
            True if pin setup successful, False otherwise
        """
        pass
    
    @abstractmethod
    def read_pin(self, pin: int) -> Optional[bool]:
        """
        Read digital value from GPIO pin.
        
        Args:
            pin: GPIO pin number
        
        Returns:
            Pin state (True=HIGH, False=LOW), or None if read failed
        """
        pass
    
    @abstractmethod
    def write_pin(self, pin: int, value: bool) -> bool:
        """
        Write digital value to GPIO pin.
        
        Args:
            pin: GPIO pin number
            value: Value to write (True=HIGH, False=LOW)
        
        Returns:
            True if write successful, False otherwise
        """
        pass
    
    @abstractmethod
    def setup_interrupt(self, pin: int, callback: Callable, edge: GPIOEdge) -> bool:
        """
        Setup interrupt handler for GPIO pin.
        
        Args:
            pin: GPIO pin number
            callback: Function to call on interrupt
            edge: Edge detection mode (RISING, FALLING, BOTH)
        
        Returns:
            True if interrupt setup successful, False otherwise
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup GPIO resources and reset pins."""
        pass


class SensorHandler(ABC):
    """Abstract sensor handler interface for environmental sensors."""
    
    @abstractmethod
    def read_temperature(self) -> Optional[float]:
        """
        Read temperature from sensor.
        
        Returns:
            Temperature in Celsius, or None if read failed
        """
        pass
    
    @abstractmethod
    def read_humidity(self) -> Optional[float]:
        """
        Read humidity from sensor.
        
        Returns:
            Relative humidity percentage (0-100), or None if read failed
        """
        pass
    
    @abstractmethod
    def read_motion_sensor(self) -> Optional[bool]:
        """
        Read motion sensor state.
        
        Returns:
            True if motion detected, False otherwise, or None if read failed
        """
        pass
    
    @abstractmethod
    def get_sensor_status(self) -> Dict[str, Any]:
        """
        Get status of all sensors.
        
        Returns:
            Dictionary with sensor status information
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if sensors are available.
        
        Returns:
            True if sensors are available and functional, False otherwise
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup sensor resources."""
        pass


__all__ = [
    'CameraHandler',
    'GPIOHandler',
    'SensorHandler',
    'CameraInfo',
    'CameraSettings',
    'GPIOMode',
    'GPIOEdge',
]
