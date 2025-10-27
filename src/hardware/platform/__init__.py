"""Platform-specific hardware implementations."""

from src.hardware.platform.raspberry_pi import (
    RaspberryPiCameraHandler,
    RaspberryPiGPIOHandler,
    RaspberryPiSensorHandler,
)
from src.hardware.platform.macos import MacOSCameraHandler
from src.hardware.platform.linux import LinuxCameraHandler
from src.hardware.platform.windows import WindowsCameraHandler

__all__ = [
    'RaspberryPiCameraHandler',
    'RaspberryPiGPIOHandler',
    'RaspberryPiSensorHandler',
    'MacOSCameraHandler',
    'LinuxCameraHandler',
    'WindowsCameraHandler',
]
