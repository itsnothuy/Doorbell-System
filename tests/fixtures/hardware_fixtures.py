#!/usr/bin/env python3
"""
Hardware Mock Fixtures

Comprehensive hardware mocking for camera, GPIO, and other hardware components.
"""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, PropertyMock
from typing import Optional, Callable, Any


@pytest.fixture
def mock_picamera2():
    """Mock PiCamera2 for Raspberry Pi camera testing."""
    camera = Mock()
    
    # Configure camera methods
    camera.create_still_configuration = Mock(return_value={'format': 'RGB888'})
    camera.create_video_configuration = Mock(return_value={'format': 'RGB888'})
    camera.configure = Mock()
    camera.start = Mock()
    camera.stop = Mock()
    camera.close = Mock()
    camera.capture_array = Mock(
        return_value=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    )
    camera.capture_file = Mock()
    
    # Properties
    type(camera).resolution = PropertyMock(return_value=(640, 480))
    type(camera).framerate = PropertyMock(return_value=30)
    
    return camera


@pytest.fixture
def mock_cv2_videocapture():
    """Mock OpenCV VideoCapture for cross-platform camera testing."""
    capture = Mock()
    
    # Configure capture methods
    capture.isOpened = Mock(return_value=True)
    capture.read = Mock(
        return_value=(True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    )
    capture.release = Mock()
    capture.set = Mock(return_value=True)
    capture.get = Mock(return_value=30.0)
    
    return capture


@pytest.fixture
def mock_gpio_module():
    """Mock RPi.GPIO module for GPIO operations."""
    gpio = Mock()
    
    # GPIO modes
    gpio.BCM = 11
    gpio.BOARD = 10
    gpio.IN = 1
    gpio.OUT = 0
    gpio.HIGH = 1
    gpio.LOW = 0
    gpio.PUD_UP = 22
    gpio.PUD_DOWN = 21
    gpio.RISING = 31
    gpio.FALLING = 32
    gpio.BOTH = 33
    
    # GPIO methods
    gpio.setmode = Mock()
    gpio.setwarnings = Mock()
    gpio.setup = Mock()
    gpio.input = Mock(return_value=False)
    gpio.output = Mock()
    gpio.cleanup = Mock()
    gpio.add_event_detect = Mock()
    gpio.remove_event_detect = Mock()
    gpio.add_event_callback = Mock()
    gpio.event_detected = Mock(return_value=False)
    gpio.wait_for_edge = Mock()
    gpio.gpio_function = Mock(return_value=gpio.IN)
    
    return gpio


@pytest.fixture
def mock_hardware_detector():
    """Mock hardware detection for testing different platforms."""
    detector = Mock()
    
    detector.is_raspberry_pi = Mock(return_value=False)
    detector.has_picamera = Mock(return_value=False)
    detector.has_gpio = Mock(return_value=False)
    detector.get_platform = Mock(return_value="linux")
    detector.get_cpu_info = Mock(return_value={"model": "x86_64", "cores": 4})
    
    return detector


@pytest.fixture
def simulated_gpio_events():
    """Simulate GPIO events for testing."""
    
    class GPIOEventSimulator:
        def __init__(self):
            self.callbacks = {}
            self.event_queue = []
            
        def register_callback(self, pin: int, callback: Callable):
            """Register a callback for a GPIO pin."""
            self.callbacks[pin] = callback
            
        def trigger_event(self, pin: int, value: int = 1):
            """Trigger a GPIO event."""
            event = {"pin": pin, "value": value, "timestamp": None}
            self.event_queue.append(event)
            
            if pin in self.callbacks:
                self.callbacks[pin](pin)
                
        def get_events(self):
            """Get all queued events."""
            return self.event_queue.copy()
            
        def clear_events(self):
            """Clear event queue."""
            self.event_queue.clear()
    
    return GPIOEventSimulator()


@pytest.fixture
def mock_camera_with_failures():
    """Mock camera with configurable failure scenarios."""
    camera = Mock()
    
    failure_mode = {"enabled": False, "error": None}
    
    def capture_with_failures(*args, **kwargs):
        if failure_mode["enabled"]:
            raise failure_mode["error"]
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    camera.capture_array = Mock(side_effect=capture_with_failures)
    camera.start = Mock()
    camera.stop = Mock()
    camera.close = Mock()
    camera.failure_mode = failure_mode
    
    return camera
