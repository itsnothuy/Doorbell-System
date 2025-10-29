#!/usr/bin/env python3
"""
Comprehensive Hardware Mocks

Complete mock implementations for hardware components.
"""

import numpy as np
import threading
import time
from typing import Optional, Callable, Any, Dict, List
from unittest.mock import Mock


class MockCamera:
    """Comprehensive mock camera implementation."""

    def __init__(self, resolution=(640, 480), fps=30):
        self.resolution = resolution
        self.fps = fps
        self.is_open = False
        self.frame_count = 0
        self.failure_mode = None
        self._callbacks = []

    def start(self):
        """Start camera."""
        if self.failure_mode == "start_failure":
            raise RuntimeError("Failed to start camera")
        self.is_open = True

    def stop(self):
        """Stop camera."""
        self.is_open = False

    def close(self):
        """Close camera."""
        self.is_open = False

    def capture_array(self) -> np.ndarray:
        """Capture a frame."""
        if not self.is_open:
            raise RuntimeError("Camera not started")

        if self.failure_mode == "capture_failure":
            raise RuntimeError("Failed to capture frame")

        self.frame_count += 1
        return self._generate_frame()

    def _generate_frame(self) -> np.ndarray:
        """Generate a test frame."""
        frame = np.random.randint(0, 255, (*self.resolution[::-1], 3), dtype=np.uint8)

        # Add frame counter text
        import cv2

        cv2.putText(
            frame,
            f"Frame {self.frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        return frame

    def set_failure_mode(self, mode: Optional[str]):
        """Set failure mode for testing."""
        self.failure_mode = mode


class MockGPIOHandler:
    """Comprehensive mock GPIO handler."""

    def __init__(self):
        self.pin_modes = {}
        self.pin_states = {}
        self.event_callbacks = {}
        self.event_detection_enabled = {}
        self.is_setup = False

    def setup(self, pin: int, mode: int, pull_up_down: Optional[int] = None):
        """Setup a GPIO pin."""
        self.pin_modes[pin] = mode
        self.pin_states[pin] = 0
        self.is_setup = True

    def input(self, pin: int) -> int:
        """Read GPIO pin state."""
        if pin not in self.pin_states:
            raise ValueError(f"Pin {pin} not setup")
        return self.pin_states[pin]

    def output(self, pin: int, state: int):
        """Set GPIO pin state."""
        if pin not in self.pin_states:
            raise ValueError(f"Pin {pin} not setup")
        self.pin_states[pin] = state

    def add_event_detect(
        self, pin: int, edge: int, callback: Optional[Callable] = None, bouncetime: int = 0
    ):
        """Add event detection to a pin."""
        self.event_detection_enabled[pin] = True
        if callback:
            self.event_callbacks[pin] = callback

    def remove_event_detect(self, pin: int):
        """Remove event detection from a pin."""
        self.event_detection_enabled[pin] = False
        if pin in self.event_callbacks:
            del self.event_callbacks[pin]

    def trigger_event(self, pin: int, state: int = 1):
        """Simulate a GPIO event."""
        self.pin_states[pin] = state
        if pin in self.event_callbacks and self.event_detection_enabled.get(pin):
            self.event_callbacks[pin](pin)

    def cleanup(self):
        """Cleanup GPIO resources."""
        self.pin_modes.clear()
        self.pin_states.clear()
        self.event_callbacks.clear()
        self.event_detection_enabled.clear()
        self.is_setup = False


class MockMessageBus:
    """Comprehensive mock message bus implementation."""

    def __init__(self):
        self.subscriptions: Dict[str, List[Callable]] = {}
        self.published_messages: List[Dict[str, Any]] = []
        self.running = False

    def start(self):
        """Start message bus."""
        self.running = True

    def stop(self):
        """Stop message bus."""
        self.running = False

    def publish(self, topic: str, message: Any):
        """Publish a message."""
        if not self.running:
            raise RuntimeError("Message bus not running")

        self.published_messages.append({"topic": topic, "message": message, "timestamp": time.time()})

        # Call subscribers
        if topic in self.subscriptions:
            for callback in self.subscriptions[topic]:
                try:
                    callback(message)
                except Exception:
                    pass

    def subscribe(self, topic: str, callback: Callable) -> str:
        """Subscribe to a topic."""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        self.subscriptions[topic].append(callback)
        return f"sub_{topic}_{len(self.subscriptions[topic])}"

    def unsubscribe(self, topic: str, subscription_id: str):
        """Unsubscribe from a topic."""
        # Simplified implementation
        pass

    def get_published_messages(self, topic: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get published messages."""
        if topic is None:
            return self.published_messages.copy()
        return [m for m in self.published_messages if m["topic"] == topic]

    def clear_messages(self):
        """Clear published messages."""
        self.published_messages.clear()

    def get_metrics(self) -> Dict[str, int]:
        """Get message bus metrics."""
        return {
            "messages_sent": len(self.published_messages),
            "active_subscriptions": sum(len(subs) for subs in self.subscriptions.values()),
        }


class MockDetector:
    """Mock face detector for testing."""

    def __init__(self, detector_type: str = "cpu"):
        self.detector_type = detector_type
        self.detection_count = 0
        self.failure_mode = None

    def detect_faces(self, image: np.ndarray) -> List[tuple]:
        """Detect faces in an image."""
        if self.failure_mode == "detection_failure":
            raise RuntimeError("Detection failed")

        self.detection_count += 1

        # Return mock detections
        return [(50, 150, 150, 50)]  # (top, right, bottom, left)

    def get_model_info(self) -> Dict[str, str]:
        """Get detector model information."""
        return {"type": self.detector_type, "version": "1.0.0"}


class MockFaceRecognizer:
    """Mock face recognizer for testing."""

    def __init__(self):
        self.known_faces = {}
        self.recognition_count = 0

    def add_known_face(self, name: str, encoding: np.ndarray):
        """Add a known face."""
        self.known_faces[name] = encoding

    def recognize_face(self, encoding: np.ndarray, tolerance: float = 0.6) -> Optional[str]:
        """Recognize a face."""
        self.recognition_count += 1

        # Simplified recognition logic for testing
        for name, known_encoding in self.known_faces.items():
            distance = np.linalg.norm(encoding - known_encoding)
            if distance < tolerance:
                return name

        return None

    def get_known_faces(self) -> List[str]:
        """Get list of known face names."""
        return list(self.known_faces.keys())
