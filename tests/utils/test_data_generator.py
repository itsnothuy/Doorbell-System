#!/usr/bin/env python3
"""
Test Data Generator

Utilities for generating realistic test data.
"""

import random
import string
import time
from typing import Dict, List, Any, Optional
import numpy as np


class TestDataGenerator:
    """Generate test data for various scenarios."""

    @staticmethod
    def generate_person_name() -> str:
        """Generate a random person name."""
        first_names = [
            "John",
            "Jane",
            "Alice",
            "Bob",
            "Charlie",
            "Diana",
            "Edward",
            "Fiona",
        ]
        last_names = [
            "Smith",
            "Johnson",
            "Williams",
            "Brown",
            "Jones",
            "Garcia",
            "Miller",
            "Davis",
        ]
        return f"{random.choice(first_names)} {random.choice(last_names)}"

    @staticmethod
    def generate_event_sequence(count: int = 10) -> List[Dict[str, Any]]:
        """Generate a sequence of events."""
        events = []
        base_time = time.time()

        event_types = [
            "doorbell_triggered",
            "motion_detected",
            "face_detected",
            "face_recognized",
        ]

        for i in range(count):
            event = {
                "id": i,
                "event_type": random.choice(event_types),
                "timestamp": base_time + (i * 0.5),
                "data": TestDataGenerator._generate_event_data(),
                "source": random.choice(["gpio", "motion_detector", "face_detector"]),
            }
            events.append(event)

        return events

    @staticmethod
    def _generate_event_data() -> Dict[str, Any]:
        """Generate random event data."""
        return {
            "confidence": random.uniform(0.7, 1.0),
            "duration": random.uniform(0.1, 2.0),
            "metadata": {"random_key": "random_value"},
        }

    @staticmethod
    def generate_face_encodings(count: int = 10) -> Dict[str, np.ndarray]:
        """Generate random face encodings."""
        encodings = {}
        for i in range(count):
            name = f"person_{i}"
            encoding = np.random.random(128).astype(np.float64)
            encodings[name] = encoding
        return encodings

    @staticmethod
    def generate_image_sequence(
        count: int = 30, width: int = 640, height: int = 480
    ) -> List[np.ndarray]:
        """Generate a sequence of test images."""
        images = []
        for _ in range(count):
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            images.append(image)
        return images

    @staticmethod
    def generate_config(scenario: str = "default") -> Dict[str, Any]:
        """Generate test configuration."""
        configs = {
            "default": {
                "frame_capture": {"enabled": True, "fps": 30},
                "face_detection": {"enabled": True, "confidence_threshold": 0.5},
                "face_recognition": {"enabled": True, "tolerance": 0.6},
            },
            "high_performance": {
                "frame_capture": {"enabled": True, "fps": 60, "worker_count": 2},
                "face_detection": {
                    "enabled": True,
                    "confidence_threshold": 0.5,
                    "worker_count": 4,
                },
                "face_recognition": {
                    "enabled": True,
                    "tolerance": 0.6,
                    "worker_count": 2,
                },
            },
            "minimal": {
                "frame_capture": {"enabled": True, "fps": 15},
                "face_detection": {"enabled": False},
                "face_recognition": {"enabled": False},
            },
        }
        return configs.get(scenario, configs["default"])

    @staticmethod
    def generate_load_pattern(pattern_type: str = "constant") -> List[float]:
        """Generate load pattern for testing."""
        duration = 60  # seconds
        if pattern_type == "constant":
            return [10.0] * duration  # 10 events/second constant
        elif pattern_type == "ramp_up":
            return [i / 10.0 for i in range(duration)]
        elif pattern_type == "spike":
            pattern = [5.0] * duration
            # Add spikes at regular intervals
            for i in range(0, duration, 10):
                pattern[i] = 50.0
            return pattern
        elif pattern_type == "random":
            return [random.uniform(1.0, 20.0) for _ in range(duration)]
        else:
            return [10.0] * duration


class ScenarioBuilder:
    """Build test scenarios."""

    def __init__(self):
        self.scenario = {"steps": [], "expected_outcomes": []}

    def add_step(self, action: str, data: Dict[str, Any]) -> "ScenarioBuilder":
        """Add a step to the scenario."""
        self.scenario["steps"].append({"action": action, "data": data})
        return self

    def expect_outcome(self, outcome: str, criteria: Dict[str, Any]) -> "ScenarioBuilder":
        """Add expected outcome."""
        self.scenario["expected_outcomes"].append({"outcome": outcome, "criteria": criteria})
        return self

    def build(self) -> Dict[str, Any]:
        """Build the scenario."""
        return self.scenario

    @staticmethod
    def doorbell_recognition_scenario() -> Dict[str, Any]:
        """Create a complete doorbell recognition scenario."""
        builder = ScenarioBuilder()
        return (
            builder.add_step("trigger_doorbell", {"source": "button"})
            .add_step("capture_frame", {"resolution": (640, 480)})
            .add_step("detect_motion", {"threshold": 25})
            .add_step("detect_faces", {"min_confidence": 0.5})
            .add_step("recognize_faces", {"tolerance": 0.6})
            .expect_outcome(
                "face_recognized", {"person_identified": True, "confidence": ">0.8"}
            )
            .expect_outcome("notification_sent", {"channel": "telegram", "success": True})
            .build()
        )

    @staticmethod
    def unknown_person_scenario() -> Dict[str, Any]:
        """Create unknown person detection scenario."""
        builder = ScenarioBuilder()
        return (
            builder.add_step("trigger_doorbell", {"source": "button"})
            .add_step("capture_frame", {"resolution": (640, 480)})
            .add_step("detect_faces", {"min_confidence": 0.5})
            .add_step("recognize_faces", {"tolerance": 0.6})
            .expect_outcome("unknown_person", {"person_identified": False})
            .expect_outcome("alert_triggered", {"alert_type": "unknown_visitor"})
            .build()
        )
