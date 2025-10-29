#!/usr/bin/env python3
"""
Test Data Fixtures

Fixtures for generating and managing test data including images, encodings, and events.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json


@pytest.fixture
def sample_images() -> Dict[str, np.ndarray]:
    """Generate various sample images for testing."""
    images = {}
    
    # Standard face image
    images["face"] = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    cv2.rectangle(images["face"], (50, 50), (150, 150), (255, 255, 255), 2)
    
    # High resolution image
    images["high_res"] = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
    
    # Low resolution image
    images["low_res"] = np.random.randint(0, 255, (160, 120, 3), dtype=np.uint8)
    
    # Grayscale image
    images["grayscale"] = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    
    # Blank image
    images["blank"] = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Noisy image
    images["noisy"] = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    return images


@pytest.fixture
def sample_face_locations() -> List[Tuple[int, int, int, int]]:
    """Generate sample face bounding box locations."""
    return [
        (50, 150, 150, 50),  # (top, right, bottom, left)
        (100, 250, 200, 150),
        (200, 350, 300, 250),
    ]


@pytest.fixture
def sample_face_encodings() -> Dict[str, np.ndarray]:
    """Generate sample face encodings for known people."""
    return {
        "john_doe": np.random.random(128).astype(np.float64),
        "jane_smith": np.random.random(128).astype(np.float64),
        "test_person": np.random.random(128).astype(np.float64),
        "unknown_person": np.random.random(128).astype(np.float64),
    }


@pytest.fixture
def sample_event_payloads() -> Dict[str, Dict[str, Any]]:
    """Generate sample event payloads for testing."""
    return {
        "doorbell_triggered": {
            "event_type": "doorbell_triggered",
            "timestamp": 1635724800.0,
            "data": {"trigger_type": "button_press", "duration": 0.5},
            "source": "gpio_handler",
        },
        "motion_detected": {
            "event_type": "motion_detected",
            "timestamp": 1635724801.0,
            "data": {
                "motion_area": 1500,
                "confidence": 0.9,
                "region": [100, 100, 300, 300],
            },
            "source": "motion_detector",
        },
        "face_detected": {
            "event_type": "face_detected",
            "timestamp": 1635724802.0,
            "data": {
                "face_count": 1,
                "confidence": 0.95,
                "bounding_boxes": [[100, 100, 200, 200]],
            },
            "source": "face_detector",
        },
        "face_recognized": {
            "event_type": "face_recognized",
            "timestamp": 1635724803.0,
            "data": {
                "person_name": "john_doe",
                "confidence": 0.85,
                "is_known": True,
                "match_distance": 0.35,
            },
            "source": "face_recognizer",
        },
        "unknown_person": {
            "event_type": "face_recognized",
            "timestamp": 1635724804.0,
            "data": {
                "person_name": "Unknown",
                "confidence": 0.0,
                "is_known": False,
                "match_distance": 0.75,
            },
            "source": "face_recognizer",
        },
    }


@pytest.fixture
def test_face_dataset(temp_test_dir: Path, sample_images: Dict[str, np.ndarray]):
    """Create a test face dataset with multiple images per person."""
    dataset_dir = temp_test_dir / "data" / "test_dataset"
    dataset_dir.mkdir(exist_ok=True)
    
    people = ["john_doe", "jane_smith", "test_person"]
    dataset = {}
    
    for person in people:
        person_dir = dataset_dir / person
        person_dir.mkdir(exist_ok=True)
        
        images = []
        for i in range(3):  # 3 images per person
            image_path = person_dir / f"{person}_{i:03d}.jpg"
            cv2.imwrite(str(image_path), sample_images["face"])
            images.append(str(image_path))
        
        dataset[person] = images
    
    return {"directory": dataset_dir, "people": dataset}


@pytest.fixture
def performance_test_data() -> Dict[str, Any]:
    """Generate data for performance testing."""
    return {
        "small_batch": list(range(10)),
        "medium_batch": list(range(100)),
        "large_batch": list(range(1000)),
        "frame_sequence": [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(30)
        ],
        "encoding_sequence": [np.random.random(128).astype(np.float64) for _ in range(100)],
    }


@pytest.fixture
def test_config_variants() -> Dict[str, Dict[str, Any]]:
    """Generate various configuration variants for testing."""
    return {
        "minimal": {
            "frame_capture": {"enabled": True},
            "face_detection": {"enabled": False},
            "face_recognition": {"enabled": False},
        },
        "face_detection_only": {
            "frame_capture": {"enabled": True},
            "face_detection": {"enabled": True},
            "face_recognition": {"enabled": False},
        },
        "full_pipeline": {
            "frame_capture": {"enabled": True},
            "face_detection": {"enabled": True},
            "face_recognition": {"enabled": True},
            "event_processing": {"enabled": True},
        },
        "high_performance": {
            "frame_capture": {"enabled": True, "worker_count": 2, "fps": 60},
            "face_detection": {"enabled": True, "worker_count": 4},
            "face_recognition": {"enabled": True, "worker_count": 2},
        },
    }
