#!/usr/bin/env python3
"""
Pipeline Component Fixtures

Fixtures for testing pipeline workers and components.
"""

import pytest
import threading
import queue
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, Optional


@pytest.fixture
def mock_frame_capture_worker():
    """Mock frame capture worker."""
    worker = Mock()
    
    worker.start = Mock(return_value=True)
    worker.stop = Mock(return_value=True)
    worker.is_running = Mock(return_value=False)
    worker.get_metrics = Mock(
        return_value={
            "frames_captured": 0,
            "frames_dropped": 0,
            "fps": 0.0,
            "queue_size": 0,
        }
    )
    worker.get_health = Mock(return_value={"status": "healthy", "errors": 0})
    worker.capture_frame = Mock()
    
    return worker


@pytest.fixture
def mock_motion_detector_worker():
    """Mock motion detection worker."""
    worker = Mock()
    
    worker.start = Mock(return_value=True)
    worker.stop = Mock(return_value=True)
    worker.is_running = Mock(return_value=False)
    worker.get_metrics = Mock(
        return_value={
            "frames_processed": 0,
            "motion_detected": 0,
            "avg_processing_time": 0.0,
        }
    )
    worker.get_health = Mock(return_value={"status": "healthy", "errors": 0})
    worker.detect_motion = Mock(return_value=False)
    
    return worker


@pytest.fixture
def mock_face_detector_worker():
    """Mock face detection worker."""
    worker = Mock()
    
    worker.start = Mock(return_value=True)
    worker.stop = Mock(return_value=True)
    worker.is_running = Mock(return_value=False)
    worker.get_metrics = Mock(
        return_value={
            "faces_detected": 0,
            "avg_detection_time": 0.0,
            "queue_size": 0,
        }
    )
    worker.get_health = Mock(return_value={"status": "healthy", "errors": 0})
    worker.detect_faces = Mock(return_value=[])
    
    return worker


@pytest.fixture
def mock_face_recognizer_worker():
    """Mock face recognition worker."""
    worker = Mock()
    
    worker.start = Mock(return_value=True)
    worker.stop = Mock(return_value=True)
    worker.is_running = Mock(return_value=False)
    worker.get_metrics = Mock(
        return_value={
            "faces_recognized": 0,
            "avg_recognition_time": 0.0,
            "cache_hits": 0,
        }
    )
    worker.get_health = Mock(return_value={"status": "healthy", "errors": 0})
    worker.recognize_face = Mock(return_value=None)
    
    return worker


@pytest.fixture
def mock_event_processor_worker():
    """Mock event processing worker."""
    worker = Mock()
    
    worker.start = Mock(return_value=True)
    worker.stop = Mock(return_value=True)
    worker.is_running = Mock(return_value=False)
    worker.get_metrics = Mock(
        return_value={
            "events_processed": 0,
            "events_enriched": 0,
            "avg_processing_time": 0.0,
        }
    )
    worker.get_health = Mock(return_value={"status": "healthy", "errors": 0})
    worker.process_event = Mock()
    
    return worker


@pytest.fixture
def mock_pipeline_workers(
    mock_frame_capture_worker,
    mock_motion_detector_worker,
    mock_face_detector_worker,
    mock_face_recognizer_worker,
    mock_event_processor_worker,
):
    """Collection of all pipeline worker mocks."""
    return {
        "frame_capture": mock_frame_capture_worker,
        "motion_detector": mock_motion_detector_worker,
        "face_detector": mock_face_detector_worker,
        "face_recognizer": mock_face_recognizer_worker,
        "event_processor": mock_event_processor_worker,
    }


@pytest.fixture
def pipeline_test_queue():
    """Test queue for pipeline communication."""
    return queue.Queue(maxsize=100)


@pytest.fixture
def pipeline_config():
    """Standard pipeline configuration for testing."""
    return {
        "frame_capture": {
            "enabled": True,
            "worker_count": 1,
            "debounce_time": 1.0,
            "fps": 30,
        },
        "motion_detection": {
            "enabled": True,
            "threshold": 25,
            "min_area": 500,
        },
        "face_detection": {
            "enabled": True,
            "worker_count": 2,
            "detector_type": "cpu",
            "confidence_threshold": 0.5,
        },
        "face_recognition": {
            "enabled": True,
            "worker_count": 1,
            "tolerance": 0.6,
            "model": "hog",
        },
        "event_processing": {
            "enabled": True,
            "max_queue_size": 100,
            "enrichment_enabled": True,
        },
        "monitoring": {
            "health_check_interval": 0.5,
            "performance_monitoring": True,
        },
    }
