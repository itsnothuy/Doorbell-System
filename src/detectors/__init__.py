#!/usr/bin/env python3
"""
Face Detectors Module

Strategy pattern implementation for pluggable face detection backends.
"""

from src.detectors.base_detector import (
    BaseDetector,
    DetectorType,
    ModelType,
    DetectionMetrics,
    FaceDetectionResult
)
from src.detectors.detection_result import DetectionResult
from src.detectors.cpu_detector import CPUDetector
from src.detectors.detector_factory import (
    DetectorFactory,
    create_detector,
    GPUDetector,
    EdgeTPUDetector,
    MockDetector
)

__all__ = [
    'BaseDetector',
    'DetectorType',
    'ModelType',
    'DetectionMetrics',
    'FaceDetectionResult',
    'DetectionResult',
    'CPUDetector',
    'GPUDetector',
    'EdgeTPUDetector',
    'MockDetector',
    'DetectorFactory',
    'create_detector'
]
