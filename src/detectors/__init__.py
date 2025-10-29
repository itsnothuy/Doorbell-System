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
from src.detectors.gpu_detector import GPUDetector
from src.detectors.edgetpu_detector import EdgeTPUDetector
from src.detectors.detector_factory import (
    DetectorFactory,
    create_detector,
    MockDetector
)
from src.detectors.model_manager import ModelManager
from src.detectors.hardware_detector import HardwareDetector
from src.detectors.performance_profiler import PerformanceProfiler, BenchmarkResult

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
    'create_detector',
    'ModelManager',
    'HardwareDetector',
    'PerformanceProfiler',
    'BenchmarkResult'
]
