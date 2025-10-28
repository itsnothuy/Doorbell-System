#!/usr/bin/env python3
"""
Test suite for Detector Benchmarking System

Tests for performance benchmarking, comparative analysis, and regression detection.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Simple mock for testing without numpy
    class np:
        uint8 = int
        
        @staticmethod
        def zeros(shape, dtype=None):
            class MockArray:
                def __init__(self, shape):
                    self.shape = shape
                    self.size = shape[0] * shape[1] * (shape[2] if len(shape) > 2 else 1)
            return MockArray(shape)
        
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0
        
        @staticmethod
        def std(values):
            if not values or len(values) < 2:
                return 0
            mean_val = np.mean(values)
            return (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5

from src.detectors.benchmarking import (
    DetectorBenchmark,
    BenchmarkMetrics
)
from src.detectors.base_detector import DetectorType, ModelType, FaceDetectionResult, DetectionMetrics
from src.detectors.detector_factory import MockDetector


class TestBenchmarkMetrics(unittest.TestCase):
    """Test suite for BenchmarkMetrics dataclass."""
    
    def test_metrics_initialization(self):
        """Test BenchmarkMetrics initialization."""
        metrics = BenchmarkMetrics(
            detector_type='cpu',
            model_type='hog'
        )
        
        self.assertEqual(metrics.detector_type, 'cpu')
        self.assertEqual(metrics.model_type, 'hog')
        self.assertEqual(metrics.avg_inference_time, 0.0)
        self.assertEqual(metrics.total_images_processed, 0)
    
    def test_metrics_to_dict(self):
        """Test metrics conversion to dictionary."""
        metrics = BenchmarkMetrics(
            detector_type='cpu',
            model_type='hog',
            avg_inference_time=0.1,
            images_per_second=10.0
        )
        
        result = metrics.to_dict()
        
        self.assertIn('detector_type', result)
        self.assertIn('performance', result)
        self.assertIn('resources', result)
        self.assertIn('accuracy', result)
        self.assertEqual(result['detector_type'], 'cpu')
        self.assertAlmostEqual(result['performance']['avg_inference_time_ms'], 100.0)


class TestDetectorBenchmark(unittest.TestCase):
    """Test suite for DetectorBenchmark system."""
    
    def setUp(self):
        """Set up test environment."""
        self.benchmark = DetectorBenchmark()
        
        # Create test images
        if NUMPY_AVAILABLE:
            self.test_images = [
                np.zeros((100, 100, 3), dtype=np.uint8)
                for _ in range(5)
            ]
        else:
            self.test_images = [
                np.zeros((100, 100, 3))
                for _ in range(5)
            ]
    
    def test_benchmark_initialization(self):
        """Test benchmark system initialization."""
        benchmark = DetectorBenchmark({'param': 'value'})
        
        self.assertIsNotNone(benchmark)
        self.assertEqual(benchmark.config, {'param': 'value'})
        self.assertEqual(len(benchmark.results_history), 0)
        self.assertIsNone(benchmark.baseline_metrics)
    
    def test_benchmark_detector_basic(self):
        """Test basic detector benchmarking."""
        # Create mock detector
        detector = MockDetector({})
        
        # Mock the detect_faces method to return predictable results
        mock_face = FaceDetectionResult(
            bounding_box=(10, 90, 90, 10),
            confidence=0.95
        )
        mock_metrics = DetectionMetrics(
            inference_time=0.01,
            total_time=0.015,
            face_count=1
        )
        
        with patch.object(detector, 'detect_faces', return_value=([mock_face], mock_metrics)):
            metrics = self.benchmark.benchmark_detector(
                detector,
                self.test_images,
                iterations=10,
                warmup_iterations=2
            )
        
        # Verify metrics
        self.assertEqual(metrics.detector_type, 'mock')
        self.assertEqual(metrics.total_images_processed, 10)
        self.assertGreater(metrics.images_per_second, 0)
        self.assertGreater(metrics.benchmark_duration, 0)
    
    def test_benchmark_detector_no_images(self):
        """Test benchmarking with no images raises error."""
        detector = MockDetector({})
        
        with self.assertRaises(ValueError):
            self.benchmark.benchmark_detector(detector, [], iterations=10)
    
    def test_benchmark_detector_performance_metrics(self):
        """Test that performance metrics are calculated correctly."""
        detector = MockDetector({})
        
        # Mock varying inference times
        inference_times = [0.01, 0.02, 0.015, 0.018, 0.012]
        call_count = [0]
        
        def mock_detect(image):
            idx = call_count[0] % len(inference_times)
            call_count[0] += 1
            
            metrics = DetectionMetrics(
                inference_time=inference_times[idx],
                total_time=inference_times[idx] + 0.005
            )
            return ([], metrics)
        
        with patch.object(detector, 'detect_faces', side_effect=mock_detect):
            metrics = self.benchmark.benchmark_detector(
                detector,
                self.test_images,
                iterations=10,
                warmup_iterations=0
            )
        
        # Check that timing metrics are captured
        self.assertGreater(metrics.avg_inference_time, 0)
        self.assertGreater(metrics.min_inference_time, 0)
        self.assertGreater(metrics.max_inference_time, 0)
    
    def test_benchmark_detector_with_detections(self):
        """Test benchmarking with actual face detections."""
        detector = MockDetector({})
        
        mock_face = FaceDetectionResult(
            bounding_box=(10, 90, 90, 10),
            confidence=0.85
        )
        mock_metrics = DetectionMetrics(
            inference_time=0.01,
            total_time=0.015,
            face_count=1
        )
        
        with patch.object(detector, 'detect_faces', return_value=([mock_face], mock_metrics)):
            metrics = self.benchmark.benchmark_detector(
                detector,
                self.test_images,
                iterations=10,
                warmup_iterations=2
            )
        
        # Check accuracy metrics
        self.assertGreater(metrics.total_detections, 0)
        self.assertGreater(metrics.detection_rate, 0)
        self.assertGreater(metrics.avg_confidence, 0)
    
    def test_compare_detectors(self):
        """Test comparing multiple detectors."""
        detector1 = MockDetector({'instance_id': 1})
        detector2 = MockDetector({'instance_id': 2})
        
        mock_metrics = DetectionMetrics(inference_time=0.01, total_time=0.015)
        
        with patch.object(MockDetector, 'detect_faces', return_value=([], mock_metrics)):
            results = self.benchmark.compare_detectors(
                [detector1, detector2],
                self.test_images,
                iterations=5
            )
        
        # Should have results (both detectors have same type/model, so same key)
        self.assertGreater(len(results), 0)
        
        # All results should be BenchmarkMetrics
        for key, metrics in results.items():
            self.assertIsInstance(metrics, BenchmarkMetrics)
    
    def test_set_baseline(self):
        """Test setting baseline metrics."""
        baseline = BenchmarkMetrics(
            detector_type='cpu',
            model_type='hog',
            avg_inference_time=0.1
        )
        
        self.benchmark.set_baseline(baseline)
        
        self.assertIsNotNone(self.benchmark.baseline_metrics)
        self.assertEqual(self.benchmark.baseline_metrics.detector_type, 'cpu')
    
    def test_check_regression_no_baseline(self):
        """Test regression check without baseline."""
        current = BenchmarkMetrics(
            detector_type='cpu',
            model_type='hog',
            avg_inference_time=0.1
        )
        
        result = self.benchmark.check_regression(current)
        
        self.assertFalse(result['has_baseline'])
        self.assertEqual(len(result['regressions']), 0)
    
    def test_check_regression_with_regression(self):
        """Test regression detection when performance degrades."""
        baseline = BenchmarkMetrics(
            detector_type='cpu',
            model_type='hog',
            avg_inference_time=0.1,
            avg_confidence=0.9
        )
        self.benchmark.set_baseline(baseline)
        
        # Current metrics show degraded performance
        current = BenchmarkMetrics(
            detector_type='cpu',
            model_type='hog',
            avg_inference_time=0.15,  # 50% slower
            avg_confidence=0.9
        )
        
        result = self.benchmark.check_regression(current, threshold=0.10)
        
        self.assertTrue(result['has_baseline'])
        self.assertGreater(len(result['regressions']), 0)
        self.assertFalse(result['passed'])
    
    def test_check_regression_with_improvement(self):
        """Test regression detection when performance improves."""
        baseline = BenchmarkMetrics(
            detector_type='cpu',
            model_type='hog',
            avg_inference_time=0.1
        )
        self.benchmark.set_baseline(baseline)
        
        # Current metrics show improved performance
        current = BenchmarkMetrics(
            detector_type='cpu',
            model_type='hog',
            avg_inference_time=0.08  # 20% faster
        )
        
        result = self.benchmark.check_regression(current)
        
        self.assertTrue(result['has_baseline'])
        self.assertGreater(len(result['improvements']), 0)
        self.assertTrue(result['passed'])
    
    def test_check_regression_passed(self):
        """Test regression check when within threshold."""
        baseline = BenchmarkMetrics(
            detector_type='cpu',
            model_type='hog',
            avg_inference_time=0.1
        )
        self.benchmark.set_baseline(baseline)
        
        # Current metrics within acceptable threshold
        current = BenchmarkMetrics(
            detector_type='cpu',
            model_type='hog',
            avg_inference_time=0.105  # 5% slower, within 10% threshold
        )
        
        result = self.benchmark.check_regression(current, threshold=0.10)
        
        self.assertTrue(result['has_baseline'])
        self.assertTrue(result['passed'])
    
    def test_get_summary_empty(self):
        """Test getting summary with no benchmark results."""
        summary = self.benchmark.get_summary()
        
        self.assertEqual(summary['total_benchmarks'], 0)
    
    def test_get_summary_with_results(self):
        """Test getting summary with benchmark results."""
        detector = MockDetector({})
        
        mock_metrics = DetectionMetrics(inference_time=0.01, total_time=0.015)
        
        with patch.object(detector, 'detect_faces', return_value=([], mock_metrics)):
            self.benchmark.benchmark_detector(detector, self.test_images, iterations=5)
            self.benchmark.benchmark_detector(detector, self.test_images, iterations=5)
        
        summary = self.benchmark.get_summary()
        
        self.assertEqual(summary['total_benchmarks'], 2)
        self.assertIn('detectors_tested', summary)
        self.assertIn('latest_results', summary)
    
    @patch('src.detectors.benchmarking.PSUTIL_AVAILABLE', False)
    def test_benchmark_without_psutil(self):
        """Test benchmarking works without psutil."""
        detector = MockDetector({})
        
        mock_metrics = DetectionMetrics(inference_time=0.01, total_time=0.015)
        
        with patch.object(detector, 'detect_faces', return_value=([], mock_metrics)):
            metrics = self.benchmark.benchmark_detector(
                detector,
                self.test_images,
                iterations=5,
                warmup_iterations=0
            )
        
        # Should still work, just with zero resource metrics
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.avg_memory_usage, 0.0)
        self.assertEqual(metrics.avg_cpu_usage, 0.0)


if __name__ == '__main__':
    unittest.main()
