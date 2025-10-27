#!/usr/bin/env python3
"""
Advanced Detector Benchmarking System

Comprehensive performance analysis and optimization framework for face detectors.
Provides detailed metrics, comparative analysis, and performance regression detection.
"""

import time
import logging
import statistics
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

import numpy as np

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from src.detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Comprehensive benchmarking metrics for a detector."""
    
    detector_type: str
    model_type: str
    
    # Performance metrics
    avg_inference_time: float = 0.0
    min_inference_time: float = float('inf')
    max_inference_time: float = 0.0
    std_inference_time: float = 0.0
    percentile_95_time: float = 0.0
    percentile_99_time: float = 0.0
    
    # Resource metrics
    avg_memory_usage: float = 0.0
    peak_memory_usage: float = 0.0
    avg_cpu_usage: float = 0.0
    peak_cpu_usage: float = 0.0
    
    # Accuracy metrics
    avg_confidence: float = 0.0
    detection_rate: float = 0.0
    total_detections: int = 0
    
    # Throughput metrics
    images_per_second: float = 0.0
    total_images_processed: int = 0
    
    # Additional metrics
    benchmark_duration: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'detector_type': self.detector_type,
            'model_type': self.model_type,
            'performance': {
                'avg_inference_time_ms': self.avg_inference_time * 1000,
                'min_inference_time_ms': self.min_inference_time * 1000,
                'max_inference_time_ms': self.max_inference_time * 1000,
                'std_inference_time_ms': self.std_inference_time * 1000,
                'percentile_95_time_ms': self.percentile_95_time * 1000,
                'percentile_99_time_ms': self.percentile_99_time * 1000,
            },
            'resources': {
                'avg_memory_usage_mb': self.avg_memory_usage,
                'peak_memory_usage_mb': self.peak_memory_usage,
                'avg_cpu_usage_percent': self.avg_cpu_usage,
                'peak_cpu_usage_percent': self.peak_cpu_usage,
            },
            'accuracy': {
                'avg_confidence': self.avg_confidence,
                'detection_rate': self.detection_rate,
                'total_detections': self.total_detections,
            },
            'throughput': {
                'images_per_second': self.images_per_second,
                'total_images_processed': self.total_images_processed,
            },
            'benchmark_duration': self.benchmark_duration,
            'timestamp': self.timestamp,
        }


class DetectorBenchmark:
    """
    Advanced benchmarking system for face detectors.
    
    Provides comprehensive performance analysis including timing,
    resource usage, and accuracy metrics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize benchmarking system.
        
        Args:
            config: Optional configuration for benchmark parameters
        """
        self.config = config or {}
        self.results_history: List[BenchmarkMetrics] = []
        self.baseline_metrics: Optional[BenchmarkMetrics] = None
        
        logger.info("Initialized detector benchmarking system")
    
    def benchmark_detector(
        self,
        detector: BaseDetector,
        test_images: List[np.ndarray],
        iterations: int = 100,
        warmup_iterations: int = 5
    ) -> BenchmarkMetrics:
        """
        Run comprehensive benchmark on a detector.
        
        Args:
            detector: Detector instance to benchmark
            test_images: List of test images
            iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations before measurement
            
        Returns:
            BenchmarkMetrics with detailed performance data
        """
        if not test_images:
            raise ValueError("No test images provided for benchmarking")
        
        logger.info(
            f"Starting benchmark for {detector.detector_type.value} detector: "
            f"{len(test_images)} images, {iterations} iterations"
        )
        
        # Initialize metrics
        metrics = BenchmarkMetrics(
            detector_type=detector.detector_type.value,
            model_type=detector.model_type.value
        )
        
        # Warmup phase
        logger.debug(f"Warmup phase: {warmup_iterations} iterations")
        for i in range(warmup_iterations):
            image = test_images[i % len(test_images)]
            detector.detect_faces(image)
        
        # Measurement phase
        inference_times = []
        memory_usage = []
        cpu_usage = []
        confidences = []
        detection_counts = []
        
        start_time = time.time()
        
        for i in range(iterations):
            # Select test image
            image = test_images[i % len(test_images)]
            
            # Measure resource usage before
            if PSUTIL_AVAILABLE:
                memory_before = psutil.virtual_memory().percent
                cpu_before = psutil.cpu_percent(interval=0.01)
            else:
                memory_before = 0.0
                cpu_before = 0.0
            
            # Run inference with timing
            inference_start = time.perf_counter()
            detections, detection_metrics = detector.detect_faces(image)
            inference_time = time.perf_counter() - inference_start
            
            # Measure resource usage after
            if PSUTIL_AVAILABLE:
                memory_after = psutil.virtual_memory().percent
                cpu_after = psutil.cpu_percent(interval=0.01)
            else:
                memory_after = 0.0
                cpu_after = 0.0
            
            # Collect metrics
            inference_times.append(inference_time)
            memory_usage.append(max(0, memory_after - memory_before))
            cpu_usage.append(max(0, cpu_after - cpu_before))
            detection_counts.append(len(detections))
            
            if detections:
                avg_conf = sum(face.confidence for face in detections) / len(detections)
                confidences.append(avg_conf)
        
        # Calculate aggregate metrics
        if inference_times:
            metrics.avg_inference_time = statistics.mean(inference_times)
            metrics.min_inference_time = min(inference_times)
            metrics.max_inference_time = max(inference_times)
            metrics.std_inference_time = statistics.stdev(inference_times) if len(inference_times) > 1 else 0.0
            
            # Calculate percentiles
            sorted_times = sorted(inference_times)
            p95_idx = int(len(sorted_times) * 0.95)
            p99_idx = int(len(sorted_times) * 0.99)
            metrics.percentile_95_time = sorted_times[p95_idx]
            metrics.percentile_99_time = sorted_times[p99_idx]
        
        if memory_usage:
            metrics.avg_memory_usage = statistics.mean(memory_usage)
            metrics.peak_memory_usage = max(memory_usage)
        
        if cpu_usage:
            metrics.avg_cpu_usage = statistics.mean(cpu_usage)
            metrics.peak_cpu_usage = max(cpu_usage)
        
        if confidences:
            metrics.avg_confidence = statistics.mean(confidences)
        
        if detection_counts:
            images_with_detections = sum(1 for count in detection_counts if count > 0)
            metrics.detection_rate = images_with_detections / len(detection_counts)
            metrics.total_detections = sum(detection_counts)
        
        metrics.benchmark_duration = time.time() - start_time
        metrics.images_per_second = iterations / metrics.benchmark_duration
        metrics.total_images_processed = iterations
        
        # Store in history
        self.results_history.append(metrics)
        
        logger.info(
            f"Benchmark completed: {metrics.avg_inference_time*1000:.2f}ms avg, "
            f"{metrics.images_per_second:.1f} FPS, "
            f"{metrics.detection_rate*100:.1f}% detection rate"
        )
        
        return metrics
    
    def compare_detectors(
        self,
        detectors: List[BaseDetector],
        test_images: List[np.ndarray],
        iterations: int = 100
    ) -> Dict[str, BenchmarkMetrics]:
        """
        Compare performance across multiple detectors.
        
        Args:
            detectors: List of detector instances to compare
            test_images: Common test images for fair comparison
            iterations: Number of iterations per detector
            
        Returns:
            Dictionary mapping detector type to benchmark metrics
        """
        logger.info(f"Comparing {len(detectors)} detectors")
        
        results = {}
        for detector in detectors:
            metrics = self.benchmark_detector(detector, test_images, iterations)
            key = f"{detector.detector_type.value}_{detector.model_type.value}"
            results[key] = metrics
        
        return results
    
    def set_baseline(self, metrics: BenchmarkMetrics) -> None:
        """
        Set baseline metrics for regression detection.
        
        Args:
            metrics: Baseline benchmark metrics
        """
        self.baseline_metrics = metrics
        logger.info(f"Set baseline metrics for {metrics.detector_type}")
    
    def check_regression(
        self,
        current_metrics: BenchmarkMetrics,
        threshold: float = 0.10
    ) -> Dict[str, Any]:
        """
        Check for performance regression against baseline.
        
        Args:
            current_metrics: Current benchmark metrics
            threshold: Acceptable performance degradation (0.10 = 10%)
            
        Returns:
            Dictionary with regression check results
        """
        if not self.baseline_metrics:
            logger.warning("No baseline metrics set for regression check")
            return {
                'has_baseline': False,
                'regressions': [],
                'improvements': []
            }
        
        regressions = []
        improvements = []
        
        # Check inference time regression
        time_change = (
            (current_metrics.avg_inference_time - self.baseline_metrics.avg_inference_time) /
            self.baseline_metrics.avg_inference_time
        )
        
        if time_change > threshold:
            regressions.append({
                'metric': 'avg_inference_time',
                'baseline': self.baseline_metrics.avg_inference_time,
                'current': current_metrics.avg_inference_time,
                'change_percent': time_change * 100
            })
        elif time_change < -0.05:  # More than 5% improvement
            improvements.append({
                'metric': 'avg_inference_time',
                'baseline': self.baseline_metrics.avg_inference_time,
                'current': current_metrics.avg_inference_time,
                'change_percent': time_change * 100
            })
        
        # Check confidence regression
        if self.baseline_metrics.avg_confidence > 0:
            conf_change = (
                (current_metrics.avg_confidence - self.baseline_metrics.avg_confidence) /
                self.baseline_metrics.avg_confidence
            )
            
            if conf_change < -threshold:
                regressions.append({
                    'metric': 'avg_confidence',
                    'baseline': self.baseline_metrics.avg_confidence,
                    'current': current_metrics.avg_confidence,
                    'change_percent': conf_change * 100
                })
        
        return {
            'has_baseline': True,
            'regressions': regressions,
            'improvements': improvements,
            'passed': len(regressions) == 0
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all benchmark results.
        
        Returns:
            Dictionary with benchmark summary
        """
        if not self.results_history:
            return {'total_benchmarks': 0}
        
        return {
            'total_benchmarks': len(self.results_history),
            'detectors_tested': list(set(
                m.detector_type for m in self.results_history
            )),
            'latest_results': [m.to_dict() for m in self.results_history[-5:]],
            'baseline_set': self.baseline_metrics is not None
        }
