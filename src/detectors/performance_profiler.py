#!/usr/bin/env python3
"""
Performance Profiler - Detector Benchmarking and Analysis

Comprehensive performance profiling for face detection implementations.
Measures inference time, throughput, latency, and resource usage.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import statistics

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a detector benchmark run."""
    detector_type: str
    model_type: str
    total_runs: int
    total_images: int
    
    # Timing metrics (milliseconds)
    avg_inference_time_ms: float
    min_inference_time_ms: float
    max_inference_time_ms: float
    std_inference_time_ms: float
    p50_inference_time_ms: float
    p95_inference_time_ms: float
    p99_inference_time_ms: float
    
    # Throughput metrics
    fps: float
    images_per_second: float
    
    # Detection metrics
    avg_faces_detected: float
    total_faces_detected: int
    
    # Resource metrics
    avg_memory_mb: float
    peak_memory_mb: float
    
    # Additional metadata
    image_sizes: List[tuple] = field(default_factory=list)
    batch_size: int = 1
    warmup_runs: int = 0


class PerformanceProfiler:
    """
    Performance profiler for face detectors.
    
    Features:
    - Comprehensive timing measurements
    - Statistical analysis (mean, median, percentiles)
    - Resource usage tracking
    - Batch performance testing
    - Comparative benchmarking
    """
    
    def __init__(self):
        """Initialize performance profiler."""
        self.results: Dict[str, BenchmarkResult] = {}
        logger.info("Performance profiler initialized")
    
    def benchmark_detector(
        self,
        detector,
        test_images: List[np.ndarray],
        iterations: int = 10,
        warmup_iterations: int = 3,
        batch_size: int = 1
    ) -> BenchmarkResult:
        """
        Benchmark a detector with test images.
        
        Args:
            detector: Detector instance to benchmark
            test_images: List of test images
            iterations: Number of iterations per image
            warmup_iterations: Number of warmup runs (not measured)
            batch_size: Batch size for inference
            
        Returns:
            BenchmarkResult with comprehensive metrics
        """
        if not test_images:
            raise ValueError("No test images provided")
        
        logger.info(f"Benchmarking {detector.detector_type.value} detector")
        logger.info(f"Images: {len(test_images)}, Iterations: {iterations}, Warmup: {warmup_iterations}")
        
        # Collect image sizes
        image_sizes = [img.shape for img in test_images]
        
        # Warmup phase
        logger.debug("Running warmup iterations...")
        for _ in range(warmup_iterations):
            for image in test_images:
                try:
                    _ = detector.detect_faces(image)
                except Exception as e:
                    logger.warning(f"Warmup iteration failed: {e}")
        
        # Benchmark phase
        inference_times = []
        memory_readings = []
        total_faces = 0
        face_counts = []
        
        logger.debug("Running benchmark iterations...")
        for iteration in range(iterations):
            for image in test_images:
                try:
                    # Measure inference time
                    start_time = time.perf_counter()
                    detections, metrics = detector.detect_faces(image)
                    end_time = time.perf_counter()
                    
                    inference_time_ms = (end_time - start_time) * 1000
                    inference_times.append(inference_time_ms)
                    
                    # Track detections
                    face_count = len(detections)
                    face_counts.append(face_count)
                    total_faces += face_count
                    
                    # Track memory
                    memory_mb = metrics.memory_usage if metrics else 0.0
                    memory_readings.append(memory_mb)
                    
                except Exception as e:
                    logger.error(f"Benchmark iteration failed: {e}")
                    inference_times.append(0.0)
                    face_counts.append(0)
        
        # Calculate statistics
        if not inference_times:
            raise RuntimeError("Benchmark failed - no successful runs")
        
        avg_time = statistics.mean(inference_times)
        min_time = min(inference_times)
        max_time = max(inference_times)
        std_time = statistics.stdev(inference_times) if len(inference_times) > 1 else 0.0
        
        # Calculate percentiles
        sorted_times = sorted(inference_times)
        p50 = sorted_times[len(sorted_times) // 2]
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]
        
        # Calculate throughput
        fps = 1000.0 / avg_time if avg_time > 0 else 0
        total_runs = len(test_images) * iterations
        
        # Calculate memory stats
        avg_memory = statistics.mean(memory_readings) if memory_readings else 0.0
        peak_memory = max(memory_readings) if memory_readings else 0.0
        
        # Create benchmark result
        result = BenchmarkResult(
            detector_type=detector.detector_type.value,
            model_type=detector.model_type.value,
            total_runs=total_runs,
            total_images=len(test_images),
            avg_inference_time_ms=avg_time,
            min_inference_time_ms=min_time,
            max_inference_time_ms=max_time,
            std_inference_time_ms=std_time,
            p50_inference_time_ms=p50,
            p95_inference_time_ms=p95,
            p99_inference_time_ms=p99,
            fps=fps,
            images_per_second=fps,
            avg_faces_detected=statistics.mean(face_counts) if face_counts else 0.0,
            total_faces_detected=total_faces,
            avg_memory_mb=avg_memory,
            peak_memory_mb=peak_memory,
            image_sizes=image_sizes,
            batch_size=batch_size,
            warmup_runs=warmup_iterations
        )
        
        # Store result
        self.results[detector.detector_type.value] = result
        
        logger.info(f"Benchmark completed: {avg_time:.2f}ms avg, {fps:.1f} FPS")
        
        return result
    
    def compare_detectors(
        self,
        detectors: List,
        test_images: List[np.ndarray],
        iterations: int = 10
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare multiple detectors with the same test images.
        
        Args:
            detectors: List of detector instances
            test_images: Test images (same for all detectors)
            iterations: Number of iterations per detector
            
        Returns:
            Dictionary mapping detector type to benchmark results
        """
        logger.info(f"Comparing {len(detectors)} detectors")
        
        results = {}
        for detector in detectors:
            try:
                result = self.benchmark_detector(detector, test_images, iterations)
                results[detector.detector_type.value] = result
            except Exception as e:
                logger.error(f"Failed to benchmark {detector.detector_type.value}: {e}")
        
        return results
    
    def get_comparison_report(self) -> str:
        """
        Generate a comparison report of all benchmarked detectors.
        
        Returns:
            Multi-line string with comparison table
        """
        if not self.results:
            return "No benchmark results available"
        
        lines = [
            "Detector Performance Comparison",
            "=" * 80,
            ""
        ]
        
        # Header
        lines.append(f"{'Detector':<20} {'Model':<15} {'Avg (ms)':<12} {'FPS':<10} {'P95 (ms)':<12} {'Memory (MB)':<12}")
        lines.append("-" * 80)
        
        # Sort by average inference time (fastest first)
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].avg_inference_time_ms
        )
        
        for detector_type, result in sorted_results:
            lines.append(
                f"{result.detector_type:<20} "
                f"{result.model_type:<15} "
                f"{result.avg_inference_time_ms:>10.2f}  "
                f"{result.fps:>8.1f}  "
                f"{result.p95_inference_time_ms:>10.2f}  "
                f"{result.avg_memory_mb:>10.1f}"
            )
        
        lines.append("")
        lines.append("Speedup Analysis (vs CPU):")
        lines.append("-" * 40)
        
        # Calculate speedup relative to CPU
        cpu_time = None
        for detector_type, result in self.results.items():
            if 'cpu' in detector_type.lower():
                cpu_time = result.avg_inference_time_ms
                break
        
        if cpu_time:
            for detector_type, result in sorted_results:
                if result.avg_inference_time_ms > 0:
                    speedup = cpu_time / result.avg_inference_time_ms
                    lines.append(f"  {result.detector_type:<20} {speedup:>6.2f}x")
        
        return "\n".join(lines)
    
    def get_detailed_report(self, detector_type: str) -> str:
        """
        Get detailed report for a specific detector.
        
        Args:
            detector_type: Type of detector to report on
            
        Returns:
            Multi-line string with detailed statistics
        """
        if detector_type not in self.results:
            return f"No results for detector: {detector_type}"
        
        result = self.results[detector_type]
        
        lines = [
            f"Detailed Report: {result.detector_type}",
            "=" * 60,
            "",
            "Configuration:",
            f"  Model: {result.model_type}",
            f"  Test Images: {result.total_images}",
            f"  Total Runs: {result.total_runs}",
            f"  Batch Size: {result.batch_size}",
            f"  Warmup Runs: {result.warmup_runs}",
            "",
            "Timing Statistics (milliseconds):",
            f"  Average: {result.avg_inference_time_ms:.3f}",
            f"  Minimum: {result.min_inference_time_ms:.3f}",
            f"  Maximum: {result.max_inference_time_ms:.3f}",
            f"  Std Dev: {result.std_inference_time_ms:.3f}",
            f"  Median (P50): {result.p50_inference_time_ms:.3f}",
            f"  P95: {result.p95_inference_time_ms:.3f}",
            f"  P99: {result.p99_inference_time_ms:.3f}",
            "",
            "Throughput:",
            f"  FPS: {result.fps:.2f}",
            f"  Images/sec: {result.images_per_second:.2f}",
            "",
            "Detection Metrics:",
            f"  Total Faces: {result.total_faces_detected}",
            f"  Avg per Image: {result.avg_faces_detected:.2f}",
            "",
            "Resource Usage:",
            f"  Avg Memory: {result.avg_memory_mb:.1f} MB",
            f"  Peak Memory: {result.peak_memory_mb:.1f} MB",
            ""
        ]
        
        if result.image_sizes:
            lines.append("Test Image Sizes:")
            unique_sizes = list(set(tuple(size) for size in result.image_sizes))
            for size in unique_sizes:
                count = sum(1 for s in result.image_sizes if tuple(s) == size)
                lines.append(f"  {size}: {count} images")
        
        return "\n".join(lines)
    
    def export_results(self) -> Dict[str, Any]:
        """
        Export all benchmark results as a dictionary.
        
        Returns:
            Dictionary containing all results
        """
        return {
            detector_type: {
                'detector_type': result.detector_type,
                'model_type': result.model_type,
                'total_runs': result.total_runs,
                'total_images': result.total_images,
                'timing': {
                    'avg_ms': result.avg_inference_time_ms,
                    'min_ms': result.min_inference_time_ms,
                    'max_ms': result.max_inference_time_ms,
                    'std_ms': result.std_inference_time_ms,
                    'p50_ms': result.p50_inference_time_ms,
                    'p95_ms': result.p95_inference_time_ms,
                    'p99_ms': result.p99_inference_time_ms
                },
                'throughput': {
                    'fps': result.fps,
                    'images_per_second': result.images_per_second
                },
                'detection': {
                    'avg_faces': result.avg_faces_detected,
                    'total_faces': result.total_faces_detected
                },
                'resources': {
                    'avg_memory_mb': result.avg_memory_mb,
                    'peak_memory_mb': result.peak_memory_mb
                }
            }
            for detector_type, result in self.results.items()
        }
    
    def clear_results(self) -> None:
        """Clear all stored benchmark results."""
        self.results.clear()
        logger.info("Benchmark results cleared")
