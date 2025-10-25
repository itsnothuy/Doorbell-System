#!/usr/bin/env python3
"""
Base Detector Interface - Strategy Pattern for Face Detection

This module defines the abstract interface for face detection implementations,
following Frigate's detector plugin architecture.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np


class DetectorType(Enum):
    """Types of face detection implementations."""
    CPU = "cpu"
    GPU = "gpu"
    EDGETPU = "edgetpu"
    MOCK = "mock"


class ModelType(Enum):
    """Face detection model types."""
    HOG = "hog"
    CNN = "cnn"
    RETINAFACE = "retinaface"
    MTCNN = "mtcnn"
    YOLOV5 = "yolov5"


@dataclass
class DetectionMetrics:
    """Performance metrics for face detection."""
    inference_time: float = 0.0
    preprocessing_time: float = 0.0
    postprocessing_time: float = 0.0
    total_time: float = 0.0
    memory_usage: float = 0.0
    confidence: float = 0.0
    face_count: int = 0


@dataclass
class FaceDetectionResult:
    """Result from face detection operation."""
    bounding_box: Tuple[int, int, int, int]  # (top, right, bottom, left)
    confidence: float
    landmarks: Optional[Dict[str, Any]] = None
    quality_score: float = 0.0
    
    @property
    def width(self) -> int:
        """Get bounding box width."""
        top, right, bottom, left = self.bounding_box
        return right - left
    
    @property
    def height(self) -> int:
        """Get bounding box height."""
        top, right, bottom, left = self.bounding_box
        return bottom - top
    
    @property
    def area(self) -> int:
        """Get bounding box area."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get bounding box center point."""
        top, right, bottom, left = self.bounding_box
        return ((left + right) // 2, (top + bottom) // 2)


class BaseDetector(ABC):
    """
    Abstract base class for face detection implementations.
    
    This follows the strategy pattern used in Frigate for detector plugins,
    allowing different detection backends to be used interchangeably.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the detector with configuration.
        
        Args:
            config: Detector-specific configuration parameters
        """
        self.config = config
        self.detector_type = self._get_detector_type()
        self.model_type = ModelType(config.get('model_type', 'hog'))
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Performance tracking
        self.total_detections = 0
        self.total_inference_time = 0.0
        self.last_metrics: Optional[DetectionMetrics] = None
        
        # Detection parameters
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.min_face_size = config.get('min_face_size', (30, 30))
        self.max_face_size = config.get('max_face_size', (1000, 1000))
        
        # Initialize the detector model
        self._initialize_model()
        
        self.logger.info(f"Initialized {self.detector_type.value} detector with {self.model_type.value} model")
    
    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """
        Check if this detector is available on the current system.
        
        Returns:
            bool: True if detector can be used, False otherwise
        """
        pass
    
    @abstractmethod
    def _get_detector_type(self) -> DetectorType:
        """Return the detector type for this implementation."""
        pass
    
    @abstractmethod
    def _initialize_model(self) -> None:
        """Initialize the detection model (implementation-specific)."""
        pass
    
    @abstractmethod
    def _run_inference(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """Run face detection inference on an image."""
        pass
    
    def detect_faces(self, image: np.ndarray) -> Tuple[List[FaceDetectionResult], DetectionMetrics]:
        """
        Detect faces in an image and return results with metrics.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Tuple of (detection results, performance metrics)
        """
        start_time = time.time()
        
        try:
            # Validate input
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")
            
            # Preprocessing
            preprocess_start = time.time()
            processed_image = self._preprocess_image(image)
            preprocess_time = time.time() - preprocess_start
            
            # Run inference
            inference_start = time.time()
            detections = self._run_inference(processed_image)
            inference_time = time.time() - inference_start
            
            # Postprocessing
            postprocess_start = time.time()
            filtered_detections = self._postprocess_detections(detections, image.shape)
            postprocess_time = time.time() - postprocess_start
            
            # Calculate metrics
            total_time = time.time() - start_time
            metrics = DetectionMetrics(
                inference_time=inference_time,
                preprocessing_time=preprocess_time,
                postprocessing_time=postprocess_time,
                total_time=total_time,
                memory_usage=self._get_memory_usage(),
                confidence=self._calculate_average_confidence(filtered_detections),
                face_count=len(filtered_detections)
            )
            
            # Update tracking
            self.total_detections += 1
            self.total_inference_time += inference_time
            self.last_metrics = metrics
            
            self.logger.debug(f"Detected {len(filtered_detections)} faces in {total_time:.3f}s")
            
            return filtered_detections, metrics
            
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            # Return empty results with error metrics
            error_metrics = DetectionMetrics(
                total_time=time.time() - start_time,
                face_count=0
            )
            return [], error_metrics
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for detection (can be overridden by subclasses).
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Default preprocessing: ensure RGB format and proper size
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB if needed (OpenCV default is BGR)
            if self._is_bgr_image(image):
                image = image[:, :, ::-1]
        
        return image
    
    def _postprocess_detections(self, detections: List[FaceDetectionResult], 
                              image_shape: Tuple[int, ...]) -> List[FaceDetectionResult]:
        """
        Filter and validate detection results.
        
        Args:
            detections: Raw detection results
            image_shape: Original image shape for validation
            
        Returns:
            Filtered detection results
        """
        filtered = []
        height, width = image_shape[:2]
        
        for detection in detections:
            # Filter by confidence threshold
            if detection.confidence < self.confidence_threshold:
                continue
            
            # Validate bounding box
            top, right, bottom, left = detection.bounding_box
            if (left < 0 or top < 0 or right > width or bottom > height or
                left >= right or top >= bottom):
                continue
            
            # Filter by face size
            face_width = right - left
            face_height = bottom - top
            min_w, min_h = self.min_face_size
            max_w, max_h = self.max_face_size
            
            if (face_width < min_w or face_height < min_h or
                face_width > max_w or face_height > max_h):
                continue
            
            # Calculate quality score
            detection.quality_score = self._calculate_quality_score(detection, image_shape)
            
            filtered.append(detection)
        
        # Sort by confidence (highest first)
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        
        return filtered
    
    def _calculate_quality_score(self, detection: FaceDetectionResult, 
                               image_shape: Tuple[int, ...]) -> float:
        """
        Calculate quality score for a detection (0.0 to 1.0).
        
        Args:
            detection: Face detection result
            image_shape: Image dimensions
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        # Base score from confidence
        score = detection.confidence
        
        # Adjust for face size (prefer medium-sized faces)
        face_area = detection.area
        image_area = image_shape[0] * image_shape[1]
        face_ratio = face_area / image_area
        
        # Optimal face ratio is around 5-20% of image
        if 0.05 <= face_ratio <= 0.20:
            score *= 1.0  # No penalty
        elif face_ratio < 0.05:
            score *= (face_ratio / 0.05)  # Penalty for small faces
        else:
            score *= max(0.5, 1.0 - ((face_ratio - 0.20) / 0.30))  # Penalty for large faces
        
        # Adjust for position (prefer centered faces)
        center_x, center_y = detection.center
        image_center_x, image_center_y = image_shape[1] // 2, image_shape[0] // 2
        
        distance_from_center = np.sqrt(
            ((center_x - image_center_x) / image_shape[1]) ** 2 +
            ((center_y - image_center_y) / image_shape[0]) ** 2
        )
        
        # Penalty for faces far from center
        if distance_from_center > 0.3:
            score *= max(0.7, 1.0 - distance_from_center)
        
        return min(1.0, max(0.0, score))
    
    def _calculate_average_confidence(self, detections: List[FaceDetectionResult]) -> float:
        """Calculate average confidence of all detections."""
        if not detections:
            return 0.0
        return sum(d.confidence for d in detections) / len(detections)
    
    def _is_bgr_image(self, image: np.ndarray) -> bool:
        """Heuristic to determine if image is in BGR format (OpenCV default)."""
        # This is a simple heuristic - in practice, you might need more sophisticated detection
        # For now, assume images from OpenCV are BGR
        return True
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB (can be overridden by subclasses)."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this detector."""
        avg_inference_time = (self.total_inference_time / max(1, self.total_detections))
        
        return {
            'detector_type': self.detector_type.value,
            'model_type': self.model_type.value,
            'total_detections': self.total_detections,
            'average_inference_time': avg_inference_time,
            'total_inference_time': self.total_inference_time,
            'last_metrics': self.last_metrics.__dict__ if self.last_metrics else None,
            'memory_usage_mb': self._get_memory_usage(),
            'config': self.config
        }
    
    def benchmark(self, test_images: List[np.ndarray], iterations: int = 10) -> Dict[str, Any]:
        """
        Benchmark the detector with test images.
        
        Args:
            test_images: List of test images
            iterations: Number of iterations per image
            
        Returns:
            Benchmark results
        """
        if not test_images:
            raise ValueError("No test images provided")
        
        self.logger.info(f"Starting benchmark with {len(test_images)} images, {iterations} iterations")
        
        all_times = []
        all_detections = []
        
        for i, image in enumerate(test_images):
            for iteration in range(iterations):
                detections, metrics = self.detect_faces(image)
                all_times.append(metrics.total_time)
                all_detections.append(len(detections))
        
        # Calculate statistics
        avg_time = np.mean(all_times)
        min_time = np.min(all_times)
        max_time = np.max(all_times)
        std_time = np.std(all_times)
        
        avg_detections = np.mean(all_detections)
        
        results = {
            'detector_type': self.detector_type.value,
            'model_type': self.model_type.value,
            'test_images': len(test_images),
            'iterations_per_image': iterations,
            'total_runs': len(all_times),
            'timing': {
                'average_ms': avg_time * 1000,
                'min_ms': min_time * 1000,
                'max_ms': max_time * 1000,
                'std_ms': std_time * 1000,
                'fps': 1.0 / avg_time if avg_time > 0 else 0
            },
            'detections': {
                'average_per_image': avg_detections,
                'total_detections': sum(all_detections)
            },
            'memory_usage_mb': self._get_memory_usage()
        }
        
        self.logger.info(f"Benchmark completed: {avg_time*1000:.2f}ms avg, {1.0/avg_time:.1f} FPS")
        
        return results
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        try:
            # Create a small test image
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Try to run detection
            start_time = time.time()
            detections, metrics = self.detect_faces(test_image)
            response_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'detector_type': self.detector_type.value,
                'model_type': self.model_type.value,
                'response_time_ms': response_time * 1000,
                'memory_usage_mb': self._get_memory_usage(),
                'total_detections': self.total_detections,
                'average_inference_time_ms': (self.total_inference_time / max(1, self.total_detections)) * 1000
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'detector_type': self.detector_type.value,
                'error': str(e),
                'memory_usage_mb': self._get_memory_usage()
            }
    
    def cleanup(self) -> None:
        """Cleanup detector resources (override in subclasses if needed)."""
        self.logger.info(f"Cleaning up {self.detector_type.value} detector")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass