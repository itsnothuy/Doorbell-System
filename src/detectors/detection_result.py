#!/usr/bin/env python3
"""
Detection Result Data Structures

Data structures for face detection results, metrics, and related information.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class DetectionMetrics:
    """Performance metrics for face detection operations."""
    inference_time: float = 0.0
    preprocessing_time: float = 0.0
    postprocessing_time: float = 0.0
    total_time: float = 0.0
    memory_usage: float = 0.0
    confidence: float = 0.0
    face_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'inference_time': self.inference_time,
            'preprocessing_time': self.preprocessing_time,
            'postprocessing_time': self.postprocessing_time,
            'total_time': self.total_time,
            'memory_usage': self.memory_usage,
            'confidence': self.confidence,
            'face_count': self.face_count
        }


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'bounding_box': {
                'top': self.bounding_box[0],
                'right': self.bounding_box[1],
                'bottom': self.bounding_box[2],
                'left': self.bounding_box[3]
            },
            'confidence': self.confidence,
            'landmarks': self.landmarks,
            'quality_score': self.quality_score,
            'width': self.width,
            'height': self.height,
            'area': self.area,
            'center': self.center
        }


@dataclass
class DetectionResult:
    """Complete result from face detection worker."""
    job_id: str
    faces: List[FaceDetectionResult]
    metrics: DetectionMetrics
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'job_id': self.job_id,
            'faces': [face.to_dict() for face in self.faces],
            'metrics': self.metrics.to_dict(),
            'success': self.success,
            'error': self.error,
            'timestamp': self.timestamp
        }
