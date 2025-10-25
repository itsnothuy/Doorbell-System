#!/usr/bin/env python3
"""
Face Recognition Result Data Structures

Data structures for face recognition results, person matches, and related metadata.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


@dataclass
class PersonMatch:
    """Represents a match between a detected face and a known person."""
    person_id: str
    person_name: Optional[str] = None
    confidence: float = 0.0
    similarity_score: float = 0.0
    match_count: int = 1
    is_blacklisted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'person_id': self.person_id,
            'person_name': self.person_name,
            'confidence': self.confidence,
            'similarity_score': self.similarity_score,
            'match_count': self.match_count,
            'is_blacklisted': self.is_blacklisted,
            'metadata': self.metadata
        }


@dataclass
class RecognitionMetadata:
    """Metadata for face recognition operation."""
    tolerance_used: float = 0.6
    cache_hit: bool = False
    processing_method: str = "database"
    encoding_time: float = 0.0
    matching_time: float = 0.0
    total_time: float = 0.0
    database_query_count: int = 0
    faces_compared: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'tolerance_used': self.tolerance_used,
            'cache_hit': self.cache_hit,
            'processing_method': self.processing_method,
            'encoding_time': self.encoding_time,
            'matching_time': self.matching_time,
            'total_time': self.total_time,
            'database_query_count': self.database_query_count,
            'faces_compared': self.faces_compared
        }


@dataclass
class FaceRecognitionResult:
    """Complete result from face recognition operation."""
    face_detection: Any  # FaceDetectionResult from detection worker
    face_encoding: Optional[Any] = None  # np.ndarray when numpy available
    is_known: bool = False
    is_blacklisted: bool = False
    person_matches: List[PersonMatch] = field(default_factory=list)
    confidence: float = 0.0
    recognition_timestamp: float = field(default_factory=time.time)
    metadata: Optional[RecognitionMetadata] = None
    
    @property
    def best_match(self) -> Optional[PersonMatch]:
        """Get the best person match if available."""
        if self.person_matches:
            return self.person_matches[0]
        return None
    
    @property
    def identity(self) -> Optional[str]:
        """Get the identity of the recognized person."""
        if self.best_match:
            return self.best_match.person_name or self.best_match.person_id
        return None
    
    @property
    def is_recognized(self) -> bool:
        """Check if face was successfully recognized."""
        return self.is_known or self.is_blacklisted
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'face_detection': self.face_detection.to_dict() if hasattr(self.face_detection, 'to_dict') else None,
            'is_known': self.is_known,
            'is_blacklisted': self.is_blacklisted,
            'person_matches': [match.to_dict() for match in self.person_matches],
            'confidence': self.confidence,
            'recognition_timestamp': self.recognition_timestamp,
            'identity': self.identity,
            'is_recognized': self.is_recognized,
            'metadata': self.metadata.to_dict() if self.metadata else None
        }
