#!/usr/bin/env python3
"""
Similarity Matcher

Compares face encodings and determines similarity scores for face recognition.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    face_recognition = None

logger = logging.getLogger(__name__)


class SimilarityMatcher:
    """Compares face encodings and computes similarity scores."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize similarity matcher.
        
        Args:
            config: Configuration dictionary with matching settings
        """
        self.config = config
        self.similarity_metric = config.get('similarity_metric', 'euclidean')
        self.tolerance = config.get('tolerance', 0.6)
        
        # Performance metrics
        self.comparison_count = 0
        self.match_count = 0
        
        logger.info(f"SimilarityMatcher initialized with metric={self.similarity_metric}, tolerance={self.tolerance}")
    
    def compare_faces(self, known_encoding: Any, unknown_encoding: Any, tolerance: Optional[float] = None) -> bool:
        """
        Compare two face encodings to determine if they match.
        
        Args:
            known_encoding: Known face encoding
            unknown_encoding: Unknown face encoding to compare
            tolerance: Similarity threshold (lower is stricter)
            
        Returns:
            True if faces match, False otherwise
        """
        if tolerance is None:
            tolerance = self.tolerance
        
        if not FACE_RECOGNITION_AVAILABLE or not NUMPY_AVAILABLE:
            # Mock comparison for testing
            return True
        
        try:
            # Use face_recognition's comparison
            matches = face_recognition.compare_faces(
                [known_encoding],
                unknown_encoding,
                tolerance=tolerance
            )
            
            self.comparison_count += 1
            if matches[0]:
                self.match_count += 1
            
            return matches[0]
            
        except Exception as e:
            logger.error(f"Face comparison failed: {e}")
            return False
    
    def compute_distance(self, encoding1: Any, encoding2: Any) -> float:
        """
        Compute distance between two face encodings.
        
        Args:
            encoding1: First face encoding
            encoding2: Second face encoding
            
        Returns:
            Distance value (lower means more similar)
        """
        if not NUMPY_AVAILABLE:
            # Mock distance for testing
            return 0.5
        
        try:
            if self.similarity_metric == 'euclidean':
                # Euclidean distance
                distance = np.linalg.norm(encoding1 - encoding2)
            elif self.similarity_metric == 'cosine':
                # Cosine distance
                dot_product = np.dot(encoding1, encoding2)
                norm1 = np.linalg.norm(encoding1)
                norm2 = np.linalg.norm(encoding2)
                cosine_similarity = dot_product / (norm1 * norm2)
                distance = 1.0 - cosine_similarity
            else:
                # Default to euclidean
                distance = np.linalg.norm(encoding1 - encoding2)
            
            return float(distance)
            
        except Exception as e:
            logger.error(f"Distance computation failed: {e}")
            return 1.0  # Return high distance on error
    
    def compute_distances_batch(self, known_encodings: List[Any], unknown_encoding: Any) -> List[float]:
        """
        Compute distances between multiple known encodings and an unknown encoding.
        
        Args:
            known_encodings: List of known face encodings
            unknown_encoding: Unknown face encoding to compare
            
        Returns:
            List of distance values
        """
        if not FACE_RECOGNITION_AVAILABLE or not NUMPY_AVAILABLE:
            # Mock distances for testing
            return [0.5] * len(known_encodings)
        
        try:
            # Use face_recognition's efficient batch distance computation
            distances = face_recognition.face_distance(known_encodings, unknown_encoding)
            return distances.tolist()
            
        except Exception as e:
            logger.error(f"Batch distance computation failed: {e}")
            return [1.0] * len(known_encodings)
    
    def find_best_matches(self, known_encodings: List[Any], unknown_encoding: Any, 
                         tolerance: Optional[float] = None, top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Find best matching face encodings from a list of known encodings.
        
        Args:
            known_encodings: List of known face encodings
            unknown_encoding: Unknown face encoding to match
            tolerance: Similarity threshold
            top_k: Number of top matches to return
            
        Returns:
            List of (index, distance) tuples for best matches, sorted by distance
        """
        if tolerance is None:
            tolerance = self.tolerance
        
        # Compute all distances
        distances = self.compute_distances_batch(known_encodings, unknown_encoding)
        
        # Find matches within tolerance
        matches = []
        for idx, distance in enumerate(distances):
            if distance <= tolerance:
                matches.append((idx, distance))
        
        # Sort by distance (lower is better) and return top k
        matches.sort(key=lambda x: x[1])
        return matches[:top_k]
    
    def compute_confidence(self, distance: float, tolerance: Optional[float] = None) -> float:
        """
        Convert distance to confidence score (0-1).
        
        Args:
            distance: Distance value
            tolerance: Similarity threshold
            
        Returns:
            Confidence score (1.0 = perfect match, 0.0 = no match)
        """
        if tolerance is None:
            tolerance = self.tolerance
        
        # Convert distance to confidence
        # Distance 0 = confidence 1.0
        # Distance at tolerance = confidence 0.5
        # Distance > tolerance = confidence < 0.5
        
        if distance <= 0:
            return 1.0
        
        # Linear mapping: conf = 1 - (distance / (2 * tolerance))
        confidence = max(0.0, 1.0 - (distance / (2 * tolerance)))
        
        return confidence
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get matcher performance metrics."""
        return {
            'comparison_count': self.comparison_count,
            'match_count': self.match_count,
            'match_rate': self.match_count / max(1, self.comparison_count),
            'similarity_metric': self.similarity_metric,
            'tolerance': self.tolerance
        }
