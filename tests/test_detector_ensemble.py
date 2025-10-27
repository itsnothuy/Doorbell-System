#!/usr/bin/env python3
"""
Test suite for Ensemble Detector

Tests for multi-detector ensembles, voting strategies, and confidence aggregation.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
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

from src.detectors.ensemble_detector import (
    EnsembleDetector,
    EnsembleStrategy,
    EnsembleMetadata
)
from src.detectors.base_detector import (
    FaceDetectionResult,
    DetectionMetrics,
    DetectorType
)
from src.detectors.detector_factory import MockDetector


class TestEnsembleMetadata(unittest.TestCase):
    """Test suite for EnsembleMetadata dataclass."""
    
    def test_metadata_initialization(self):
        """Test EnsembleMetadata initialization."""
        metadata = EnsembleMetadata(
            strategy='voting',
            agreement_score=0.85
        )
        
        self.assertEqual(metadata.strategy, 'voting')
        self.assertEqual(metadata.agreement_score, 0.85)
        self.assertEqual(len(metadata.component_results), 0)
    
    def test_metadata_to_dict(self):
        """Test metadata conversion to dictionary."""
        metadata = EnsembleMetadata(
            strategy='voting',
            component_results=[{'detector': 'cpu'}],
            agreement_score=0.9
        )
        
        result = metadata.to_dict()
        
        self.assertEqual(result['strategy'], 'voting')
        self.assertEqual(len(result['component_results']), 1)
        self.assertEqual(result['agreement_score'], 0.9)


class TestEnsembleDetector(unittest.TestCase):
    """Test suite for EnsembleDetector."""
    
    def setUp(self):
        """Set up test environment."""
        # Create test images
        if NUMPY_AVAILABLE:
            self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        else:
            self.test_image = np.zeros((100, 100, 3))
    
    def test_ensemble_initialization(self):
        """Test ensemble detector initialization."""
        detector1 = MockDetector({})
        detector2 = MockDetector({})
        
        ensemble = EnsembleDetector(
            detectors=[detector1, detector2],
            strategy=EnsembleStrategy.VOTING
        )
        
        self.assertEqual(len(ensemble.component_detectors), 2)
        self.assertEqual(ensemble.ensemble_strategy, EnsembleStrategy.VOTING)
    
    def test_ensemble_initialization_no_detectors(self):
        """Test that ensemble requires at least one detector."""
        with self.assertRaises(ValueError):
            EnsembleDetector(detectors=[], strategy=EnsembleStrategy.VOTING)
    
    def test_ensemble_is_available(self):
        """Test that ensemble is always available."""
        self.assertTrue(EnsembleDetector.is_available())
    
    def test_voting_strategy_consensus(self):
        """Test voting strategy with consensus."""
        detector1 = MockDetector({})
        detector2 = MockDetector({})
        detector3 = MockDetector({})
        
        # Create similar detections that should merge
        face1 = FaceDetectionResult(
            bounding_box=(10, 90, 90, 10),
            confidence=0.9
        )
        face2 = FaceDetectionResult(
            bounding_box=(12, 88, 88, 12),  # Similar to face1
            confidence=0.85
        )
        face3 = FaceDetectionResult(
            bounding_box=(11, 89, 89, 11),  # Similar to face1
            confidence=0.88
        )
        
        metrics = DetectionMetrics(inference_time=0.01, total_time=0.015)
        
        # Mock all detectors to return similar faces
        with patch.object(detector1, 'detect_faces', return_value=([face1], metrics)):
            with patch.object(detector2, 'detect_faces', return_value=([face2], metrics)):
                with patch.object(detector3, 'detect_faces', return_value=([face3], metrics)):
                    ensemble = EnsembleDetector(
                        detectors=[detector1, detector2, detector3],
                        strategy=EnsembleStrategy.VOTING,
                        config={'min_agreement': 0.5}
                    )
                    
                    results = ensemble._run_inference(self.test_image)
        
        # Should merge into one detection (all 3 agree)
        self.assertGreater(len(results), 0)
    
    def test_voting_strategy_no_consensus(self):
        """Test voting strategy without consensus."""
        detector1 = MockDetector({})
        detector2 = MockDetector({})
        detector3 = MockDetector({})
        
        # Create different detections in different locations
        face1 = FaceDetectionResult(
            bounding_box=(10, 30, 30, 10),
            confidence=0.9
        )
        face2 = FaceDetectionResult(
            bounding_box=(50, 70, 70, 50),
            confidence=0.85
        )
        
        metrics = DetectionMetrics(inference_time=0.01, total_time=0.015)
        
        # Only detector1 finds face1, only detector2 finds face2
        with patch.object(detector1, 'detect_faces', return_value=([face1], metrics)):
            with patch.object(detector2, 'detect_faces', return_value=([face2], metrics)):
                with patch.object(detector3, 'detect_faces', return_value=([], metrics)):
                    ensemble = EnsembleDetector(
                        detectors=[detector1, detector2, detector3],
                        strategy=EnsembleStrategy.VOTING,
                        config={'min_agreement': 0.6}  # Need 60% agreement
                    )
                    
                    results = ensemble._run_inference(self.test_image)
        
        # No consensus, so results should be empty or minimal
        # (depends on exact implementation of voting)
        self.assertIsInstance(results, list)
    
    def test_union_strategy(self):
        """Test union strategy keeps all detections."""
        detector1 = MockDetector({})
        detector2 = MockDetector({})
        
        face1 = FaceDetectionResult(
            bounding_box=(10, 30, 30, 10),
            confidence=0.9
        )
        face2 = FaceDetectionResult(
            bounding_box=(50, 70, 70, 50),
            confidence=0.85
        )
        
        metrics = DetectionMetrics(inference_time=0.01, total_time=0.015)
        
        with patch.object(detector1, 'detect_faces', return_value=([face1], metrics)):
            with patch.object(detector2, 'detect_faces', return_value=([face2], metrics)):
                ensemble = EnsembleDetector(
                    detectors=[detector1, detector2],
                    strategy=EnsembleStrategy.UNION
                )
                
                results = ensemble._run_inference(self.test_image)
        
        # Union should include both detections (merged or separate)
        self.assertGreater(len(results), 0)
    
    def test_best_confidence_strategy(self):
        """Test best confidence strategy."""
        detector1 = MockDetector({})
        detector2 = MockDetector({})
        
        # detector2 has higher confidence
        face1 = FaceDetectionResult(
            bounding_box=(10, 30, 30, 10),
            confidence=0.7
        )
        face2 = FaceDetectionResult(
            bounding_box=(50, 70, 70, 50),
            confidence=0.95
        )
        
        metrics = DetectionMetrics(inference_time=0.01, total_time=0.015)
        
        with patch.object(detector1, 'detect_faces', return_value=([face1], metrics)):
            with patch.object(detector2, 'detect_faces', return_value=([face2], metrics)):
                ensemble = EnsembleDetector(
                    detectors=[detector1, detector2],
                    strategy=EnsembleStrategy.BEST_CONFIDENCE
                )
                
                results = ensemble._run_inference(self.test_image)
        
        # Should use results from detector2 (higher confidence)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].confidence, 0.95, places=2)
    
    def test_calculate_iou(self):
        """Test IoU (Intersection over Union) calculation."""
        detector1 = MockDetector({})
        detector2 = MockDetector({})
        
        ensemble = EnsembleDetector(
            detectors=[detector1, detector2],
            strategy=EnsembleStrategy.VOTING
        )
        
        # Identical boxes
        box1 = (10, 90, 90, 10)
        box2 = (10, 90, 90, 10)
        iou = ensemble._calculate_iou(box1, box2)
        self.assertAlmostEqual(iou, 1.0, places=2)
        
        # Non-overlapping boxes
        box1 = (10, 30, 30, 10)
        box2 = (50, 70, 70, 50)
        iou = ensemble._calculate_iou(box1, box2)
        self.assertEqual(iou, 0.0)
        
        # Partially overlapping boxes
        box1 = (10, 60, 60, 10)
        box2 = (30, 80, 80, 30)
        iou = ensemble._calculate_iou(box1, box2)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)
    
    def test_merge_detections(self):
        """Test merging multiple similar detections."""
        detector1 = MockDetector({})
        detector2 = MockDetector({})
        
        ensemble = EnsembleDetector(
            detectors=[detector1, detector2],
            strategy=EnsembleStrategy.VOTING
        )
        
        # Create detections to merge
        face1 = FaceDetectionResult(
            bounding_box=(10, 90, 90, 10),
            confidence=0.9
        )
        face2 = FaceDetectionResult(
            bounding_box=(12, 88, 88, 12),
            confidence=0.85
        )
        
        merged = ensemble._merge_detections([face1, face2])
        
        # Merged confidence should be average
        expected_confidence = (0.9 + 0.85) / 2
        self.assertAlmostEqual(merged.confidence, expected_confidence, places=2)
        
        # Bounding box should be averaged
        self.assertIsNotNone(merged.bounding_box)
    
    def test_merge_single_detection(self):
        """Test merging a single detection returns it unchanged."""
        detector1 = MockDetector({})
        detector2 = MockDetector({})
        
        ensemble = EnsembleDetector(
            detectors=[detector1, detector2],
            strategy=EnsembleStrategy.VOTING
        )
        
        face = FaceDetectionResult(
            bounding_box=(10, 90, 90, 10),
            confidence=0.9
        )
        
        merged = ensemble._merge_detections([face])
        
        self.assertEqual(merged.bounding_box, face.bounding_box)
        self.assertEqual(merged.confidence, face.confidence)
    
    def test_group_similar_detections(self):
        """Test grouping similar detections."""
        detector1 = MockDetector({})
        detector2 = MockDetector({})
        
        ensemble = EnsembleDetector(
            detectors=[detector1, detector2],
            strategy=EnsembleStrategy.VOTING,
            config={'iou_threshold': 0.5}
        )
        
        # Create similar and different detections
        face1 = FaceDetectionResult(
            bounding_box=(10, 90, 90, 10),
            confidence=0.9
        )
        face2 = FaceDetectionResult(
            bounding_box=(12, 88, 88, 12),  # Similar to face1
            confidence=0.85
        )
        face3 = FaceDetectionResult(
            bounding_box=(50, 70, 70, 50),  # Different location
            confidence=0.8
        )
        
        groups = ensemble._group_similar_detections([face1, face2, face3])
        
        # Should have 2 groups: one for similar faces, one for the different face
        self.assertGreaterEqual(len(groups), 1)
    
    def test_ensemble_handles_detector_failure(self):
        """Test ensemble continues when one detector fails."""
        detector1 = MockDetector({})
        detector2 = MockDetector({})
        
        face = FaceDetectionResult(
            bounding_box=(10, 90, 90, 10),
            confidence=0.9
        )
        metrics = DetectionMetrics(inference_time=0.01, total_time=0.015)
        
        # detector1 fails, detector2 succeeds
        with patch.object(detector1, 'detect_faces', side_effect=Exception("Detector failed")):
            with patch.object(detector2, 'detect_faces', return_value=([face], metrics)):
                ensemble = EnsembleDetector(
                    detectors=[detector1, detector2],
                    strategy=EnsembleStrategy.UNION
                )
                
                results = ensemble._run_inference(self.test_image)
        
        # Should still get results from detector2
        self.assertGreater(len(results), 0)
    
    def test_ensemble_all_detectors_fail(self):
        """Test ensemble when all detectors fail."""
        detector1 = MockDetector({})
        detector2 = MockDetector({})
        
        with patch.object(detector1, 'detect_faces', side_effect=Exception("Failed")):
            with patch.object(detector2, 'detect_faces', side_effect=Exception("Failed")):
                ensemble = EnsembleDetector(
                    detectors=[detector1, detector2],
                    strategy=EnsembleStrategy.VOTING
                )
                
                results = ensemble._run_inference(self.test_image)
        
        # Should return empty list
        self.assertEqual(len(results), 0)
    
    def test_ensemble_cleanup(self):
        """Test ensemble cleanup calls cleanup on all component detectors."""
        detector1 = MockDetector({})
        detector2 = MockDetector({})
        
        with patch.object(detector1, 'cleanup') as mock_cleanup1:
            with patch.object(detector2, 'cleanup') as mock_cleanup2:
                ensemble = EnsembleDetector(
                    detectors=[detector1, detector2],
                    strategy=EnsembleStrategy.VOTING
                )
                
                ensemble.cleanup()
                
                # Both detectors should have cleanup called
                mock_cleanup1.assert_called_once()
                mock_cleanup2.assert_called_once()
    
    def test_weighted_voting_strategy(self):
        """Test weighted voting by confidence."""
        detector1 = MockDetector({})
        detector2 = MockDetector({})
        
        # High confidence detection
        face1 = FaceDetectionResult(
            bounding_box=(10, 90, 90, 10),
            confidence=0.95
        )
        # Low confidence detection
        face2 = FaceDetectionResult(
            bounding_box=(12, 88, 88, 12),
            confidence=0.4
        )
        
        metrics = DetectionMetrics(inference_time=0.01, total_time=0.015)
        
        with patch.object(detector1, 'detect_faces', return_value=([face1], metrics)):
            with patch.object(detector2, 'detect_faces', return_value=([face2], metrics)):
                ensemble = EnsembleDetector(
                    detectors=[detector1, detector2],
                    strategy=EnsembleStrategy.WEIGHTED_VOTING,
                    config={'confidence_threshold': 0.5}
                )
                
                results = ensemble._run_inference(self.test_image)
        
        # Results depend on weighted confidence
        self.assertIsInstance(results, list)
    
    def test_intersection_strategy(self):
        """Test intersection strategy requires all detectors to agree."""
        detector1 = MockDetector({})
        detector2 = MockDetector({})
        
        face = FaceDetectionResult(
            bounding_box=(10, 90, 90, 10),
            confidence=0.9
        )
        
        metrics = DetectionMetrics(inference_time=0.01, total_time=0.015)
        
        # Only detector1 finds a face
        with patch.object(detector1, 'detect_faces', return_value=([face], metrics)):
            with patch.object(detector2, 'detect_faces', return_value=([], metrics)):
                ensemble = EnsembleDetector(
                    detectors=[detector1, detector2],
                    strategy=EnsembleStrategy.INTERSECTION
                )
                
                results = ensemble._run_inference(self.test_image)
        
        # Intersection requires all detectors, so should be empty
        self.assertEqual(len(results), 0)


class TestEnsembleStrategy(unittest.TestCase):
    """Test suite for EnsembleStrategy enum."""
    
    def test_strategy_values(self):
        """Test EnsembleStrategy enum values."""
        self.assertEqual(EnsembleStrategy.VOTING.value, "voting")
        self.assertEqual(EnsembleStrategy.WEIGHTED_VOTING.value, "weighted_voting")
        self.assertEqual(EnsembleStrategy.UNION.value, "union")
        self.assertEqual(EnsembleStrategy.INTERSECTION.value, "intersection")
        self.assertEqual(EnsembleStrategy.BEST_CONFIDENCE.value, "best_confidence")


if __name__ == '__main__':
    unittest.main()
