#!/usr/bin/env python3
"""
Test suite for Enhanced Ensemble Detector Features

Tests for new ensemble features including:
- Advanced fusion strategies (NMS, consensus, confidence-weighted)
- Adaptive detector selection
- Parallel/sequential execution
- Performance tracking
- DetectorConfig and DetectorPriority
"""

import sys
import unittest
import time
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
    FusionStrategy,
    EnsembleStrategy,
    DetectorPriority,
    DetectorConfig,
    EnsembleDetection,
    DetectionFuser,
    AdaptiveEnsembleSelector
)
from src.detectors.base_detector import (
    FaceDetectionResult,
    DetectionMetrics,
    DetectorType
)
from src.detectors.detector_factory import MockDetector


class TestFusionStrategy(unittest.TestCase):
    """Test suite for FusionStrategy enum."""
    
    def test_fusion_strategy_values(self):
        """Test FusionStrategy enum values."""
        self.assertEqual(FusionStrategy.SIMPLE_VOTING.value, "simple_voting")
        self.assertEqual(FusionStrategy.WEIGHTED_VOTING.value, "weighted_voting")
        self.assertEqual(FusionStrategy.CONSENSUS.value, "consensus")
        self.assertEqual(FusionStrategy.UNION.value, "union")
        self.assertEqual(FusionStrategy.INTERSECTION.value, "intersection")
        self.assertEqual(FusionStrategy.ADAPTIVE.value, "adaptive")
        self.assertEqual(FusionStrategy.CONFIDENCE_WEIGHTED.value, "confidence_weighted")
        self.assertEqual(FusionStrategy.NMS_FUSION.value, "nms_fusion")


class TestDetectorPriority(unittest.TestCase):
    """Test suite for DetectorPriority enum."""
    
    def test_priority_values(self):
        """Test DetectorPriority enum values."""
        self.assertEqual(DetectorPriority.LOW.value, 1)
        self.assertEqual(DetectorPriority.MEDIUM.value, 2)
        self.assertEqual(DetectorPriority.HIGH.value, 3)
        self.assertEqual(DetectorPriority.CRITICAL.value, 4)
    
    def test_priority_ordering(self):
        """Test priority ordering."""
        self.assertLess(DetectorPriority.LOW.value, DetectorPriority.MEDIUM.value)
        self.assertLess(DetectorPriority.MEDIUM.value, DetectorPriority.HIGH.value)
        self.assertLess(DetectorPriority.HIGH.value, DetectorPriority.CRITICAL.value)


class TestDetectorConfig(unittest.TestCase):
    """Test suite for DetectorConfig dataclass."""
    
    def test_detector_config_initialization(self):
        """Test DetectorConfig initialization with defaults."""
        detector = MockDetector({})
        
        config = DetectorConfig(detector_instance=detector)
        
        self.assertEqual(config.detector_instance, detector)
        self.assertEqual(config.weight, 1.0)
        self.assertEqual(config.priority, DetectorPriority.MEDIUM)
        self.assertEqual(config.timeout, 5.0)
        self.assertTrue(config.enabled)
        self.assertEqual(config.min_confidence, 0.3)
        self.assertEqual(config.max_detections, 10)
        self.assertTrue(config.use_for_speed)
        self.assertTrue(config.use_for_accuracy)
    
    def test_detector_config_custom_values(self):
        """Test DetectorConfig with custom values."""
        detector = MockDetector({})
        
        config = DetectorConfig(
            detector_instance=detector,
            weight=2.0,
            priority=DetectorPriority.HIGH,
            timeout=10.0,
            enabled=False,
            min_confidence=0.5,
            max_detections=5,
            use_for_speed=False,
            use_for_accuracy=True
        )
        
        self.assertEqual(config.weight, 2.0)
        self.assertEqual(config.priority, DetectorPriority.HIGH)
        self.assertEqual(config.timeout, 10.0)
        self.assertFalse(config.enabled)
        self.assertEqual(config.min_confidence, 0.5)
        self.assertEqual(config.max_detections, 5)
        self.assertFalse(config.use_for_speed)
        self.assertTrue(config.use_for_accuracy)


class TestEnsembleDetection(unittest.TestCase):
    """Test suite for EnsembleDetection dataclass."""
    
    def test_ensemble_detection_initialization(self):
        """Test EnsembleDetection initialization."""
        detection = EnsembleDetection(
            bbox=(10, 90, 90, 10),
            confidence=0.9,
            detector_votes=['cpu', 'gpu'],
            individual_confidences={'cpu': 0.85, 'gpu': 0.95}
        )
        
        self.assertEqual(detection.bbox, (10, 90, 90, 10))
        self.assertEqual(detection.confidence, 0.9)
        self.assertEqual(detection.detector_votes, ['cpu', 'gpu'])
        self.assertEqual(detection.individual_confidences, {'cpu': 0.85, 'gpu': 0.95})
    
    def test_ensemble_detection_to_dict(self):
        """Test EnsembleDetection to_dict conversion."""
        detection = EnsembleDetection(
            bbox=(10, 90, 90, 10),
            confidence=0.9,
            detector_votes=['cpu'],
            individual_confidences={'cpu': 0.9},
            fusion_score=0.85,
            consensus_level=0.75
        )
        
        result = detection.to_dict()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['bbox'], [10, 90, 90, 10])
        self.assertEqual(result['confidence'], 0.9)
        self.assertEqual(result['detector_votes'], ['cpu'])
        self.assertEqual(result['individual_confidences'], {'cpu': 0.9})
        self.assertEqual(result['fusion_score'], 0.85)
        self.assertEqual(result['consensus_level'], 0.75)
    
    def test_ensemble_detection_to_face_detection_result(self):
        """Test conversion to FaceDetectionResult."""
        detection = EnsembleDetection(
            bbox=(10, 90, 90, 10),
            confidence=0.9,
            fusion_score=0.85
        )
        
        result = detection.to_face_detection_result()
        
        self.assertIsInstance(result, FaceDetectionResult)
        self.assertEqual(result.bounding_box, (10, 90, 90, 10))
        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(result.quality_score, 0.85)


class TestDetectionFuser(unittest.TestCase):
    """Test suite for DetectionFuser class."""
    
    def setUp(self):
        """Set up test environment."""
        self.fuser = DetectionFuser({
            'iou_threshold': 0.5,
            'confidence_threshold': 0.5,
            'consensus_threshold': 0.6
        })
    
    def test_calculate_iou_identical_boxes(self):
        """Test IoU calculation with identical boxes."""
        box1 = (10, 90, 90, 10)
        box2 = (10, 90, 90, 10)
        
        iou = self.fuser.calculate_iou(box1, box2)
        
        self.assertAlmostEqual(iou, 1.0, places=2)
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU calculation with non-overlapping boxes."""
        box1 = (10, 30, 30, 10)
        box2 = (50, 70, 70, 50)
        
        iou = self.fuser.calculate_iou(box1, box2)
        
        self.assertEqual(iou, 0.0)
    
    def test_calculate_iou_partial_overlap(self):
        """Test IoU calculation with partial overlap."""
        box1 = (10, 60, 60, 10)
        box2 = (30, 80, 80, 30)
        
        iou = self.fuser.calculate_iou(box1, box2)
        
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)
    
    def test_simple_voting_fusion(self):
        """Test simple voting fusion."""
        detector1 = MockDetector({})
        detector2 = MockDetector({})
        
        detector_configs = {
            'det1': DetectorConfig(detector1, weight=1.0),
            'det2': DetectorConfig(detector2, weight=1.0)
        }
        
        # Create similar detections
        detection1 = FaceDetectionResult((10, 90, 90, 10), 0.9)
        detection2 = FaceDetectionResult((12, 88, 88, 12), 0.85)
        
        detector_results = {
            'det1': [detection1],
            'det2': [detection2]
        }
        
        fused = self.fuser.simple_voting_fusion(detector_results, detector_configs)
        
        # Should merge into one detection (similar locations)
        self.assertGreater(len(fused), 0)
        self.assertIsInstance(fused[0], EnsembleDetection)
    
    def test_nms_fusion(self):
        """Test NMS fusion."""
        detector1 = MockDetector({})
        detector2 = MockDetector({})
        
        detector_configs = {
            'det1': DetectorConfig(detector1, weight=1.0),
            'det2': DetectorConfig(detector2, weight=0.8)
        }
        
        # Create overlapping detections with different confidences
        detection1 = FaceDetectionResult((10, 90, 90, 10), 0.95)
        detection2 = FaceDetectionResult((12, 88, 88, 12), 0.85)
        
        detector_results = {
            'det1': [detection1],
            'det2': [detection2]
        }
        
        fused = self.fuser.nms_fusion(detector_results, detector_configs)
        
        # NMS should keep highest confidence and suppress overlapping
        self.assertGreater(len(fused), 0)
        # Should have merged votes from both detectors
        self.assertGreaterEqual(len(fused[0].detector_votes), 1)
    
    def test_consensus_fusion(self):
        """Test consensus fusion."""
        detector_configs = {
            'det1': DetectorConfig(MockDetector({}), weight=1.0),
            'det2': DetectorConfig(MockDetector({}), weight=1.0),
            'det3': DetectorConfig(MockDetector({}), weight=1.0)
        }
        
        # Create similar detections (should reach consensus)
        detection1 = FaceDetectionResult((10, 90, 90, 10), 0.9)
        detection2 = FaceDetectionResult((12, 88, 88, 12), 0.85)
        detection3 = FaceDetectionResult((11, 89, 89, 11), 0.88)
        
        detector_results = {
            'det1': [detection1],
            'det2': [detection2],
            'det3': [detection3]
        }
        
        fused = self.fuser.consensus_fusion(detector_results, detector_configs)
        
        # Should reach consensus with 3/3 detectors
        self.assertGreater(len(fused), 0)
        self.assertGreaterEqual(fused[0].consensus_level, 0.6)
    
    def test_confidence_weighted_fusion(self):
        """Test confidence-weighted fusion."""
        detector_configs = {
            'det1': DetectorConfig(MockDetector({}), weight=1.0),
            'det2': DetectorConfig(MockDetector({}), weight=1.0)
        }
        
        # High and low confidence detections
        detection1 = FaceDetectionResult((10, 90, 90, 10), 0.95)
        detection2 = FaceDetectionResult((12, 88, 88, 12), 0.6)
        
        detector_results = {
            'det1': [detection1],
            'det2': [detection2]
        }
        
        fused = self.fuser.confidence_weighted_fusion(detector_results, detector_configs)
        
        # Should weight by confidence
        self.assertGreater(len(fused), 0)
        # Fused confidence should be between avg and max
        self.assertGreater(fused[0].confidence, 0.6)
        self.assertLess(fused[0].confidence, 0.95)


class TestAdaptiveEnsembleSelector(unittest.TestCase):
    """Test suite for AdaptiveEnsembleSelector class."""
    
    def setUp(self):
        """Set up test environment."""
        self.selector = AdaptiveEnsembleSelector({})
    
    def test_select_optimal_detectors_prefer_speed(self):
        """Test detector selection with speed preference."""
        detector_configs = {
            'cpu': DetectorConfig(
                MockDetector({}),
                priority=DetectorPriority.HIGH,
                use_for_speed=True,
                use_for_accuracy=False
            ),
            'gpu': DetectorConfig(
                MockDetector({}),
                priority=DetectorPriority.MEDIUM,
                use_for_speed=False,
                use_for_accuracy=True
            )
        }
        
        performance_requirements = {
            'max_latency': 0.5,
            'prefer_speed': True
        }
        
        selected = self.selector.select_optimal_detectors(
            detector_configs,
            performance_requirements
        )
        
        # Should prioritize speed detectors
        self.assertIn('cpu', selected)
    
    def test_select_optimal_detectors_prefer_accuracy(self):
        """Test detector selection with accuracy preference."""
        detector_configs = {
            'cpu': DetectorConfig(
                MockDetector({}),
                priority=DetectorPriority.MEDIUM,
                use_for_speed=True,
                use_for_accuracy=False
            ),
            'gpu': DetectorConfig(
                MockDetector({}),
                priority=DetectorPriority.HIGH,
                use_for_speed=False,
                use_for_accuracy=True
            )
        }
        
        performance_requirements = {
            'min_accuracy': 0.8,
            'prefer_speed': False
        }
        
        selected = self.selector.select_optimal_detectors(
            detector_configs,
            performance_requirements
        )
        
        # Should prioritize accuracy detectors
        self.assertIn('gpu', selected)
    
    def test_update_performance_history(self):
        """Test performance history tracking."""
        self.selector.update_performance_history('cpu', 0.1, 0.85)
        self.selector.update_performance_history('cpu', 0.12, 0.87)
        
        self.assertIn('cpu', self.selector.performance_history)
        history = self.selector.performance_history['cpu']
        
        self.assertEqual(len(history['latencies']), 2)
        self.assertEqual(len(history['accuracies']), 2)
        self.assertGreater(history['avg_latency'], 0)
        self.assertGreater(history['avg_accuracy'], 0)


class TestEnhancedEnsembleDetector(unittest.TestCase):
    """Test suite for enhanced EnsembleDetector features."""
    
    def setUp(self):
        """Set up test environment."""
        if NUMPY_AVAILABLE:
            self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        else:
            self.test_image = np.zeros((100, 100, 3))
    
    def test_add_detector(self):
        """Test adding detector to ensemble."""
        ensemble = EnsembleDetector()
        detector = MockDetector({})
        
        ensemble.add_detector('test_detector', detector, weight=1.5, priority=DetectorPriority.HIGH)
        
        self.assertIn('test_detector', ensemble.detector_configs)
        config = ensemble.detector_configs['test_detector']
        self.assertEqual(config.weight, 1.5)
        self.assertEqual(config.priority, DetectorPriority.HIGH)
    
    def test_remove_detector(self):
        """Test removing detector from ensemble."""
        ensemble = EnsembleDetector()
        detector = MockDetector({})
        
        ensemble.add_detector('test_detector', detector)
        self.assertIn('test_detector', ensemble.detector_configs)
        
        result = ensemble.remove_detector('test_detector')
        
        self.assertTrue(result)
        self.assertNotIn('test_detector', ensemble.detector_configs)
    
    def test_remove_nonexistent_detector(self):
        """Test removing detector that doesn't exist."""
        ensemble = EnsembleDetector()
        
        result = ensemble.remove_detector('nonexistent')
        
        self.assertFalse(result)
    
    def test_load_model(self):
        """Test load_model initialization."""
        ensemble = EnsembleDetector()
        ensemble.add_detector('test', MockDetector({}))
        
        result = ensemble.load_model()
        
        self.assertTrue(result)
        self.assertTrue(ensemble.is_initialized)
    
    def test_load_model_already_initialized(self):
        """Test load_model when already initialized."""
        ensemble = EnsembleDetector()
        ensemble.add_detector('test', MockDetector({}))
        
        ensemble.load_model()
        result = ensemble.load_model()  # Second call
        
        self.assertTrue(result)
    
    def test_detect_faces_with_performance_requirements(self):
        """Test detection with performance requirements."""
        detector1 = MockDetector({})
        detector2 = MockDetector({})
        
        face = FaceDetectionResult((10, 90, 90, 10), 0.9)
        metrics = DetectionMetrics(inference_time=0.01, total_time=0.015)
        
        with patch.object(detector1, 'detect_faces', return_value=([face], metrics)):
            with patch.object(detector2, 'detect_faces', return_value=([face], metrics)):
                ensemble = EnsembleDetector()
                ensemble.add_detector('det1', detector1, use_for_speed=True)
                ensemble.add_detector('det2', detector2, use_for_accuracy=True)
                
                performance_requirements = {
                    'max_latency': 1.0,
                    'prefer_speed': True
                }
                
                results, result_metrics = ensemble.detect_faces(
                    self.test_image,
                    performance_requirements
                )
                
                self.assertIsInstance(results, list)
                self.assertIsInstance(result_metrics, DetectionMetrics)
    
    def test_parallel_execution_mode(self):
        """Test parallel execution mode."""
        ensemble = EnsembleDetector(config={'parallel_execution': True, 'max_workers': 2})
        ensemble.add_detector('det1', MockDetector({}))
        ensemble.add_detector('det2', MockDetector({}))
        
        self.assertTrue(ensemble.parallel_execution)
        self.assertEqual(ensemble.max_workers, 2)
    
    def test_sequential_execution_mode(self):
        """Test sequential execution mode."""
        ensemble = EnsembleDetector(config={'parallel_execution': False})
        ensemble.add_detector('det1', MockDetector({}))
        
        self.assertFalse(ensemble.parallel_execution)
    
    def test_ensemble_metrics_tracking(self):
        """Test ensemble performance metrics tracking."""
        ensemble = EnsembleDetector()
        
        self.assertIn('total_detections', ensemble.ensemble_metrics)
        self.assertIn('successful_detections', ensemble.ensemble_metrics)
        self.assertIn('fusion_times', ensemble.ensemble_metrics)
        self.assertIn('detector_performance', ensemble.ensemble_metrics)
    
    def test_fusion_strategy_configuration(self):
        """Test different fusion strategies."""
        strategies = [
            FusionStrategy.SIMPLE_VOTING,
            FusionStrategy.WEIGHTED_VOTING,
            FusionStrategy.CONSENSUS,
            FusionStrategy.NMS_FUSION,
            FusionStrategy.CONFIDENCE_WEIGHTED
        ]
        
        for strategy in strategies:
            ensemble = EnsembleDetector(config={'fusion_strategy': strategy.value})
            self.assertEqual(ensemble.fusion_strategy, strategy)
    
    def test_backward_compatibility_with_legacy_api(self):
        """Test backward compatibility with legacy constructor."""
        detector1 = MockDetector({})
        detector2 = MockDetector({})
        
        ensemble = EnsembleDetector(
            detectors=[detector1, detector2],
            strategy=EnsembleStrategy.VOTING
        )
        
        self.assertEqual(len(ensemble.component_detectors), 2)
        self.assertEqual(ensemble.ensemble_strategy, EnsembleStrategy.VOTING)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for real-world scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        if NUMPY_AVAILABLE:
            self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        else:
            self.test_image = np.zeros((100, 100, 3))
    
    def test_multi_detector_ensemble_with_different_priorities(self):
        """Test ensemble with multiple detectors at different priorities."""
        ensemble = EnsembleDetector(config={'fusion_strategy': 'weighted_voting'})
        
        # Add detectors with different priorities and weights
        ensemble.add_detector('cpu', MockDetector({}), weight=1.0, priority=DetectorPriority.MEDIUM)
        ensemble.add_detector('gpu', MockDetector({}), weight=1.5, priority=DetectorPriority.HIGH)
        ensemble.add_detector('edgetpu', MockDetector({}), weight=2.0, priority=DetectorPriority.CRITICAL)
        
        self.assertEqual(len(ensemble.detector_configs), 3)
        
        # Verify priorities
        self.assertEqual(ensemble.detector_configs['cpu'].priority, DetectorPriority.MEDIUM)
        self.assertEqual(ensemble.detector_configs['gpu'].priority, DetectorPriority.HIGH)
        self.assertEqual(ensemble.detector_configs['edgetpu'].priority, DetectorPriority.CRITICAL)
    
    def test_adaptive_detector_selection_high_accuracy_mode(self):
        """Test adaptive selection in high accuracy mode."""
        ensemble = EnsembleDetector()
        
        ensemble.add_detector('fast', MockDetector({}), 
                            priority=DetectorPriority.MEDIUM,
                            use_for_speed=True,
                            use_for_accuracy=False)
        ensemble.add_detector('accurate', MockDetector({}),
                            priority=DetectorPriority.HIGH,
                            use_for_speed=False,
                            use_for_accuracy=True)
        
        selected = ensemble.adaptive_selector.select_optimal_detectors(
            ensemble.detector_configs,
            {'prefer_speed': False, 'min_accuracy': 0.9}
        )
        
        # Should select accuracy-focused detector
        self.assertIn('accurate', selected)
    
    def test_adaptive_detector_selection_high_speed_mode(self):
        """Test adaptive selection in high speed mode."""
        ensemble = EnsembleDetector()
        
        ensemble.add_detector('fast', MockDetector({}),
                            priority=DetectorPriority.HIGH,
                            use_for_speed=True,
                            use_for_accuracy=False)
        ensemble.add_detector('accurate', MockDetector({}),
                            priority=DetectorPriority.MEDIUM,
                            use_for_speed=False,
                            use_for_accuracy=True)
        
        selected = ensemble.adaptive_selector.select_optimal_detectors(
            ensemble.detector_configs,
            {'prefer_speed': True, 'max_latency': 0.1}
        )
        
        # Should select speed-optimized detector
        self.assertIn('fast', selected)


if __name__ == '__main__':
    unittest.main()
