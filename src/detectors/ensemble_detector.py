#!/usr/bin/env python3
"""
Production Ensemble Face Detection System

Combines multiple detection models with intelligent fusion algorithms
to maximize accuracy while optimizing performance for real-time applications.
"""

import logging
import time
import statistics
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from src.detectors.base_detector import (
    BaseDetector,
    DetectorType,
    ModelType,
    FaceDetectionResult,
    DetectionMetrics
)

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Available detection fusion strategies."""
    SIMPLE_VOTING = "simple_voting"
    WEIGHTED_VOTING = "weighted_voting"
    CONSENSUS = "consensus"
    UNION = "union"
    INTERSECTION = "intersection"
    ADAPTIVE = "adaptive"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    NMS_FUSION = "nms_fusion"


# Backwards compatibility alias
class EnsembleStrategy(Enum):
    """Ensemble voting strategies (legacy compatibility)."""
    VOTING = "voting"                        # Majority voting (maps to simple_voting)
    WEIGHTED_VOTING = "weighted_voting"  # Weighted by confidence
    UNION = "union"                      # Union of all detections
    INTERSECTION = "intersection"         # Only common detections
    BEST_CONFIDENCE = "best_confidence"  # Highest confidence detector


class DetectorPriority(Enum):
    """Detector priority levels for adaptive selection."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class DetectorConfig:
    """Configuration for individual detector in ensemble."""
    detector_instance: BaseDetector
    weight: float = 1.0
    priority: DetectorPriority = DetectorPriority.MEDIUM
    timeout: float = 5.0
    enabled: bool = True
    min_confidence: float = 0.3
    max_detections: int = 10
    use_for_speed: bool = True
    use_for_accuracy: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleDetection:
    """Enhanced detection result with ensemble metadata."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) or (top, right, bottom, left)
    confidence: float
    detector_votes: List[str] = field(default_factory=list)
    individual_confidences: Dict[str, float] = field(default_factory=dict)
    fusion_score: float = 0.0
    consensus_level: float = 0.0
    detection_source: str = "ensemble"
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'bbox': list(self.bbox),
            'confidence': self.confidence,
            'detector_votes': self.detector_votes,
            'individual_confidences': self.individual_confidences,
            'fusion_score': self.fusion_score,
            'consensus_level': self.consensus_level,
            'detection_source': self.detection_source,
            'processing_time': self.processing_time
        }

    def to_face_detection_result(self) -> FaceDetectionResult:
        """Convert to FaceDetectionResult format."""
        # Assume bbox is (top, right, bottom, left) format
        return FaceDetectionResult(
            bounding_box=self.bbox,
            confidence=self.confidence,
            landmarks=None,
            quality_score=self.fusion_score
        )


@dataclass
class EnsembleMetadata:
    """Metadata about ensemble detection process (legacy compatibility)."""

    strategy: str
    component_results: List[Dict[str, Any]] = field(default_factory=list)
    voting_results: Optional[Dict[str, Any]] = None
    agreement_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'strategy': self.strategy,
            'component_results': self.component_results,
            'voting_results': self.voting_results,
            'agreement_score': self.agreement_score,
        }


class DetectionFuser:
    """Advanced detection fusion algorithms."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.iou_threshold = config.get('iou_threshold', 0.5)
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.consensus_threshold = config.get('consensus_threshold', 0.6)

    def calculate_iou(self, box1: Tuple[int, int, int, int],
                      box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        # Boxes are in (top, right, bottom, left) format
        top1, right1, bottom1, left1 = box1
        top2, right2, bottom2, left2 = box2

        # Calculate intersection
        inter_left = max(left1, left2)
        inter_top = max(top1, top2)
        inter_right = min(right1, right2)
        inter_bottom = min(bottom1, bottom2)

        if inter_right < inter_left or inter_bottom < inter_top:
            return 0.0

        inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)

        # Calculate union
        box1_area = (right1 - left1) * (bottom1 - top1)
        box2_area = (right2 - left2) * (bottom2 - top2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _convert_detection_to_ensemble(self, detection: FaceDetectionResult,
                                       detector_name: str) -> EnsembleDetection:
        """Convert FaceDetectionResult to EnsembleDetection."""
        return EnsembleDetection(
            bbox=detection.bounding_box,
            confidence=detection.confidence,
            detector_votes=[detector_name],
            individual_confidences={detector_name: detection.confidence}
        )

    def simple_voting_fusion(self,
                             detector_results: Dict[str,
                                                    List[FaceDetectionResult]],
                             detector_configs: Dict[str,
                                                    DetectorConfig]) -> List[EnsembleDetection]:
        """Simple majority voting fusion."""
        all_detections = []

        # Collect all detections
        for detector_name, detections in detector_results.items():
            for detection in detections:
                ensemble_detection = self._convert_detection_to_ensemble(detection, detector_name)
                all_detections.append(ensemble_detection)

        # Group similar detections
        grouped_detections = self._group_similar_detections(all_detections)

        # Filter by minimum votes
        min_votes = max(1, len(detector_results) // 2)
        final_detections = []

        for group in grouped_detections:
            if len(group) >= min_votes:
                # Merge group into single detection
                merged_detection = self._merge_detection_group(group)
                final_detections.append(merged_detection)

        return final_detections

    def weighted_voting_fusion(self,
                               detector_results: Dict[str,
                                                      List[FaceDetectionResult]],
                               detector_configs: Dict[str,
                                                      DetectorConfig]) -> List[EnsembleDetection]:
        """Weighted voting fusion based on detector weights."""
        all_detections = []

        # Collect all detections with weights
        for detector_name, detections in detector_results.items():
            weight = detector_configs[detector_name].weight

            for detection in detections:
                # Apply weight to confidence
                weighted_confidence = detection.confidence * weight

                ensemble_detection = EnsembleDetection(
                    bbox=detection.bounding_box,
                    confidence=weighted_confidence,
                    detector_votes=[detector_name],
                    individual_confidences={detector_name: detection.confidence},
                    fusion_score=weight
                )
                all_detections.append(ensemble_detection)

        # Group and merge similar detections
        grouped_detections = self._group_similar_detections(all_detections)
        final_detections = []

        for group in grouped_detections:
            # Calculate total weight for group
            total_weight = sum(d.fusion_score for d in group)
            threshold_weight = sum(detector_configs[name].weight for name in detector_configs) * 0.3

            if total_weight >= threshold_weight:
                merged_detection = self._merge_detection_group(group)
                final_detections.append(merged_detection)

        return final_detections

    def consensus_fusion(self, detector_results: Dict[str, List[FaceDetectionResult]],
                         detector_configs: Dict[str, DetectorConfig]) -> List[EnsembleDetection]:
        """Consensus-based fusion requiring agreement from multiple detectors."""
        all_detections = []

        # Collect all detections
        for detector_name, detections in detector_results.items():
            for detection in detections:
                ensemble_detection = self._convert_detection_to_ensemble(detection, detector_name)
                all_detections.append(ensemble_detection)

        # Group similar detections
        grouped_detections = self._group_similar_detections(all_detections)
        final_detections = []

        for group in grouped_detections:
            # Calculate consensus level
            consensus_level = len(group) / len(detector_results)

            if consensus_level >= self.consensus_threshold:
                merged_detection = self._merge_detection_group(group)
                merged_detection.consensus_level = consensus_level
                final_detections.append(merged_detection)

        return final_detections

    def confidence_weighted_fusion(
            self, detector_results: Dict[str, List[FaceDetectionResult]],
            detector_configs: Dict[str, DetectorConfig]) -> List[EnsembleDetection]:
        """Confidence-weighted fusion emphasizing high-confidence
        detections."""
        all_detections = []

        # Collect all detections
        for detector_name, detections in detector_results.items():
            for detection in detections:
                ensemble_detection = self._convert_detection_to_ensemble(detection, detector_name)
                all_detections.append(ensemble_detection)

        # Group similar detections
        grouped_detections = self._group_similar_detections(all_detections)
        final_detections = []

        for group in grouped_detections:
            # Calculate confidence-weighted score
            confidence_weights = [d.confidence for d in group]
            avg_confidence = statistics.mean(confidence_weights)
            max_confidence = max(confidence_weights)

            # Combine average and max confidence
            fusion_confidence = (avg_confidence * 0.6) + (max_confidence * 0.4)

            if fusion_confidence >= self.confidence_threshold:
                merged_detection = self._merge_detection_group(group)
                merged_detection.confidence = fusion_confidence
                final_detections.append(merged_detection)

        return final_detections

    def nms_fusion(self, detector_results: Dict[str, List[FaceDetectionResult]],
                   detector_configs: Dict[str, DetectorConfig]) -> List[EnsembleDetection]:
        """Non-Maximum Suppression fusion to remove overlapping detections."""
        all_detections = []

        # Collect all detections
        for detector_name, detections in detector_results.items():
            weight = detector_configs[detector_name].weight

            for detection in detections:
                ensemble_detection = EnsembleDetection(
                    bbox=detection.bounding_box,
                    confidence=detection.confidence * weight,
                    detector_votes=[detector_name],
                    individual_confidences={detector_name: detection.confidence},
                    fusion_score=weight
                )
                all_detections.append(ensemble_detection)

        # Sort by confidence
        all_detections.sort(key=lambda x: x.confidence, reverse=True)

        # Apply NMS
        final_detections = []

        while all_detections:
            # Take highest confidence detection
            current_detection = all_detections.pop(0)
            final_detections.append(current_detection)

            # Remove overlapping detections
            remaining_detections = []
            for detection in all_detections:
                iou = self.calculate_iou(current_detection.bbox, detection.bbox)
                if iou < self.iou_threshold:
                    remaining_detections.append(detection)
                else:
                    # Merge overlapping detection information
                    current_detection.detector_votes.extend(detection.detector_votes)
                    current_detection.individual_confidences.update(
                        detection.individual_confidences)

            all_detections = remaining_detections

        return final_detections

    def _group_similar_detections(
            self, detections: List[EnsembleDetection]) -> List[List[EnsembleDetection]]:
        """Group detections that are spatially similar."""
        groups = []
        used_indices = set()

        for i, detection in enumerate(detections):
            if i in used_indices:
                continue

            # Start new group
            current_group = [detection]
            used_indices.add(i)

            # Find similar detections
            for j, other_detection in enumerate(detections[i + 1:], i + 1):
                if j in used_indices:
                    continue

                iou = self.calculate_iou(detection.bbox, other_detection.bbox)
                if iou >= self.iou_threshold:
                    current_group.append(other_detection)
                    used_indices.add(j)

            groups.append(current_group)

        return groups

    def _merge_detection_group(self, group: List[EnsembleDetection]) -> EnsembleDetection:
        """Merge a group of similar detections into single detection."""
        if len(group) == 1:
            return group[0]

        # Calculate weighted average bbox
        total_confidence = sum(d.confidence for d in group)

        avg_top = sum(d.bbox[0] * d.confidence for d in group) / total_confidence
        avg_right = sum(d.bbox[1] * d.confidence for d in group) / total_confidence
        avg_bottom = sum(d.bbox[2] * d.confidence for d in group) / total_confidence
        avg_left = sum(d.bbox[3] * d.confidence for d in group) / total_confidence

        merged_bbox = (int(avg_top), int(avg_right), int(avg_bottom), int(avg_left))

        # Merge metadata
        all_votes = []
        all_confidences = {}

        for detection in group:
            all_votes.extend(detection.detector_votes)
            all_confidences.update(detection.individual_confidences)

        # Calculate merged confidence
        merged_confidence = total_confidence / len(group)

        return EnsembleDetection(
            bbox=merged_bbox,
            confidence=merged_confidence,
            detector_votes=list(set(all_votes)),
            individual_confidences=all_confidences,
            fusion_score=total_confidence
        )


class AdaptiveEnsembleSelector:
    """Adaptive selector for optimal detector combination based on conditions."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_history = {}
        self.selection_criteria = config.get('selection_criteria', {
            'latency_weight': 0.4,
            'accuracy_weight': 0.6,
            'min_accuracy': 0.8,
            'max_latency': 0.5  # seconds
        })

    def select_optimal_detectors(self, detector_configs: Dict[str, DetectorConfig],
                                 performance_requirements: Dict[str, Any]) -> List[str]:
        """Select optimal detector combination based on requirements."""

        # Get requirements
        max_latency = performance_requirements.get('max_latency', 1.0)
        min_accuracy = performance_requirements.get('min_accuracy', 0.7)
        prefer_speed = performance_requirements.get('prefer_speed', False)

        available_detectors = [name for name, config in detector_configs.items() if config.enabled]

        if prefer_speed:
            # Prioritize fast detectors
            selected_detectors = []

            # Add speed-optimized detectors first
            for detector_name in available_detectors:
                config = detector_configs[detector_name]
                if config.use_for_speed and config.priority.value >= DetectorPriority.MEDIUM.value:
                    selected_detectors.append(detector_name)

            # If not enough selected, add accuracy-focused detectors
            if len(selected_detectors) < 2:
                for detector_name in available_detectors:
                    if detector_name not in selected_detectors:
                        config = detector_configs[detector_name]
                        if config.use_for_accuracy:
                            selected_detectors.append(detector_name)
                            if len(selected_detectors) >= 2:
                                break
        else:
            # Prioritize accuracy
            selected_detectors = []

            # Add accuracy-focused detectors first
            for detector_name in available_detectors:
                config = detector_configs[detector_name]
                if config.use_for_accuracy and config.priority.value >= DetectorPriority.MEDIUM.value:
                    selected_detectors.append(detector_name)

            # Ensure at least one speed detector for responsiveness
            speed_detector_added = False
            for detector_name in available_detectors:
                if detector_name not in selected_detectors:
                    config = detector_configs[detector_name]
                    if config.use_for_speed and not speed_detector_added:
                        selected_detectors.append(detector_name)
                        speed_detector_added = True
                        break

        # Ensure we have at least one detector
        if not selected_detectors and available_detectors:
            selected_detectors = [available_detectors[0]]

        logger.info(f"Selected detectors: {selected_detectors} (prefer_speed: {prefer_speed})")
        return selected_detectors

    def update_performance_history(self, detector_name: str,
                                   latency: float, accuracy: float) -> None:
        """Update performance history for adaptive selection."""
        if detector_name not in self.performance_history:
            self.performance_history[detector_name] = {
                'latencies': [],
                'accuracies': [],
                'avg_latency': 0.0,
                'avg_accuracy': 0.0
            }

        history = self.performance_history[detector_name]
        history['latencies'].append(latency)
        history['accuracies'].append(accuracy)

        # Keep only recent history
        max_history = 100
        if len(history['latencies']) > max_history:
            history['latencies'] = history['latencies'][-max_history:]
            history['accuracies'] = history['accuracies'][-max_history:]

        # Update averages
        history['avg_latency'] = statistics.mean(history['latencies'])
        history['avg_accuracy'] = statistics.mean(history['accuracies'])


class EnsembleDetector(BaseDetector):
    """
    Production ensemble face detector with advanced fusion strategies.

    Combines multiple detection models with intelligent fusion algorithms,
    adaptive detector selection, and performance optimization.
    """

    def __init__(
        self,
        detectors: Optional[List[BaseDetector]] = None,
        strategy: Optional[EnsembleStrategy] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ensemble detector.

        Args:
            detectors: Optional list of component detector instances (for legacy compatibility)
            strategy: Optional ensemble strategy to use (for legacy compatibility)
            config: Optional configuration
        """
        # Backward compatibility: if detectors is explicitly an empty list, raise error
        if detectors is not None and len(detectors) == 0:
            raise ValueError("At least one detector required for ensemble")

        # Initialize with config
        config = config or {}
        config['model_type'] = 'ensemble'

        # Legacy fields for backward compatibility - INITIALIZE FIRST
        self.component_detectors = []
        self.ensemble_strategy = strategy or EnsembleStrategy.VOTING
        self._config = config
        self._detectors = []
        self._strategy = strategy or EnsembleStrategy.VOTING

        # Map EnsembleStrategy to FusionStrategy
        strategy_mapping = {
            'voting': 'simple_voting',
            'weighted_voting': 'weighted_voting',
            'union': 'union',
            'intersection': 'intersection',
            'best_confidence': 'confidence_weighted'  # Map to confidence_weighted
        }

        # Configuration
        if strategy:
            fusion_strategy_value = strategy_mapping.get(strategy.value, strategy.value)
        else:
            fusion_strategy_value = config.get('fusion_strategy', 'simple_voting')

        self.fusion_strategy = FusionStrategy(fusion_strategy_value)
        self.parallel_execution = config.get('parallel_execution', True)
        self.timeout = config.get('timeout', 10.0)
        self.max_workers = config.get('max_workers', 4)

        # Components
        self.detector_configs: Dict[str, DetectorConfig] = {}
        self.fusion_engine = DetectionFuser(config.get('fusion_config', {}))
        self.adaptive_selector = AdaptiveEnsembleSelector(config.get('adaptive_config', {}))

        # State
        self.is_initialized = False
        self.executor = None

        # Performance tracking
        self.ensemble_metrics = {
            'total_detections': 0,
            'successful_detections': 0,
            'fusion_times': [],
            'detector_performance': {}
        }

        # Legacy support: if detectors provided, add them
        if detectors:
            for i, detector in enumerate(detectors):
                self.add_detector(
                    f"detector_{i}",
                    detector,
                    weight=1.0,
                    priority=DetectorPriority.MEDIUM
                )

        # Set required BaseDetector attributes
        self.config = config
        self.detector_type = DetectorType.CPU  # Default type for ensemble
        self.model_type = ModelType.HOG  # Placeholder
        self.logger = logging.getLogger(self.__class__.__name__)

        # BaseDetector performance tracking
        self.total_detections = 0
        self.total_inference_time = 0.0
        self.last_metrics: Optional[DetectionMetrics] = None

        # Detection parameters
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.min_face_size = config.get('min_face_size', (30, 30))
        self.max_face_size = config.get('max_face_size', (1000, 1000))

        # Ensemble-specific config (legacy)
        self.min_agreement = config.get('min_agreement', 0.5)
        self.iou_threshold = config.get('iou_threshold', 0.5)

        logger.info(f"Ensemble detector created with strategy: {self.fusion_strategy.value}")

    @classmethod
    def is_available(cls) -> bool:
        """Ensemble detector is available if any component detector is available."""
        return True

    def _get_detector_type(self) -> DetectorType:
        """Return detector type as CPU (ensemble operates at CPU level)."""
        return DetectorType.CPU

    def _initialize_model(self) -> None:
        """Model initialization handled by component detectors."""
        pass

    def add_detector(self, name: str, detector: BaseDetector,
                     weight: float = 1.0, priority: DetectorPriority = DetectorPriority.MEDIUM,
                     **kwargs) -> None:
        """Add a detector to the ensemble."""
        detector_config = DetectorConfig(
            detector_instance=detector,
            weight=weight,
            priority=priority,
            **kwargs
        )

        self.detector_configs[name] = detector_config

        # Update legacy component_detectors list
        if detector not in self.component_detectors:
            self.component_detectors.append(detector)

        logger.info(
            f"Added detector '{name}' to ensemble (weight: {weight}, priority: {
                priority.name})")

    def remove_detector(self, name: str) -> bool:
        """Remove a detector from the ensemble."""
        if name in self.detector_configs:
            detector = self.detector_configs[name].detector_instance
            del self.detector_configs[name]

            # Update legacy component_detectors list
            if detector in self.component_detectors:
                self.component_detectors.remove(detector)

            logger.info(f"Removed detector '{name}' from ensemble")
            return True
        return False

    def load_model(self) -> bool:
        """Initialize all detectors in the ensemble."""
        if self.is_initialized:
            return True

        try:
            # Initialize thread pool for parallel execution
            if self.parallel_execution:
                self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

            # Initialize all detectors
            initialization_results = {}

            for name, config in self.detector_configs.items():
                try:
                    # Detectors are typically initialized in their __init__
                    # But we can call load_model if it exists
                    if hasattr(config.detector_instance, 'load_model'):
                        success = config.detector_instance.load_model()
                    else:
                        success = True

                    initialization_results[name] = success

                    if success:
                        logger.info(f"Successfully initialized detector: {name}")
                    else:
                        logger.warning(f"Failed to initialize detector: {name}")
                        config.enabled = False

                except Exception as e:
                    logger.error(f"Error initializing detector {name}: {e}")
                    initialization_results[name] = False
                    config.enabled = False

            # Check if at least one detector is available
            enabled_detectors = [name for name,
                                 config in self.detector_configs.items() if config.enabled]

            if not enabled_detectors:
                logger.error("No detectors successfully initialized")
                return False

            self.is_initialized = True
            logger.info(
                f"Ensemble detector initialized with {
                    len(enabled_detectors)} active detectors")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize ensemble detector: {e}")
            return False

    def detect_faces(self,
                     image: np.ndarray,
                     performance_requirements: Optional[Dict[str,
                                                             Any]] = None) -> Tuple[List[FaceDetectionResult],
                                                                                    DetectionMetrics]:
        """
        Detect faces using ensemble of detectors (enhanced version).

        Args:
            image: Input image as numpy array
            performance_requirements: Optional performance requirements for adaptive selection

        Returns:
            Tuple of (detection results, performance metrics)
        """
        # If using new API with detector_configs
        if self.detector_configs:
            return self._detect_faces_enhanced(image, performance_requirements)

        # Otherwise use legacy implementation
        return super().detect_faces(image)

    def _detect_faces_enhanced(
            self, image: np.ndarray,
            performance_requirements: Optional[Dict[str, Any]]
    ) -> Tuple[List[FaceDetectionResult], DetectionMetrics]:
        """Enhanced detection with adaptive selection and fusion."""
        if not self.is_initialized:
            if not self.load_model():
                return [], DetectionMetrics()

        start_time = time.time()

        try:
            # Select optimal detectors based on requirements
            if performance_requirements:
                selected_detector_names = self.adaptive_selector.select_optimal_detectors(
                    self.detector_configs, performance_requirements
                )
            else:
                selected_detector_names = [name for name,
                                           config in self.detector_configs.items() if config.enabled]

            if not selected_detector_names:
                logger.warning("No detectors selected for ensemble detection")
                return [], DetectionMetrics()

            logger.debug(
                f"Running ensemble detection with {
                    len(selected_detector_names)} detectors")

            # Run detection on selected detectors
            detector_results: Dict[str, List[FaceDetectionResult]] = {}

            if self.parallel_execution and len(selected_detector_names) > 1:
                # Parallel execution
                detector_results = self._run_parallel_detection(image, selected_detector_names)
            else:
                # Sequential execution
                detector_results = self._run_sequential_detection(image, selected_detector_names)

            # Fuse detection results
            fusion_start = time.time()
            ensemble_detections = self._fuse_detections_new(detector_results)
            fusion_time = time.time() - fusion_start

            # Convert to FaceDetectionResult format
            final_results = [ed.to_face_detection_result() for ed in ensemble_detections]

            # Update metrics
            total_time = time.time() - start_time
            self._update_metrics(detector_results, fusion_time, total_time)

            # Create detection metrics
            metrics = DetectionMetrics(
                inference_time=total_time -
                fusion_time,
                preprocessing_time=0.0,
                postprocessing_time=fusion_time,
                total_time=total_time,
                face_count=len(final_results),
                confidence=sum(
                    r.confidence for r in final_results) /
                len(final_results) if final_results else 0.0)

            logger.debug(
                f"Ensemble detection completed in {
                    total_time *
                    1000:.2f}ms, found {
                    len(final_results)} faces")

            return final_results, metrics

        except Exception as e:
            logger.error(f"Ensemble detection failed: {e}")
            return [], DetectionMetrics(total_time=time.time() - start_time)

    def _run_parallel_detection(self, image: np.ndarray,
                                detector_names: List[str]) -> Dict[str, List[FaceDetectionResult]]:
        """Run detection on multiple detectors in parallel."""
        detector_results = {}

        # Submit detection tasks
        future_to_detector = {}

        for detector_name in detector_names:
            config = self.detector_configs[detector_name]
            if config.enabled:
                future = self.executor.submit(
                    self._safe_detect,
                    detector_name,
                    config.detector_instance,
                    image,
                    config.timeout
                )
                future_to_detector[future] = detector_name

        # Collect results
        for future in as_completed(future_to_detector.keys(), timeout=self.timeout):
            detector_name = future_to_detector[future]

            try:
                result = future.result()
                if result is not None:
                    detector_results[detector_name] = result
                    logger.debug(
                        f"Detector {detector_name} completed with {
                            len(result)} detections")
            except Exception as e:
                logger.error(f"Detector {detector_name} failed: {e}")

        return detector_results

    def _run_sequential_detection(self,
                                  image: np.ndarray,
                                  detector_names: List[str]) -> Dict[str,
                                                                     List[FaceDetectionResult]]:
        """Run detection on multiple detectors sequentially."""
        detector_results = {}

        for detector_name in detector_names:
            config = self.detector_configs[detector_name]
            if config.enabled:
                result = self._safe_detect(
                    detector_name, config.detector_instance, image, config.timeout)
                if result is not None:
                    detector_results[detector_name] = result

        return detector_results

    def _safe_detect(self, detector_name: str, detector: BaseDetector,
                     image: np.ndarray, timeout: float) -> Optional[List[FaceDetectionResult]]:
        """Safely run detection with timeout and error handling."""
        try:
            detections, _ = detector.detect_faces(image)
            return detections
        except Exception as e:
            logger.error(f"Detection failed for {detector_name}: {e}")
            return None

    def _fuse_detections_new(self,
                             detector_results: Dict[str,
                                                    List[FaceDetectionResult]]) -> List[EnsembleDetection]:
        """Fuse detection results using selected fusion strategy."""
        if not detector_results:
            return []

        # Map fusion strategy to fusion method
        if self.fusion_strategy == FusionStrategy.SIMPLE_VOTING:
            return self.fusion_engine.simple_voting_fusion(detector_results, self.detector_configs)
        elif self.fusion_strategy == FusionStrategy.WEIGHTED_VOTING:
            return self.fusion_engine.weighted_voting_fusion(
                detector_results, self.detector_configs)
        elif self.fusion_strategy == FusionStrategy.CONSENSUS:
            return self.fusion_engine.consensus_fusion(detector_results, self.detector_configs)
        elif self.fusion_strategy == FusionStrategy.CONFIDENCE_WEIGHTED:
            return self.fusion_engine.confidence_weighted_fusion(
                detector_results, self.detector_configs)
        elif self.fusion_strategy == FusionStrategy.NMS_FUSION:
            return self.fusion_engine.nms_fusion(detector_results, self.detector_configs)
        elif self.fusion_strategy == FusionStrategy.UNION:
            # Use simple voting with min_votes=1 for union
            all_detections = []
            for detector_name, detections in detector_results.items():
                for detection in detections:
                    all_detections.append(
                        self.fusion_engine._convert_detection_to_ensemble(
                            detection, detector_name))
            grouped = self.fusion_engine._group_similar_detections(all_detections)
            return [self.fusion_engine._merge_detection_group(g) for g in grouped]
        elif self.fusion_strategy == FusionStrategy.INTERSECTION:
            # Require all detectors to agree
            return self.fusion_engine.consensus_fusion(detector_results, self.detector_configs)
        else:
            logger.warning(f"Unknown fusion strategy {self.fusion_strategy}, using simple voting")
            return self.fusion_engine.simple_voting_fusion(detector_results, self.detector_configs)

    def _update_metrics(self, detector_results: Dict[str, List[FaceDetectionResult]],
                        fusion_time: float, total_time: float) -> None:
        """Update ensemble performance metrics."""
        self.ensemble_metrics['total_detections'] += 1
        if detector_results:
            self.ensemble_metrics['successful_detections'] += 1

        self.ensemble_metrics['fusion_times'].append(fusion_time)

        # Keep only recent history
        if len(self.ensemble_metrics['fusion_times']) > 100:
            self.ensemble_metrics['fusion_times'] = self.ensemble_metrics['fusion_times'][-100:]

        # Update per-detector performance
        for detector_name, detections in detector_results.items():
            if detector_name not in self.ensemble_metrics['detector_performance']:
                self.ensemble_metrics['detector_performance'][detector_name] = {
                    'calls': 0,
                    'total_detections': 0
                }

            perf = self.ensemble_metrics['detector_performance'][detector_name]
            perf['calls'] += 1
            perf['total_detections'] += len(detections)

    def _run_inference(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """
        Run ensemble inference on an image.

        Args:
            image: Input image in RGB format

        Returns:
            List of aggregated face detection results
        """
        # Collect results from all component detectors
        all_results: List[Tuple[BaseDetector, List[FaceDetectionResult], DetectionMetrics]] = []

        for detector in self.component_detectors:
            try:
                detections, metrics = detector.detect_faces(image)
                all_results.append((detector, detections, metrics))
            except Exception as e:
                logger.warning(
                    f"Component detector {detector.detector_type.value} failed: {e}"
                )
                # Continue with other detectors

        if not all_results:
            logger.error("All component detectors failed")
            return []

        # Apply ensemble strategy
        if self.ensemble_strategy == EnsembleStrategy.VOTING:
            return self._voting_strategy(all_results)
        elif self.ensemble_strategy == EnsembleStrategy.WEIGHTED_VOTING:
            return self._weighted_voting_strategy(all_results)
        elif self.ensemble_strategy == EnsembleStrategy.UNION:
            return self._union_strategy(all_results)
        elif self.ensemble_strategy == EnsembleStrategy.INTERSECTION:
            return self._intersection_strategy(all_results)
        elif self.ensemble_strategy == EnsembleStrategy.BEST_CONFIDENCE:
            return self._best_confidence_strategy(all_results)
        else:
            logger.warning(f"Unknown strategy {self.ensemble_strategy}, using voting")
            return self._voting_strategy(all_results)

    def _voting_strategy(
        self,
        all_results: List[Tuple[BaseDetector, List[FaceDetectionResult], DetectionMetrics]]
    ) -> List[FaceDetectionResult]:
        """
        Majority voting strategy - keep detections agreed upon by majority.

        Args:
            all_results: Results from all component detectors

        Returns:
            List of consensus detections
        """
        if not all_results:
            return []

        # Extract all detections
        all_detections = []
        for detector, detections, metrics in all_results:
            all_detections.extend(detections)

        if not all_detections:
            return []

        # Group similar detections
        groups = self._group_similar_detections(all_detections)

        # Keep groups with enough votes
        min_votes = max(1, int(len(self.component_detectors) * self.min_agreement))
        consensus_detections = []

        for group in groups:
            if len(group) >= min_votes:
                # Merge detections in this group
                merged = self._merge_detections(group)
                consensus_detections.append(merged)

        return consensus_detections

    def _weighted_voting_strategy(
        self,
        all_results: List[Tuple[BaseDetector, List[FaceDetectionResult], DetectionMetrics]]
    ) -> List[FaceDetectionResult]:
        """
        Weighted voting by confidence scores.

        Args:
            all_results: Results from all component detectors

        Returns:
            List of weighted consensus detections
        """
        # Similar to voting but weight by confidence
        all_detections = []
        for detector, detections, metrics in all_results:
            all_detections.extend(detections)

        if not all_detections:
            return []

        groups = self._group_similar_detections(all_detections)

        consensus_detections = []
        for group in groups:
            # Weight by confidence
            total_confidence = sum(d.confidence for d in group)
            threshold_confidence = self.confidence_threshold * len(self.component_detectors)

            if total_confidence >= threshold_confidence:
                merged = self._merge_detections(group)
                consensus_detections.append(merged)

        return consensus_detections

    def _union_strategy(
        self,
        all_results: List[Tuple[BaseDetector, List[FaceDetectionResult], DetectionMetrics]]
    ) -> List[FaceDetectionResult]:
        """
        Union strategy - keep all unique detections.

        Args:
            all_results: Results from all component detectors

        Returns:
            List of all unique detections
        """
        all_detections = []
        for detector, detections, metrics in all_results:
            all_detections.extend(detections)

        if not all_detections:
            return []

        # Group and merge similar detections
        groups = self._group_similar_detections(all_detections)

        return [self._merge_detections(group) for group in groups]

    def _intersection_strategy(
        self,
        all_results: List[Tuple[BaseDetector, List[FaceDetectionResult], DetectionMetrics]]
    ) -> List[FaceDetectionResult]:
        """
        Intersection strategy - only keep detections found by all detectors.

        Args:
            all_results: Results from all component detectors

        Returns:
            List of detections found by all detectors
        """
        all_detections = []
        for detector, detections, metrics in all_results:
            all_detections.extend(detections)

        if not all_detections:
            return []

        groups = self._group_similar_detections(all_detections)

        # Only keep groups where all detectors agree
        consensus_detections = []
        for group in groups:
            if len(group) == len(self.component_detectors):
                merged = self._merge_detections(group)
                consensus_detections.append(merged)

        return consensus_detections

    def _best_confidence_strategy(
        self,
        all_results: List[Tuple[BaseDetector, List[FaceDetectionResult], DetectionMetrics]]
    ) -> List[FaceDetectionResult]:
        """
        Best confidence strategy - use results from detector with highest confidence.

        Args:
            all_results: Results from all component detectors

        Returns:
            List of detections from best detector
        """
        if not all_results:
            return []

        # Find detector with highest average confidence
        best_results = None
        best_confidence = 0.0

        for detector, detections, metrics in all_results:
            if detections:
                avg_confidence = sum(d.confidence for d in detections) / len(detections)
                if avg_confidence > best_confidence:
                    best_confidence = avg_confidence
                    best_results = detections

        return best_results or []

    def _group_similar_detections(
        self,
        detections: List[FaceDetectionResult]
    ) -> List[List[FaceDetectionResult]]:
        """
        Group similar detections based on IoU (Intersection over Union).

        Args:
            detections: List of face detections

        Returns:
            List of detection groups
        """
        if not detections:
            return []

        groups: List[List[FaceDetectionResult]] = []
        used = set()

        for i, det1 in enumerate(detections):
            if i in used:
                continue

            group = [det1]
            used.add(i)

            for j, det2 in enumerate(detections):
                if j <= i or j in used:
                    continue

                iou = self._calculate_iou(det1.bounding_box, det2.bounding_box)

                if iou >= self.iou_threshold:
                    group.append(det2)
                    used.add(j)

            groups.append(group)

        return groups

    def _calculate_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.

        Args:
            box1: First bounding box (top, right, bottom, left)
            box2: Second bounding box (top, right, bottom, left)

        Returns:
            IoU value between 0 and 1
        """
        top1, right1, bottom1, left1 = box1
        top2, right2, bottom2, left2 = box2

        # Calculate intersection
        inter_left = max(left1, left2)
        inter_top = max(top1, top2)
        inter_right = min(right1, right2)
        inter_bottom = min(bottom1, bottom2)

        if inter_right < inter_left or inter_bottom < inter_top:
            return 0.0

        inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)

        # Calculate union
        box1_area = (right1 - left1) * (bottom1 - top1)
        box2_area = (right2 - left2) * (bottom2 - top2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _merge_detections(
        self,
        detections: List[FaceDetectionResult]
    ) -> FaceDetectionResult:
        """
        Merge multiple detections into a single detection.

        Args:
            detections: List of similar detections to merge

        Returns:
            Merged detection result
        """
        if len(detections) == 1:
            return detections[0]

        # Average bounding boxes
        tops, rights, bottoms, lefts = [], [], [], []
        confidences = []

        for det in detections:
            top, right, bottom, left = det.bounding_box
            tops.append(top)
            rights.append(right)
            bottoms.append(bottom)
            lefts.append(left)
            confidences.append(det.confidence)

        merged_box = (
            int(np.mean(tops)),
            int(np.mean(rights)),
            int(np.mean(bottoms)),
            int(np.mean(lefts))
        )

        # Average confidence
        merged_confidence = float(np.mean(confidences))

        return FaceDetectionResult(
            bounding_box=merged_box,
            confidence=merged_confidence,
            landmarks=None,
            quality_score=0.0
        )

    def cleanup(self) -> None:
        """Cleanup all component detectors."""
        for detector in self.component_detectors:
            try:
                detector.cleanup()
            except Exception as e:
                logger.warning(f"Cleanup failed for component detector: {e}")

        logger.info("Ensemble detector cleaned up")
