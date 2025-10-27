#!/usr/bin/env python3
"""
Ensemble Detector Implementation

Multi-detector ensemble with voting mechanisms and confidence aggregation.
Provides improved accuracy through detector diversity and fallback strategies.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass, field

import numpy as np

from src.detectors.base_detector import (
    BaseDetector,
    DetectorType,
    ModelType,
    FaceDetectionResult,
    DetectionMetrics
)

logger = logging.getLogger(__name__)


class EnsembleStrategy(Enum):
    """Ensemble voting strategies."""
    VOTING = "voting"                    # Majority voting
    WEIGHTED_VOTING = "weighted_voting"  # Weighted by confidence
    UNION = "union"                      # Union of all detections
    INTERSECTION = "intersection"         # Only common detections
    BEST_CONFIDENCE = "best_confidence"  # Highest confidence detector


@dataclass
class EnsembleMetadata:
    """Metadata about ensemble detection process."""
    
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


class EnsembleDetector(BaseDetector):
    """
    Ensemble detector combining multiple detection models.
    
    Uses various voting and aggregation strategies to improve
    detection accuracy and robustness through detector diversity.
    """
    
    def __init__(
        self,
        detectors: List[BaseDetector],
        strategy: EnsembleStrategy = EnsembleStrategy.VOTING,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ensemble detector.
        
        Args:
            detectors: List of component detector instances
            strategy: Ensemble strategy to use
            config: Optional configuration
        """
        if not detectors:
            raise ValueError("At least one detector required for ensemble")
        
        self.component_detectors = detectors
        self.ensemble_strategy = strategy
        
        # Initialize with config
        config = config or {}
        config['model_type'] = 'ensemble'
        
        # Store for later use
        self._config = config
        self._detectors = detectors
        self._strategy = strategy
        
        # Don't call super().__init__() yet - set required attributes first
        self.config = config
        self.detector_type = DetectorType.CPU  # Default type for ensemble
        self.model_type = ModelType.HOG  # Placeholder
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance tracking
        self.total_detections = 0
        self.total_inference_time = 0.0
        self.last_metrics: Optional[DetectionMetrics] = None
        
        # Detection parameters
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.min_face_size = config.get('min_face_size', (30, 30))
        self.max_face_size = config.get('max_face_size', (1000, 1000))
        
        # Ensemble-specific config
        self.min_agreement = config.get('min_agreement', 0.5)  # 50% of detectors
        self.iou_threshold = config.get('iou_threshold', 0.5)  # IoU for matching detections
        
        logger.info(
            f"Initialized ensemble detector with {len(detectors)} components "
            f"using {strategy.value} strategy"
        )
    
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
