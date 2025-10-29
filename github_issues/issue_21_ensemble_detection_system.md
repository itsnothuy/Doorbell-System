# Issue #21: Ensemble Detection System Implementation

## Issue Summary

**Priority**: Medium  
**Type**: AI/ML Enhancement, Detection Accuracy  
**Component**: Detection Pipeline, AI Strategy  
**Estimated Effort**: 25-35 hours  
**Dependencies**: Base Detector Framework, Multiple Detection Backends  

## Overview

Implement intelligent ensemble detection system that combines multiple face detection models (CPU, GPU, EdgeTPU) with sophisticated fusion algorithms to maximize detection accuracy, minimize false positives, and provide adaptive performance optimization based on real-time conditions.

## Current State Analysis

### Existing Ensemble Placeholder
```python
# Current incomplete implementation in src/detectors/ensemble_detector.py

class EnsembleDetector(BaseDetector):
    """Ensemble face detector combining multiple models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.detectors = []
        self.fusion_strategy = config.get('fusion_strategy', 'voting')
        
        # TODO: Initialize multiple detector backends
        logger.info("Ensemble detector created")
    
    def add_detector(self, detector: BaseDetector) -> None:
        """Add a detector to the ensemble."""
        # TODO: Implement detector management
        logger.warning("Ensemble detector management not implemented")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using ensemble of detectors."""
        # TODO: Implement ensemble detection and fusion
        logger.warning("Ensemble detection not implemented")
        return []
    
    def _fuse_detections(self, all_detections: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Fuse detections from multiple detectors."""
        # TODO: Implement detection fusion algorithms
        logger.warning("Detection fusion not implemented")
        return []
```

### Missing Capabilities
- **Multi-Model Integration**: No support for combining different detectors
- **Fusion Algorithms**: No intelligent result combination strategies
- **Adaptive Selection**: No dynamic detector selection based on conditions
- **Performance Optimization**: No latency vs accuracy balancing
- **Confidence Calibration**: No unified confidence scoring

## Technical Specifications

### Comprehensive Ensemble Detection Framework

#### Production Ensemble System Implementation
```python
#!/usr/bin/env python3
"""
Production Ensemble Face Detection System

Combines multiple detection models with intelligent fusion algorithms
to maximize accuracy while optimizing performance for real-time applications.
"""

import cv2
import numpy as np
import logging
import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import statistics
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from src.detectors.base_detector import BaseDetector

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
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
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
            'bbox': [self.bbox[0], self.bbox[1], 
                    self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]],  # Convert to (x, y, w, h)
            'confidence': self.confidence,
            'detector_votes': self.detector_votes,
            'individual_confidences': self.individual_confidences,
            'fusion_score': self.fusion_score,
            'consensus_level': self.consensus_level,
            'detection_source': self.detection_source,
            'processing_time': self.processing_time
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
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def simple_voting_fusion(self, detector_results: Dict[str, List[Dict[str, Any]]], 
                           detector_configs: Dict[str, DetectorConfig]) -> List[EnsembleDetection]:
        """Simple majority voting fusion."""
        all_detections = []
        
        # Collect all detections
        for detector_name, detections in detector_results.items():
            for detection in detections:
                bbox = detection['bbox']
                # Convert (x, y, w, h) to (x1, y1, x2, y2)
                x, y, w, h = bbox
                bbox_corners = (x, y, x + w, y + h)
                
                ensemble_detection = EnsembleDetection(
                    bbox=bbox_corners,
                    confidence=detection['confidence'],
                    detector_votes=[detector_name],
                    individual_confidences={detector_name: detection['confidence']}
                )
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
    
    def weighted_voting_fusion(self, detector_results: Dict[str, List[Dict[str, Any]]], 
                             detector_configs: Dict[str, DetectorConfig]) -> List[EnsembleDetection]:
        """Weighted voting fusion based on detector weights."""
        all_detections = []
        
        # Collect all detections with weights
        for detector_name, detections in detector_results.items():
            weight = detector_configs[detector_name].weight
            
            for detection in detections:
                bbox = detection['bbox']
                x, y, w, h = bbox
                bbox_corners = (x, y, x + w, y + h)
                
                # Apply weight to confidence
                weighted_confidence = detection['confidence'] * weight
                
                ensemble_detection = EnsembleDetection(
                    bbox=bbox_corners,
                    confidence=weighted_confidence,
                    detector_votes=[detector_name],
                    individual_confidences={detector_name: detection['confidence']},
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
    
    def consensus_fusion(self, detector_results: Dict[str, List[Dict[str, Any]]], 
                        detector_configs: Dict[str, DetectorConfig]) -> List[EnsembleDetection]:
        """Consensus-based fusion requiring agreement from multiple detectors."""
        all_detections = []
        
        # Collect all detections
        for detector_name, detections in detector_results.items():
            for detection in detections:
                bbox = detection['bbox']
                x, y, w, h = bbox
                bbox_corners = (x, y, x + w, y + h)
                
                ensemble_detection = EnsembleDetection(
                    bbox=bbox_corners,
                    confidence=detection['confidence'],
                    detector_votes=[detector_name],
                    individual_confidences={detector_name: detection['confidence']}
                )
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
    
    def confidence_weighted_fusion(self, detector_results: Dict[str, List[Dict[str, Any]]], 
                                 detector_configs: Dict[str, DetectorConfig]) -> List[EnsembleDetection]:
        """Confidence-weighted fusion emphasizing high-confidence detections."""
        all_detections = []
        
        # Collect all detections
        for detector_name, detections in detector_results.items():
            for detection in detections:
                bbox = detection['bbox']
                x, y, w, h = bbox
                bbox_corners = (x, y, x + w, y + h)
                
                ensemble_detection = EnsembleDetection(
                    bbox=bbox_corners,
                    confidence=detection['confidence'],
                    detector_votes=[detector_name],
                    individual_confidences={detector_name: detection['confidence']}
                )
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
    
    def nms_fusion(self, detector_results: Dict[str, List[Dict[str, Any]]], 
                  detector_configs: Dict[str, DetectorConfig]) -> List[EnsembleDetection]:
        """Non-Maximum Suppression fusion to remove overlapping detections."""
        all_detections = []
        
        # Collect all detections
        for detector_name, detections in detector_results.items():
            weight = detector_configs[detector_name].weight
            
            for detection in detections:
                bbox = detection['bbox']
                x, y, w, h = bbox
                bbox_corners = (x, y, x + w, y + h)
                
                ensemble_detection = EnsembleDetection(
                    bbox=bbox_corners,
                    confidence=detection['confidence'] * weight,
                    detector_votes=[detector_name],
                    individual_confidences={detector_name: detection['confidence']},
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
                    current_detection.individual_confidences.update(detection.individual_confidences)
            
            all_detections = remaining_detections
        
        return final_detections
    
    def _group_similar_detections(self, detections: List[EnsembleDetection]) -> List[List[EnsembleDetection]]:
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
            for j, other_detection in enumerate(detections[i+1:], i+1):
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
        
        avg_x1 = sum(d.bbox[0] * d.confidence for d in group) / total_confidence
        avg_y1 = sum(d.bbox[1] * d.confidence for d in group) / total_confidence
        avg_x2 = sum(d.bbox[2] * d.confidence for d in group) / total_confidence
        avg_y2 = sum(d.bbox[3] * d.confidence for d in group) / total_confidence
        
        merged_bbox = (int(avg_x1), int(avg_y1), int(avg_x2), int(avg_y2))
        
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
    """Production ensemble face detector with advanced fusion strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Configuration
        self.fusion_strategy = FusionStrategy(config.get('fusion_strategy', 'weighted_voting'))
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
        
        logger.info(f"Ensemble detector created with strategy: {self.fusion_strategy.value}")
    
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
        logger.info(f"Added detector '{name}' to ensemble (weight: {weight}, priority: {priority.name})")
    
    def remove_detector(self, name: str) -> bool:
        """Remove a detector from the ensemble."""
        if name in self.detector_configs:
            del self.detector_configs[name]
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
                    success = config.detector_instance.load_model()
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
            enabled_detectors = [name for name, config in self.detector_configs.items() if config.enabled]
            
            if not enabled_detectors:
                logger.error("No detectors successfully initialized")
                return False
            
            self.is_initialized = True
            logger.info(f"Ensemble detector initialized with {len(enabled_detectors)} active detectors")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ensemble detector: {e}")
            return False
    
    def detect_faces(self, image: np.ndarray, 
                    performance_requirements: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Detect faces using ensemble of detectors."""
        if not self.is_initialized:
            if not self.load_model():
                return []
        
        start_time = time.time()
        
        try:
            # Select optimal detectors based on requirements
            if performance_requirements:
                selected_detector_names = self.adaptive_selector.select_optimal_detectors(
                    self.detector_configs, performance_requirements
                )
            else:
                selected_detector_names = [name for name, config in self.detector_configs.items() if config.enabled]
            
            if not selected_detector_names:
                logger.warning("No detectors selected for ensemble detection")
                return []
            
            logger.debug(f"Running ensemble detection with {len(selected_detector_names)} detectors")
            
            # Run detection on selected detectors
            detector_results = {}
            
            if self.parallel_execution and len(selected_detector_names) > 1:
                # Parallel execution
                detector_results = self._run_parallel_detection(image, selected_detector_names)
            else:
                # Sequential execution
                detector_results = self._run_sequential_detection(image, selected_detector_names)
            
            # Fuse detection results
            fusion_start = time.time()
            ensemble_detections = self._fuse_detections(detector_results)
            fusion_time = time.time() - fusion_start
            
            # Convert to expected format
            final_results = []
            for detection in ensemble_detections:
                result_dict = detection.to_dict()
                result_dict['detector'] = 'ensemble'
                final_results.append(result_dict)
            
            # Update metrics
            total_time = time.time() - start_time
            self._update_metrics(detector_results, fusion_time, total_time)
            
            logger.debug(f"Ensemble detection completed in {total_time*1000:.2f}ms, found {len(final_results)} faces")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Ensemble detection failed: {e}")
            return []
    
    def _run_parallel_detection(self, image: np.ndarray, 
                              detector_names: List[str]) -> Dict[str, List[Dict[str, Any]]]:
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
                    logger.debug(f"Detector {detector_name} completed with {len(result)} detections")
                else:
                    logger.warning(f"Detector {detector_name} returned None")
                    
            except Exception as e:
                logger.error(f"Detector {detector_name} failed: {e}")
        
        return detector_results
    
    def _run_sequential_detection(self, image: np.ndarray, 
                                detector_names: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Run detection on detectors sequentially."""
        detector_results = {}
        
        for detector_name in detector_names:
            config = self.detector_configs[detector_name]
            if config.enabled:
                try:
                    result = self._safe_detect(detector_name, config.detector_instance, image, config.timeout)
                    if result is not None:
                        detector_results[detector_name] = result
                        logger.debug(f"Detector {detector_name} completed with {len(result)} detections")
                        
                except Exception as e:
                    logger.error(f"Detector {detector_name} failed: {e}")
        
        return detector_results
    
    def _safe_detect(self, detector_name: str, detector: BaseDetector, 
                    image: np.ndarray, timeout: float) -> Optional[List[Dict[str, Any]]]:
        """Safely run detection with timeout handling."""
        try:
            start_time = time.time()
            
            # Run detection
            result = detector.detect_faces(image)
            
            detection_time = time.time() - start_time
            
            # Update performance history
            if result:
                # Estimate accuracy based on detection count (simplified)
                estimated_accuracy = min(1.0, len(result) / 3.0)  # Assume 3 faces is "good"
                self.adaptive_selector.update_performance_history(
                    detector_name, detection_time, estimated_accuracy
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Detection error in {detector_name}: {e}")
            return None
    
    def _fuse_detections(self, detector_results: Dict[str, List[Dict[str, Any]]]) -> List[EnsembleDetection]:
        """Fuse detection results using configured strategy."""
        if not detector_results:
            return []
        
        # Select fusion strategy
        fusion_method = {
            FusionStrategy.SIMPLE_VOTING: self.fusion_engine.simple_voting_fusion,
            FusionStrategy.WEIGHTED_VOTING: self.fusion_engine.weighted_voting_fusion,
            FusionStrategy.CONSENSUS: self.fusion_engine.consensus_fusion,
            FusionStrategy.CONFIDENCE_WEIGHTED: self.fusion_engine.confidence_weighted_fusion,
            FusionStrategy.NMS_FUSION: self.fusion_engine.nms_fusion
        }.get(self.fusion_strategy, self.fusion_engine.weighted_voting_fusion)
        
        try:
            return fusion_method(detector_results, self.detector_configs)
        except Exception as e:
            logger.error(f"Fusion failed with {self.fusion_strategy.value}: {e}")
            # Fallback to simple concatenation
            all_detections = []
            for detector_name, detections in detector_results.items():
                for detection in detections:
                    bbox = detection['bbox']
                    x, y, w, h = bbox
                    bbox_corners = (x, y, x + w, y + h)
                    
                    ensemble_detection = EnsembleDetection(
                        bbox=bbox_corners,
                        confidence=detection['confidence'],
                        detector_votes=[detector_name],
                        individual_confidences={detector_name: detection['confidence']}
                    )
                    all_detections.append(ensemble_detection)
            
            return all_detections
    
    def _update_metrics(self, detector_results: Dict[str, List[Dict[str, Any]]], 
                       fusion_time: float, total_time: float) -> None:
        """Update ensemble performance metrics."""
        self.ensemble_metrics['total_detections'] += 1
        
        if detector_results:
            self.ensemble_metrics['successful_detections'] += 1
        
        self.ensemble_metrics['fusion_times'].append(fusion_time)
        
        # Keep only recent metrics
        if len(self.ensemble_metrics['fusion_times']) > 1000:
            self.ensemble_metrics['fusion_times'] = self.ensemble_metrics['fusion_times'][-1000:]
        
        # Update detector-specific metrics
        for detector_name, results in detector_results.items():
            if detector_name not in self.ensemble_metrics['detector_performance']:
                self.ensemble_metrics['detector_performance'][detector_name] = {
                    'invocations': 0,
                    'successful_invocations': 0,
                    'total_detections': 0
                }
            
            metrics = self.ensemble_metrics['detector_performance'][detector_name]
            metrics['invocations'] += 1
            
            if results:
                metrics['successful_invocations'] += 1
                metrics['total_detections'] += len(results)
    
    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get comprehensive ensemble performance statistics."""
        stats = {
            'fusion_strategy': self.fusion_strategy.value,
            'active_detectors': len([name for name, config in self.detector_configs.items() if config.enabled]),
            'total_detectors': len(self.detector_configs),
            'parallel_execution': self.parallel_execution,
            'ensemble_metrics': self.ensemble_metrics.copy()
        }
        
        # Calculate fusion statistics
        if self.ensemble_metrics['fusion_times']:
            fusion_times = self.ensemble_metrics['fusion_times']
            stats['fusion_statistics'] = {
                'mean_fusion_time': statistics.mean(fusion_times),
                'median_fusion_time': statistics.median(fusion_times),
                'min_fusion_time': min(fusion_times),
                'max_fusion_time': max(fusion_times)
            }
        
        # Add detector-specific statistics
        detector_stats = {}
        for detector_name, config in self.detector_configs.items():
            perf_metrics = self.ensemble_metrics['detector_performance'].get(detector_name, {})
            
            detector_stats[detector_name] = {
                'enabled': config.enabled,
                'weight': config.weight,
                'priority': config.priority.name,
                'timeout': config.timeout,
                'performance': perf_metrics
            }
        
        stats['detector_statistics'] = detector_stats
        
        return stats
    
    def is_available(self) -> bool:
        """Check if ensemble detector is available."""
        return any(config.enabled for config in self.detector_configs.values())
    
    def set_fusion_strategy(self, strategy: FusionStrategy) -> None:
        """Change fusion strategy at runtime."""
        self.fusion_strategy = strategy
        logger.info(f"Changed fusion strategy to: {strategy.value}")
    
    def enable_detector(self, detector_name: str) -> bool:
        """Enable a specific detector in the ensemble."""
        if detector_name in self.detector_configs:
            self.detector_configs[detector_name].enabled = True
            logger.info(f"Enabled detector: {detector_name}")
            return True
        return False
    
    def disable_detector(self, detector_name: str) -> bool:
        """Disable a specific detector in the ensemble."""
        if detector_name in self.detector_configs:
            self.detector_configs[detector_name].enabled = False
            logger.info(f"Disabled detector: {detector_name}")
            return True
        return False
    
    def cleanup(self) -> None:
        """Cleanup resources used by ensemble detector."""
        if self.executor:
            self.executor.shutdown(wait=True)
            logger.info("Ensemble detector cleaned up")
```

## Implementation Plan

### Phase 1: Core Ensemble Framework (Week 1-2)
1. **Base Infrastructure**
   - [ ] Implement `EnsembleDetection` data structure
   - [ ] Create `DetectorConfig` management
   - [ ] Build detector registration system
   - [ ] Add ensemble initialization logic

2. **Basic Fusion Algorithms**
   - [ ] Implement simple voting fusion
   - [ ] Create weighted voting fusion
   - [ ] Add IoU calculation utilities
   - [ ] Build detection grouping logic

### Phase 2: Advanced Fusion Strategies (Week 2-3)
1. **Sophisticated Fusion Methods**
   - [ ] Implement consensus-based fusion
   - [ ] Create confidence-weighted fusion
   - [ ] Add NMS-based fusion
   - [ ] Build adaptive fusion selection

2. **Performance Optimization**
   - [ ] Implement parallel detector execution
   - [ ] Add timeout handling for detectors
   - [ ] Create performance monitoring
   - [ ] Build metrics collection system

### Phase 3: Adaptive Selection (Week 3-4)
1. **Dynamic Detector Selection**
   - [ ] Implement adaptive selector
   - [ ] Create performance history tracking
   - [ ] Add requirement-based selection
   - [ ] Build optimization algorithms

2. **Real-time Adaptation**
   - [ ] Create runtime strategy switching
   - [ ] Add detector enable/disable controls
   - [ ] Implement performance feedback loops
   - [ ] Build optimization heuristics

### Phase 4: Integration and Testing (Week 4-5)
1. **System Integration**
   - [ ] Integrate with existing detector framework
   - [ ] Add configuration management
   - [ ] Create comprehensive testing
   - [ ] Build performance benchmarks

2. **Production Optimization**
   - [ ] Optimize for different scenarios
   - [ ] Add error handling and recovery
   - [ ] Create documentation and examples
   - [ ] Validate performance improvements

## Testing Strategy

### Fusion Algorithm Testing
```python
def test_ensemble_fusion_strategies():
    """Test different fusion strategies."""
    # Create mock detector results
    detector_results = {
        'cpu_detector': [
            {'bbox': [100, 100, 50, 50], 'confidence': 0.8},
            {'bbox': [200, 150, 60, 60], 'confidence': 0.7}
        ],
        'gpu_detector': [
            {'bbox': [105, 105, 45, 45], 'confidence': 0.9},  # Similar to first CPU detection
            {'bbox': [300, 200, 55, 55], 'confidence': 0.6}
        ]
    }
    
    # Test weighted voting
    ensemble = EnsembleDetector({'fusion_strategy': 'weighted_voting'})
    ensemble.add_detector('cpu', MockDetector(), weight=1.0)
    ensemble.add_detector('gpu', MockDetector(), weight=1.5)
    
    fused_results = ensemble.fusion_engine.weighted_voting_fusion(
        detector_results, ensemble.detector_configs
    )
    
    # Should merge similar detections
    assert len(fused_results) <= 3
    assert any(d.confidence > 0.8 for d in fused_results)

def test_adaptive_detector_selection():
    """Test adaptive detector selection."""
    selector = AdaptiveEnsembleSelector({})
    
    detector_configs = {
        'fast_detector': DetectorConfig(
            detector_instance=MockDetector(),
            use_for_speed=True,
            priority=DetectorPriority.HIGH
        ),
        'accurate_detector': DetectorConfig(
            detector_instance=MockDetector(),
            use_for_accuracy=True,
            priority=DetectorPriority.HIGH
        )
    }
    
    # Test speed preference
    selected = selector.select_optimal_detectors(
        detector_configs, 
        {'prefer_speed': True}
    )
    assert 'fast_detector' in selected
    
    # Test accuracy preference
    selected = selector.select_optimal_detectors(
        detector_configs, 
        {'prefer_speed': False}
    )
    assert 'accurate_detector' in selected
```

### Performance Testing
```python
def test_ensemble_performance():
    """Test ensemble performance vs individual detectors."""
    # Create ensemble with multiple detectors
    ensemble = EnsembleDetector({
        'fusion_strategy': 'confidence_weighted',
        'parallel_execution': True
    })
    
    cpu_detector = CPUDetector()
    gpu_detector = GPUDetector()
    
    ensemble.add_detector('cpu', cpu_detector, weight=1.0)
    ensemble.add_detector('gpu', gpu_detector, weight=1.2)
    ensemble.load_model()
    
    # Test with challenging image
    test_image = cv2.imread('test_images/multiple_faces.jpg')
    
    # Compare results
    cpu_results = cpu_detector.detect_faces(test_image)
    gpu_results = gpu_detector.detect_faces(test_image)
    ensemble_results = ensemble.detect_faces(test_image)
    
    # Ensemble should find at least as many faces as best individual detector
    max_individual = max(len(cpu_results), len(gpu_results))
    assert len(ensemble_results) >= max_individual * 0.8  # Allow some tolerance
    
    # Check confidence quality
    if ensemble_results:
        avg_ensemble_conf = sum(r['confidence'] for r in ensemble_results) / len(ensemble_results)
        assert avg_ensemble_conf > 0.5
```

## Acceptance Criteria

### Fusion Algorithm Requirements
- [ ] All fusion strategies (voting, consensus, confidence-weighted, NMS) implemented
- [ ] IoU-based detection matching working correctly
- [ ] Confidence calibration and normalization functional
- [ ] Detection quality improved compared to individual detectors

### Performance Requirements
- [ ] Parallel detector execution reduces latency by 40%+
- [ ] Adaptive selection chooses optimal detectors for conditions
- [ ] Ensemble overhead adds <20% to fastest individual detector
- [ ] Memory usage scales linearly with number of detectors

### Integration Requirements
- [ ] Seamless integration with existing detector framework
- [ ] Runtime detector enable/disable functionality
- [ ] Configuration through standard config system
- [ ] Comprehensive metrics and monitoring

### Quality Requirements
- [ ] Detection accuracy improved by 10%+ over best individual detector
- [ ] False positive rate reduced by 15%+ through consensus filtering
- [ ] Robust operation with detector failures
- [ ] Production-ready error handling and recovery

This implementation provides intelligent ensemble detection that maximizes accuracy while maintaining real-time performance, adapting dynamically to changing conditions and requirements.