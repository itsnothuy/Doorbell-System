#!/usr/bin/env python3
"""
Detector Factory - Strategy Pattern Implementation

Factory for creating face detector instances with automatic hardware detection
and fallback strategies. Enhanced with ensemble support and detector pooling.
"""

import logging
from typing import Dict, Any, Type, Optional, List

from src.detectors.base_detector import BaseDetector, DetectorType
from src.detectors.cpu_detector import CPUDetector

logger = logging.getLogger(__name__)


# Mock detectors for GPU and EdgeTPU (not yet implemented)
class GPUDetector(BaseDetector):
    """Placeholder GPU detector (to be implemented in future PR)."""
    
    @classmethod
    def is_available(cls) -> bool:
        return False
    
    def _get_detector_type(self) -> DetectorType:
        return DetectorType.GPU
    
    def _initialize_model(self) -> None:
        raise NotImplementedError("GPU detector not yet implemented")
    
    def _run_inference(self, image):
        raise NotImplementedError("GPU detector not yet implemented")


class EdgeTPUDetector(BaseDetector):
    """Placeholder EdgeTPU detector (to be implemented in future PR)."""
    
    @classmethod
    def is_available(cls) -> bool:
        return False
    
    def _get_detector_type(self) -> DetectorType:
        return DetectorType.EDGETPU
    
    def _initialize_model(self) -> None:
        raise NotImplementedError("EdgeTPU detector not yet implemented")
    
    def _run_inference(self, image):
        raise NotImplementedError("EdgeTPU detector not yet implemented")


class MockDetector(BaseDetector):
    """Mock detector for testing purposes."""
    
    @classmethod
    def is_available(cls) -> bool:
        return True
    
    def _get_detector_type(self) -> DetectorType:
        return DetectorType.MOCK
    
    def _initialize_model(self) -> None:
        logger.info("Mock detector initialized")
    
    def _run_inference(self, image):
        """Return empty list for mock detector."""
        return []


class DetectorFactory:
    """
    Factory for creating face detector instances.
    
    Implements automatic hardware detection and fallback strategies
    following the Frigate detector plugin architecture.
    """
    
    # Registry of available detector implementations
    _detectors: Dict[str, Type[BaseDetector]] = {
        'cpu': CPUDetector,
        'gpu': GPUDetector,
        'edgetpu': EdgeTPUDetector,
        'mock': MockDetector
    }
    
    @classmethod
    def create(cls, detector_type: str, config: Dict[str, Any]) -> BaseDetector:
        """
        Create a detector instance with hardware validation and fallback.
        
        Args:
            detector_type: Type of detector to create ('cpu', 'gpu', 'edgetpu', 'mock')
            config: Detector configuration
            
        Returns:
            BaseDetector instance
            
        Raises:
            ValueError: If detector type is invalid or no detector is available
        """
        logger.info(f"Creating detector: {detector_type}")
        
        # Validate detector type
        detector_type = detector_type.lower()
        if detector_type not in cls._detectors:
            logger.warning(f"Unknown detector type: {detector_type}, falling back to CPU")
            detector_type = 'cpu'
        
        # Get detector class
        detector_class = cls._detectors[detector_type]
        
        # Check if detector is available
        if not detector_class.is_available():
            logger.warning(f"{detector_type} detector not available, falling back to CPU")
            detector_class = cls._detectors['cpu']
            
            # If CPU also not available, raise error
            if not detector_class.is_available():
                raise ValueError("No face detector available - face_recognition library not installed")
        
        # Create and return detector instance
        try:
            detector = detector_class(config)
            logger.info(f"Created {detector.detector_type.value} detector successfully")
            return detector
        except Exception as e:
            logger.error(f"Failed to create detector: {e}")
            raise
    
    @classmethod
    def get_detector_class(cls, detector_type: str) -> Type[BaseDetector]:
        """
        Get the detector class for a given type without instantiation.
        
        Args:
            detector_type: Type of detector
            
        Returns:
            BaseDetector class
            
        Raises:
            ValueError: If detector type is unknown
        """
        detector_type = detector_type.lower()
        if detector_type not in cls._detectors:
            raise ValueError(f"Unknown detector type: {detector_type}")
        
        return cls._detectors[detector_type]
    
    @classmethod
    def register_detector(cls, detector_type: str, detector_class: Type[BaseDetector]) -> None:
        """
        Register a new detector implementation.
        
        Args:
            detector_type: Unique identifier for the detector
            detector_class: BaseDetector subclass
        """
        if detector_type in cls._detectors:
            logger.warning(f"Overwriting existing detector: {detector_type}")
        
        cls._detectors[detector_type] = detector_class
        logger.info(f"Registered detector: {detector_type}")
    
    @classmethod
    def list_detectors(cls) -> Dict[str, bool]:
        """
        List all registered detectors and their availability.
        
        Returns:
            Dictionary mapping detector types to availability status
        """
        return {
            detector_type: detector_class.is_available()
            for detector_type, detector_class in cls._detectors.items()
        }
    
    @classmethod
    def get_available_detectors(cls) -> list:
        """
        Get list of available detector types.
        
        Returns:
            List of detector type strings
        """
        return [
            detector_type
            for detector_type, detector_class in cls._detectors.items()
            if detector_class.is_available()
        ]
    
    @classmethod
    def auto_detect_best_detector(cls) -> str:
        """
        Automatically detect the best available detector for the system.
        
        Returns:
            Name of the best available detector
        """
        # Priority order: EdgeTPU > GPU > CPU > Mock
        priority_order = ['edgetpu', 'gpu', 'cpu', 'mock']
        
        for detector_type in priority_order:
            if detector_type in cls._detectors:
                detector_class = cls._detectors[detector_type]
                if detector_class.is_available():
                    logger.info(f"Auto-detected best detector: {detector_type}")
                    return detector_type
        
        # Fallback to CPU if nothing else available
        logger.warning("No optimal detector found, defaulting to CPU")
        return 'cpu'
    
    @classmethod
    def create_detector_ensemble(
        cls,
        detector_configs: List[Dict[str, Any]],
        ensemble_strategy: str = "voting",
        ensemble_config: Optional[Dict[str, Any]] = None
    ) -> 'BaseDetector':
        """
        Create an ensemble of detectors with specified strategy.
        
        Args:
            detector_configs: List of detector configurations, each with 'type' key
            ensemble_strategy: Strategy for combining results ('voting', 'union', etc.)
            ensemble_config: Additional ensemble configuration
            
        Returns:
            EnsembleDetector instance
            
        Example:
            >>> ensemble = DetectorFactory.create_detector_ensemble([
            ...     {'type': 'cpu', 'model': 'hog'},
            ...     {'type': 'cpu', 'model': 'cnn'}
            ... ], strategy='voting')
        """
        from src.detectors.ensemble_detector import EnsembleDetector, EnsembleStrategy
        
        if not detector_configs:
            raise ValueError("At least one detector configuration required")
        
        logger.info(f"Creating ensemble with {len(detector_configs)} detectors")
        
        # Create component detectors
        detectors = []
        for i, config in enumerate(detector_configs):
            detector_type = config.get('type', 'cpu')
            try:
                detector = cls.create(detector_type, config)
                detectors.append(detector)
            except Exception as e:
                logger.warning(f"Failed to create detector {i} (type={detector_type}): {e}")
        
        if not detectors:
            raise ValueError("Failed to create any component detectors")
        
        # Parse strategy
        try:
            strategy = EnsembleStrategy(ensemble_strategy.lower())
        except ValueError:
            logger.warning(f"Unknown strategy '{ensemble_strategy}', using 'voting'")
            strategy = EnsembleStrategy.VOTING
        
        # Create ensemble
        ensemble = EnsembleDetector(
            detectors=detectors,
            strategy=strategy,
            config=ensemble_config or {}
        )
        
        logger.info(f"Created ensemble detector with {len(detectors)} components")
        return ensemble
    
    @classmethod
    def create_detector_pool(
        cls,
        detector_type: str,
        pool_size: int,
        config: Dict[str, Any]
    ) -> List[BaseDetector]:
        """
        Create a pool of detector instances for parallel processing.
        
        Args:
            detector_type: Type of detector to create
            pool_size: Number of detector instances in pool
            config: Base configuration for all detectors
            
        Returns:
            List of detector instances
            
        Example:
            >>> pool = DetectorFactory.create_detector_pool('cpu', 4, {'model': 'hog'})
        """
        if pool_size < 1:
            raise ValueError(f"Pool size must be >= 1, got {pool_size}")
        
        logger.info(f"Creating detector pool: {detector_type} x {pool_size}")
        
        pool = []
        for i in range(pool_size):
            # Create a copy of config for each instance
            instance_config = config.copy()
            instance_config['instance_id'] = i
            instance_config['pool_size'] = pool_size
            
            try:
                detector = cls.create(detector_type, instance_config)
                pool.append(detector)
            except Exception as e:
                logger.error(f"Failed to create detector {i} in pool: {e}")
                # Clean up created detectors
                for detector in pool:
                    try:
                        detector.cleanup()
                    except:
                        pass
                raise
        
        logger.info(f"Created detector pool with {len(pool)} instances")
        return pool


def create_detector(detector_type: Optional[str] = None, 
                   config: Optional[Dict[str, Any]] = None) -> BaseDetector:
    """
    Convenience function to create a detector instance.
    
    Args:
        detector_type: Type of detector (auto-detected if None)
        config: Detector configuration (uses defaults if None)
        
    Returns:
        BaseDetector instance
    """
    if detector_type is None:
        detector_type = DetectorFactory.auto_detect_best_detector()
    
    if config is None:
        config = {}
    
    return DetectorFactory.create(detector_type, config)
