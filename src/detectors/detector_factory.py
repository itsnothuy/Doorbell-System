#!/usr/bin/env python3
"""
Detector Factory - Strategy Pattern Implementation

Factory for creating face detector instances with automatic hardware detection
and fallback strategies.
"""

import logging
from typing import Dict, Any, Type, Optional

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
