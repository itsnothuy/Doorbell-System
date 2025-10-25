#!/usr/bin/env python3
"""
Test suite for Detector Factory

Tests for the detector factory pattern implementation and detector selection logic.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.detectors.detector_factory import (
    DetectorFactory,
    create_detector,
    CPUDetector,
    GPUDetector,
    EdgeTPUDetector,
    MockDetector
)
from src.detectors.base_detector import DetectorType


class TestDetectorFactory(unittest.TestCase):
    """Test suite for detector factory."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'model': 'hog',
            'confidence_threshold': 0.5,
            'min_face_size': (30, 30)
        }
    
    def test_list_detectors(self):
        """Test listing all registered detectors."""
        detectors = DetectorFactory.list_detectors()
        
        # Should include all registered detector types
        self.assertIn('cpu', detectors)
        self.assertIn('gpu', detectors)
        self.assertIn('edgetpu', detectors)
        self.assertIn('mock', detectors)
        
        # Values should be boolean availability status
        for detector_type, available in detectors.items():
            self.assertIsInstance(available, bool)
    
    def test_get_available_detectors(self):
        """Test getting list of available detectors."""
        available = DetectorFactory.get_available_detectors()
        
        # Should be a list
        self.assertIsInstance(available, list)
        
        # Mock detector should always be available
        self.assertIn('mock', available)
        
        # Each detector should be available according to is_available()
        for detector_type in available:
            detector_class = DetectorFactory.get_detector_class(detector_type)
            self.assertTrue(detector_class.is_available())
    
    def test_get_detector_class(self):
        """Test getting detector class by type."""
        # Test valid detector types
        cpu_class = DetectorFactory.get_detector_class('cpu')
        self.assertEqual(cpu_class, CPUDetector)
        
        mock_class = DetectorFactory.get_detector_class('mock')
        self.assertEqual(mock_class, MockDetector)
        
        # Test case insensitivity
        cpu_class2 = DetectorFactory.get_detector_class('CPU')
        self.assertEqual(cpu_class2, CPUDetector)
        
        # Test invalid detector type
        with self.assertRaises(ValueError):
            DetectorFactory.get_detector_class('invalid')
    
    def test_create_mock_detector(self):
        """Test creating mock detector."""
        detector = DetectorFactory.create('mock', self.config)
        
        # Should be MockDetector instance
        self.assertIsInstance(detector, MockDetector)
        self.assertEqual(detector.detector_type, DetectorType.MOCK)
    
    def test_create_with_invalid_type_falls_back_to_cpu(self):
        """Test that invalid detector type falls back to CPU."""
        with patch.object(CPUDetector, 'is_available', return_value=True):
            detector = DetectorFactory.create('invalid_detector', self.config)
            # Should fall back to CPU
            self.assertIsInstance(detector, (CPUDetector, MockDetector))
    
    def test_create_with_unavailable_detector_falls_back(self):
        """Test fallback when requested detector is not available."""
        # Mock GPU detector as unavailable
        with patch.object(GPUDetector, 'is_available', return_value=False):
            with patch.object(CPUDetector, 'is_available', return_value=True):
                detector = DetectorFactory.create('gpu', self.config)
                # Should fall back to CPU
                self.assertEqual(detector.detector_type, DetectorType.CPU)
    
    def test_auto_detect_best_detector(self):
        """Test automatic detection of best available detector."""
        best_detector = DetectorFactory.auto_detect_best_detector()
        
        # Should return a valid detector type
        self.assertIn(best_detector, ['cpu', 'gpu', 'edgetpu', 'mock'])
        
        # Should be an available detector
        detector_class = DetectorFactory.get_detector_class(best_detector)
        self.assertTrue(detector_class.is_available())
    
    def test_register_custom_detector(self):
        """Test registering a custom detector."""
        # Create a mock detector class
        class CustomDetector(MockDetector):
            @classmethod
            def is_available(cls):
                return True
        
        # Register custom detector
        DetectorFactory.register_detector('custom', CustomDetector)
        
        # Verify registration
        self.assertIn('custom', DetectorFactory._detectors)
        self.assertEqual(DetectorFactory.get_detector_class('custom'), CustomDetector)
        
        # Create instance
        detector = DetectorFactory.create('custom', self.config)
        self.assertIsInstance(detector, CustomDetector)
        
        # Cleanup
        del DetectorFactory._detectors['custom']
    
    def test_create_detector_convenience_function(self):
        """Test convenience function for creating detectors."""
        # Test with explicit type
        detector = create_detector('mock', self.config)
        self.assertIsInstance(detector, MockDetector)
        
        # Test with auto-detection
        detector2 = create_detector(None, self.config)
        self.assertIsNotNone(detector2)
        
        # Test with no config (uses defaults)
        detector3 = create_detector('mock', None)
        self.assertIsInstance(detector3, MockDetector)
    
    def test_detector_priority_order(self):
        """Test that auto-detect follows correct priority order."""
        # EdgeTPU > GPU > CPU > Mock
        
        # Mock all detectors as available
        with patch.object(EdgeTPUDetector, 'is_available', return_value=True):
            best = DetectorFactory.auto_detect_best_detector()
            self.assertEqual(best, 'edgetpu')
        
        # Mock GPU as best available
        with patch.object(EdgeTPUDetector, 'is_available', return_value=False):
            with patch.object(GPUDetector, 'is_available', return_value=True):
                best = DetectorFactory.auto_detect_best_detector()
                self.assertEqual(best, 'gpu')
        
        # CPU should be selected when EdgeTPU and GPU unavailable
        with patch.object(EdgeTPUDetector, 'is_available', return_value=False):
            with patch.object(GPUDetector, 'is_available', return_value=False):
                with patch.object(CPUDetector, 'is_available', return_value=True):
                    best = DetectorFactory.auto_detect_best_detector()
                    self.assertEqual(best, 'cpu')


class TestMockDetector(unittest.TestCase):
    """Test suite for mock detector."""
    
    def test_mock_detector_always_available(self):
        """Test that mock detector is always available."""
        self.assertTrue(MockDetector.is_available())
    
    def test_mock_detector_returns_empty_results(self):
        """Test that mock detector returns empty results."""
        import numpy as np
        
        detector = MockDetector({})
        
        # Create test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Run detection
        faces, metrics = detector.detect_faces(test_image)
        
        # Should return empty list
        self.assertEqual(len(faces), 0)
        self.assertIsNotNone(metrics)
    
    def test_mock_detector_health_check(self):
        """Test mock detector health check."""
        detector = MockDetector({})
        health = detector.health_check()
        
        self.assertEqual(health['status'], 'healthy')
        self.assertEqual(health['detector_type'], 'mock')


if __name__ == '__main__':
    unittest.main()
