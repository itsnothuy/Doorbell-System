#!/usr/bin/env python3
"""
Test suite for Face Recognition Result Data Structures

Tests for PersonMatch, RecognitionMetadata, and FaceRecognitionResult classes.
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.recognition.recognition_result import (
    PersonMatch,
    RecognitionMetadata,
    FaceRecognitionResult
)


class TestPersonMatch(unittest.TestCase):
    """Test suite for PersonMatch class."""
    
    def test_person_match_creation(self):
        """Test creating a PersonMatch instance."""
        match = PersonMatch(
            person_id="person_001",
            person_name="John Doe",
            confidence=0.85,
            similarity_score=0.92,
            match_count=5
        )
        
        self.assertEqual(match.person_id, "person_001")
        self.assertEqual(match.person_name, "John Doe")
        self.assertEqual(match.confidence, 0.85)
        self.assertEqual(match.similarity_score, 0.92)
        self.assertEqual(match.match_count, 5)
        self.assertFalse(match.is_blacklisted)
    
    def test_person_match_blacklisted(self):
        """Test creating a blacklisted PersonMatch."""
        match = PersonMatch(
            person_id="blacklist_001",
            person_name="Suspicious Person",
            confidence=0.95,
            is_blacklisted=True
        )
        
        self.assertTrue(match.is_blacklisted)
        self.assertEqual(match.person_id, "blacklist_001")
    
    def test_person_match_to_dict(self):
        """Test converting PersonMatch to dictionary."""
        match = PersonMatch(
            person_id="person_002",
            person_name="Jane Smith",
            confidence=0.78,
            similarity_score=0.82,
            metadata={'source': 'test'}
        )
        
        match_dict = match.to_dict()
        
        self.assertIsInstance(match_dict, dict)
        self.assertEqual(match_dict['person_id'], "person_002")
        self.assertEqual(match_dict['person_name'], "Jane Smith")
        self.assertEqual(match_dict['confidence'], 0.78)
        self.assertEqual(match_dict['metadata']['source'], 'test')
    
    def test_person_match_defaults(self):
        """Test PersonMatch with default values."""
        match = PersonMatch(person_id="person_003")
        
        self.assertIsNone(match.person_name)
        self.assertEqual(match.confidence, 0.0)
        self.assertEqual(match.similarity_score, 0.0)
        self.assertEqual(match.match_count, 1)
        self.assertFalse(match.is_blacklisted)


class TestRecognitionMetadata(unittest.TestCase):
    """Test suite for RecognitionMetadata class."""
    
    def test_metadata_creation(self):
        """Test creating RecognitionMetadata instance."""
        metadata = RecognitionMetadata(
            tolerance_used=0.6,
            cache_hit=True,
            processing_method="cached",
            encoding_time=0.05,
            matching_time=0.02,
            total_time=0.07
        )
        
        self.assertEqual(metadata.tolerance_used, 0.6)
        self.assertTrue(metadata.cache_hit)
        self.assertEqual(metadata.processing_method, "cached")
        self.assertEqual(metadata.encoding_time, 0.05)
        self.assertEqual(metadata.matching_time, 0.02)
        self.assertEqual(metadata.total_time, 0.07)
    
    def test_metadata_to_dict(self):
        """Test converting RecognitionMetadata to dictionary."""
        metadata = RecognitionMetadata(
            tolerance_used=0.5,
            cache_hit=False,
            database_query_count=3,
            faces_compared=25
        )
        
        metadata_dict = metadata.to_dict()
        
        self.assertIsInstance(metadata_dict, dict)
        self.assertEqual(metadata_dict['tolerance_used'], 0.5)
        self.assertFalse(metadata_dict['cache_hit'])
        self.assertEqual(metadata_dict['database_query_count'], 3)
        self.assertEqual(metadata_dict['faces_compared'], 25)
    
    def test_metadata_defaults(self):
        """Test RecognitionMetadata with default values."""
        metadata = RecognitionMetadata()
        
        self.assertEqual(metadata.tolerance_used, 0.6)
        self.assertFalse(metadata.cache_hit)
        self.assertEqual(metadata.processing_method, "database")
        self.assertEqual(metadata.encoding_time, 0.0)
        self.assertEqual(metadata.database_query_count, 0)


class TestFaceRecognitionResult(unittest.TestCase):
    """Test suite for FaceRecognitionResult class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock face detection
        self.mock_face_detection = Mock()
        self.mock_face_detection.to_dict = Mock(return_value={'test': 'detection'})
    
    def test_recognition_result_creation(self):
        """Test creating FaceRecognitionResult instance."""
        result = FaceRecognitionResult(
            face_detection=self.mock_face_detection,
            is_known=True,
            confidence=0.85
        )
        
        self.assertEqual(result.face_detection, self.mock_face_detection)
        self.assertTrue(result.is_known)
        self.assertFalse(result.is_blacklisted)
        self.assertEqual(result.confidence, 0.85)
        self.assertIsInstance(result.recognition_timestamp, float)
    
    def test_recognition_result_with_matches(self):
        """Test FaceRecognitionResult with person matches."""
        matches = [
            PersonMatch(person_id="person_001", person_name="John Doe", confidence=0.92),
            PersonMatch(person_id="person_002", person_name="Jane Smith", confidence=0.78)
        ]
        
        result = FaceRecognitionResult(
            face_detection=self.mock_face_detection,
            is_known=True,
            person_matches=matches,
            confidence=0.92
        )
        
        self.assertEqual(len(result.person_matches), 2)
        self.assertEqual(result.best_match.person_id, "person_001")
        self.assertEqual(result.identity, "John Doe")
        self.assertTrue(result.is_recognized)
    
    def test_recognition_result_blacklisted(self):
        """Test FaceRecognitionResult for blacklisted person."""
        matches = [
            PersonMatch(person_id="blacklist_001", confidence=0.95, is_blacklisted=True)
        ]
        
        result = FaceRecognitionResult(
            face_detection=self.mock_face_detection,
            is_blacklisted=True,
            person_matches=matches,
            confidence=0.95
        )
        
        self.assertTrue(result.is_blacklisted)
        self.assertTrue(result.is_recognized)
        self.assertEqual(result.best_match.person_id, "blacklist_001")
    
    def test_recognition_result_unknown(self):
        """Test FaceRecognitionResult for unknown person."""
        result = FaceRecognitionResult(
            face_detection=self.mock_face_detection,
            is_known=False,
            confidence=0.0
        )
        
        self.assertFalse(result.is_known)
        self.assertFalse(result.is_blacklisted)
        self.assertFalse(result.is_recognized)
        self.assertIsNone(result.best_match)
        self.assertIsNone(result.identity)
    
    def test_recognition_result_to_dict(self):
        """Test converting FaceRecognitionResult to dictionary."""
        metadata = RecognitionMetadata(cache_hit=True, total_time=0.1)
        
        result = FaceRecognitionResult(
            face_detection=self.mock_face_detection,
            is_known=True,
            confidence=0.85,
            metadata=metadata
        )
        
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertTrue(result_dict['is_known'])
        self.assertEqual(result_dict['confidence'], 0.85)
        self.assertTrue(result_dict['is_recognized'])
        self.assertIsNotNone(result_dict['metadata'])
    
    def test_best_match_empty_list(self):
        """Test best_match property with empty person_matches."""
        result = FaceRecognitionResult(
            face_detection=self.mock_face_detection,
            person_matches=[]
        )
        
        self.assertIsNone(result.best_match)
        self.assertIsNone(result.identity)
    
    def test_identity_without_name(self):
        """Test identity property when person has no name."""
        matches = [
            PersonMatch(person_id="person_003", confidence=0.8)
        ]
        
        result = FaceRecognitionResult(
            face_detection=self.mock_face_detection,
            person_matches=matches
        )
        
        # Should fall back to person_id
        self.assertEqual(result.identity, "person_003")


class TestRecognitionResultIntegration(unittest.TestCase):
    """Integration tests for recognition result structures."""
    
    def test_complete_recognition_workflow(self):
        """Test complete recognition result workflow."""
        # Create mock face detection
        mock_detection = Mock()
        mock_detection.to_dict = Mock(return_value={'bbox': [0, 0, 100, 100]})
        
        # Create metadata
        metadata = RecognitionMetadata(
            tolerance_used=0.6,
            cache_hit=False,
            processing_method="database",
            encoding_time=0.05,
            matching_time=0.15,
            total_time=0.20,
            database_query_count=1,
            faces_compared=50
        )
        
        # Create person matches
        matches = [
            PersonMatch(
                person_id="person_001",
                person_name="Alice",
                confidence=0.88,
                similarity_score=0.92,
                match_count=10
            ),
            PersonMatch(
                person_id="person_002",
                person_name="Bob",
                confidence=0.65,
                similarity_score=0.70,
                match_count=3
            )
        ]
        
        # Create recognition result
        result = FaceRecognitionResult(
            face_detection=mock_detection,
            is_known=True,
            person_matches=matches,
            confidence=0.88,
            metadata=metadata
        )
        
        # Verify complete result
        self.assertTrue(result.is_recognized)
        self.assertEqual(result.best_match.person_name, "Alice")
        self.assertEqual(result.identity, "Alice")
        self.assertEqual(len(result.person_matches), 2)
        self.assertFalse(result.metadata.cache_hit)
        self.assertEqual(result.metadata.faces_compared, 50)
        
        # Verify serialization
        result_dict = result.to_dict()
        self.assertIn('is_recognized', result_dict)
        self.assertIn('metadata', result_dict)
        self.assertEqual(result_dict['person_matches'][0]['person_name'], "Alice")


if __name__ == '__main__':
    unittest.main()
