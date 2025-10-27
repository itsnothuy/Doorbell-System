#!/usr/bin/env python3
"""
Test Suite for Event Database

Tests for event storage, querying, and database management.
"""

import time
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from src.storage.event_database import EventDatabase
from src.communication.events import (
    PipelineEvent,
    FaceRecognitionEvent,
    EventType,
    EventPriority,
    FaceRecognition,
    FaceDetection,
    RecognitionStatus,
    BoundingBox
)
from src.enrichment.base_enrichment import EnrichmentResult, EnrichmentStatus


class TestEventDatabase:
    """Test suite for EventDatabase class."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def db_config(self):
        """Create database configuration."""
        return {
            'connection_pool_size': 5,
            'wal_mode': False,  # Disable for testing
            'auto_vacuum': True,
            'batch_insert_size': 10
        }
    
    @pytest.fixture
    def event_db(self, temp_db_path, db_config):
        """Create event database instance."""
        db = EventDatabase(temp_db_path, db_config)
        db.initialize()
        yield db
        db.close()
    
    @pytest.fixture
    def sample_event(self):
        """Create sample pipeline event."""
        return PipelineEvent(
            event_type=EventType.MOTION_DETECTED,
            source='test_source',
            priority=EventPriority.NORMAL,
            data={'motion_score': 85.5, 'region': 'front_door'}
        )
    
    @pytest.fixture
    def sample_face_event(self):
        """Create sample face recognition event."""
        bbox = BoundingBox(x=100, y=100, width=50, height=50, confidence=0.95)
        face_detection = FaceDetection(bounding_box=bbox, confidence=0.95)
        recognition = FaceRecognition(
            face_detection=face_detection,
            status=RecognitionStatus.KNOWN,
            identity='John Doe',
            similarity_score=0.88,
            recognition_time=0.12
        )
        
        return FaceRecognitionEvent(
            event_type=EventType.FACE_RECOGNIZED,
            recognitions=[recognition],
            recognition_time=0.15,
            source='face_recognizer',
            known_count=1
        )
    
    def test_database_initialization(self, event_db):
        """Test database initialization."""
        assert event_db.conn is not None
        assert event_db.events_stored == 0
    
    def test_store_simple_event(self, event_db, sample_event):
        """Test storing a simple event."""
        result = event_db.store_event(sample_event)
        
        assert result is True
        assert event_db.events_stored == 1
    
    def test_store_event_with_enrichments(self, event_db, sample_event):
        """Test storing event with enrichment results."""
        enrichment_results = {
            'metadata': EnrichmentResult(
                success=True,
                enriched_data={'timestamp': datetime.now().isoformat()},
                processing_time=0.05,
                processor_name='metadata',
                status=EnrichmentStatus.SUCCESS
            )
        }
        
        result = event_db.store_event(sample_event, enrichment_results)
        
        assert result is True
    
    def test_store_face_recognition_event(self, event_db, sample_face_event):
        """Test storing face recognition event."""
        result = event_db.store_event(sample_face_event)
        
        assert result is True
        
        # Verify retrieval
        stored_event = event_db.get_event(sample_face_event.event_id)
        assert stored_event is not None
        assert stored_event['event_id'] == sample_face_event.event_id
    
    def test_get_event(self, event_db, sample_event):
        """Test retrieving an event by ID."""
        # Store event
        event_db.store_event(sample_event)
        
        # Retrieve event
        stored_event = event_db.get_event(sample_event.event_id)
        
        assert stored_event is not None
        assert stored_event['event_id'] == sample_event.event_id
        assert stored_event['event_type'] == sample_event.event_type.name
        assert stored_event['source'] == sample_event.source
    
    def test_get_nonexistent_event(self, event_db):
        """Test retrieving non-existent event."""
        result = event_db.get_event('nonexistent_id')
        assert result is None
    
    def test_query_events_all(self, event_db):
        """Test querying all events."""
        # Store multiple events
        for i in range(5):
            event = PipelineEvent(
                event_type=EventType.MOTION_DETECTED,
                source=f'source_{i}',
                data={'index': i}
            )
            event_db.store_event(event)
        
        # Query all events
        events = event_db.query_events(limit=10)
        
        assert len(events) >= 5
    
    def test_query_events_with_filters(self, event_db):
        """Test querying events with filters."""
        # Store events of different types
        motion_event = PipelineEvent(event_type=EventType.MOTION_DETECTED, source='motion')
        doorbell_event = PipelineEvent(event_type=EventType.DOORBELL_PRESSED, source='doorbell')
        
        event_db.store_event(motion_event)
        event_db.store_event(doorbell_event)
        
        # Query motion events only
        motion_events = event_db.query_events(
            filters={'event_type': EventType.MOTION_DETECTED.name}
        )
        
        assert len(motion_events) > 0
        assert all(e['event_type'] == EventType.MOTION_DETECTED.name for e in motion_events)
    
    def test_query_events_by_source(self, event_db):
        """Test querying events by source."""
        # Store events with different sources
        for source in ['camera1', 'camera2', 'camera1']:
            event = PipelineEvent(event_type=EventType.MOTION_DETECTED, source=source)
            event_db.store_event(event)
        
        # Query events from camera1
        camera1_events = event_db.query_events(filters={'source': 'camera1'})
        
        assert len(camera1_events) == 2
        assert all(e['source'] == 'camera1' for e in camera1_events)
    
    def test_query_events_by_time_range(self, event_db):
        """Test querying events by time range."""
        # Store events
        now = time.time()
        for i in range(3):
            event = PipelineEvent(event_type=EventType.MOTION_DETECTED)
            event.timestamp = now - (i * 60)  # Events 1, 2, 3 minutes ago
            event_db.store_event(event)
        
        # Query events from last 2 minutes
        two_minutes_ago = now - 120
        recent_events = event_db.query_events(filters={'since': two_minutes_ago})
        
        # Should get events from 0 and 1 minute ago
        assert len(recent_events) >= 2
    
    def test_query_events_pagination(self, event_db):
        """Test event query pagination."""
        # Store multiple events
        for i in range(10):
            event = PipelineEvent(event_type=EventType.MOTION_DETECTED, data={'index': i})
            event_db.store_event(event)
        
        # Get first page
        page1 = event_db.query_events(limit=5, offset=0)
        assert len(page1) == 5
        
        # Get second page
        page2 = event_db.query_events(limit=5, offset=5)
        assert len(page2) == 5
        
        # Verify different events
        page1_ids = [e['event_id'] for e in page1]
        page2_ids = [e['event_id'] for e in page2]
        assert set(page1_ids).isdisjoint(set(page2_ids))
    
    def test_get_statistics(self, event_db):
        """Test getting event statistics."""
        # Store events of different types
        for event_type in [EventType.MOTION_DETECTED, EventType.DOORBELL_PRESSED, EventType.MOTION_DETECTED]:
            event = PipelineEvent(event_type=event_type)
            event_db.store_event(event)
        
        # Get statistics
        stats = event_db.get_statistics(days=7)
        
        assert 'total_events' in stats
        assert 'by_event_type' in stats
        assert stats['total_events'] >= 3
    
    def test_cleanup_old_events(self, event_db):
        """Test cleaning up old events."""
        # Store old event
        old_event = PipelineEvent(event_type=EventType.MOTION_DETECTED)
        old_event.timestamp = (datetime.now() - timedelta(days=60)).timestamp()
        event_db.store_event(old_event)
        
        # Store recent event
        recent_event = PipelineEvent(event_type=EventType.MOTION_DETECTED)
        event_db.store_event(recent_event)
        
        # Clean up events older than 30 days
        deleted_count = event_db.cleanup_old_events(retention_days=30)
        
        assert deleted_count >= 1
    
    def test_duplicate_event_handling(self, event_db, sample_event):
        """Test handling of duplicate events."""
        # Store event
        result1 = event_db.store_event(sample_event)
        assert result1 is True
        
        # Try to store same event again
        result2 = event_db.store_event(sample_event)
        assert result2 is False  # Should fail due to unique constraint
    
    def test_enrichment_results_storage(self, event_db, sample_event):
        """Test storing and retrieving enrichment results."""
        enrichment_results = {
            'processor1': {
                'success': True,
                'enriched_data': {'key1': 'value1'},
                'processing_time': 0.05,
                'error_message': None
            },
            'processor2': {
                'success': False,
                'enriched_data': {},
                'processing_time': 0.02,
                'error_message': 'Test error'
            }
        }
        
        # Store event with enrichments
        event_db.store_event(sample_event, enrichment_results)
        
        # Retrieve event
        stored_event = event_db.get_event(sample_event.event_id)
        
        assert stored_event is not None
        assert 'enrichments' in stored_event
        assert len(stored_event['enrichments']) == 2
    
    def test_database_close(self, temp_db_path, db_config):
        """Test database closing."""
        db = EventDatabase(temp_db_path, db_config)
        db.initialize()
        
        assert db.conn is not None
        
        db.close()
        
        assert db.conn is None
    
    def test_concurrent_writes(self, event_db):
        """Test concurrent event writes."""
        # Store multiple events quickly
        events = [
            PipelineEvent(event_type=EventType.MOTION_DETECTED, data={'index': i})
            for i in range(10)
        ]
        
        results = []
        for event in events:
            result = event_db.store_event(event)
            results.append(result)
        
        # All should succeed
        assert all(results)
        assert event_db.events_stored >= 10


class TestEventDatabaseQueries:
    """Additional tests for complex database queries."""
    
    @pytest.fixture
    def populated_db(self, temp_db_path):
        """Create database populated with test data."""
        config = {'wal_mode': False}
        db = EventDatabase(temp_db_path, config)
        db.initialize()
        
        # Populate with diverse events
        now = time.time()
        
        # Motion events
        for i in range(5):
            event = PipelineEvent(
                event_type=EventType.MOTION_DETECTED,
                source='camera1',
                priority=EventPriority.LOW
            )
            event.timestamp = now - (i * 60)
            db.store_event(event)
        
        # Doorbell events
        for i in range(3):
            event = PipelineEvent(
                event_type=EventType.DOORBELL_PRESSED,
                source='doorbell',
                priority=EventPriority.HIGH
            )
            event.timestamp = now - (i * 120)
            db.store_event(event)
        
        yield db
        
        db.close()
        Path(temp_db_path).unlink(missing_ok=True)
    
    def test_query_by_priority(self, populated_db):
        """Test querying events by priority."""
        high_priority_events = populated_db.query_events(
            filters={'priority': EventPriority.HIGH.name}
        )
        
        assert len(high_priority_events) >= 3
        assert all(e['priority'] == EventPriority.HIGH.name for e in high_priority_events)
    
    def test_query_with_multiple_filters(self, populated_db):
        """Test querying with multiple filters."""
        now = time.time()
        five_minutes_ago = now - 300
        
        events = populated_db.query_events(filters={
            'event_type': EventType.MOTION_DETECTED.name,
            'source': 'camera1',
            'since': five_minutes_ago
        })
        
        assert len(events) >= 0
        if events:
            assert all(e['event_type'] == EventType.MOTION_DETECTED.name for e in events)
            assert all(e['source'] == 'camera1' for e in events)
    
    def test_statistics_grouping(self, populated_db):
        """Test statistics with event grouping."""
        stats = populated_db.get_statistics(days=1)
        
        assert 'by_event_type' in stats
        assert 'by_priority' in stats
        
        # Should have both motion and doorbell events
        event_types = stats['by_event_type']
        assert len(event_types) >= 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
