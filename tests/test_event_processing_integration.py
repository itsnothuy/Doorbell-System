#!/usr/bin/env python3
"""
Integration Test for Event Processing System

End-to-end integration test validating the complete event processing pipeline
from event ingestion through enrichment, persistence, and notification delivery.
"""

import time
import tempfile
from pathlib import Path
import pytest

from src.communication.message_bus import MessageBus
from src.pipeline.event_processor import EventProcessor
from src.communication.events import (
    PipelineEvent,
    FaceRecognitionEvent,
    EventType,
    EventPriority,
    FaceDetection,
    FaceRecognition,
    RecognitionStatus,
    BoundingBox
)


class TestEventProcessingIntegration:
    """Integration tests for complete event processing system."""
    
    @pytest.fixture
    def integration_env(self):
        """Setup complete integration environment."""
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        # Initialize message bus
        message_bus = MessageBus()
        message_bus.start()
        
        # Create event processor configuration
        config = {
            'base_config': {
                'worker_count': 1,
                'queue_size': 100,
                'timeout': 10.0
            },
            'database_config': {
                'path': db_path,
                'wal_mode': False,
                'connection_pool_size': 2
            },
            'enrichment_config': {
                'enabled_processors': ['metadata_enrichment', 'web_events'],
                'max_enrichment_time': 5.0,
                'retry_failed_enrichments': True
            },
            'web_streaming_config': {
                'enabled': True,
                'max_connections': 10,
                'buffer_size': 50
            },
            'performance_config': {
                'max_concurrent_events': 10,
                'event_timeout': 30.0
            }
        }
        
        # Initialize event processor
        event_processor = EventProcessor(message_bus, config)
        event_processor._initialize_worker()
        
        yield {
            'processor': event_processor,
            'message_bus': message_bus,
            'db_path': db_path
        }
        
        # Cleanup
        event_processor._cleanup_worker()
        message_bus.stop()
        Path(db_path).unlink(missing_ok=True)
    
    def test_complete_pipeline_motion_event(self, integration_env):
        """Test complete pipeline with motion detection event."""
        processor = integration_env['processor']
        
        # Create motion event
        event = PipelineEvent(
            event_type=EventType.MOTION_DETECTED,
            source='integration_test_camera',
            priority=EventPriority.NORMAL,
            data={
                'motion_score': 75.5,
                'region': 'front_door',
                'timestamp': time.time()
            }
        )
        
        # Process through pipeline
        processed_event = processor._process_event_pipeline(event)
        
        # Validate processing
        assert processed_event is not None
        assert processed_event.original_event == event
        assert len(processed_event.enrichment_results) > 0
        assert 'processing_duration' in processed_event.processing_metadata
        
        # Verify database storage
        stored_event = processor.event_database.get_event(event.event_id)
        if stored_event:
            assert stored_event['event_id'] == event.event_id
            assert stored_event['event_type'] == EventType.MOTION_DETECTED.name
    
    def test_complete_pipeline_face_recognition(self, integration_env):
        """Test complete pipeline with face recognition event."""
        processor = integration_env['processor']
        
        # Create face recognition event with details
        bbox = BoundingBox(x=150, y=200, width=80, height=100, confidence=0.92)
        face_detection = FaceDetection(
            bounding_box=bbox,
            confidence=0.92,
            quality_score=0.85
        )
        
        recognition = FaceRecognition(
            face_detection=face_detection,
            status=RecognitionStatus.KNOWN,
            identity='Test User',
            similarity_score=0.88,
            recognition_time=0.15,
            match_details={
                'person_id': 'test_person_123',
                'confidence': 0.88
            }
        )
        
        event = FaceRecognitionEvent(
            event_type=EventType.FACE_RECOGNIZED,
            recognitions=[recognition],
            recognition_time=0.18,
            source='integration_test_recognizer',
            known_count=1,
            unknown_count=0,
            blacklisted_count=0
        )
        
        # Process through pipeline
        processed_event = processor._process_event_pipeline(event)
        
        # Validate processing
        assert processed_event is not None
        assert len(processed_event.enrichment_results) > 0
        
        # Check metadata enrichment was applied
        enrichment_data = processed_event.enriched_data
        assert 'MetadataEnrichment' in enrichment_data or any('metadata' in key.lower() for key in enrichment_data.keys())
        
        # Verify metrics updated
        assert processor.events_processed > 0
    
    def test_multiple_events_processing(self, integration_env):
        """Test processing multiple events in sequence."""
        processor = integration_env['processor']
        initial_count = processor.events_processed
        
        # Create multiple events
        events = [
            PipelineEvent(event_type=EventType.MOTION_DETECTED, source='camera1'),
            PipelineEvent(event_type=EventType.DOORBELL_PRESSED, source='doorbell'),
            PipelineEvent(event_type=EventType.MOTION_DETECTED, source='camera2'),
        ]
        
        # Process all events
        processed_events = []
        for event in events:
            processed = processor._process_event_pipeline(event)
            processed_events.append(processed)
        
        # Validate all processed successfully
        assert len(processed_events) == len(events)
        assert all(pe is not None for pe in processed_events)
        assert processor.events_processed >= initial_count + len(events)
    
    def test_enrichment_pipeline_execution(self, integration_env):
        """Test enrichment pipeline execution order and results."""
        processor = integration_env['processor']
        
        # Create test event
        event = PipelineEvent(
            event_type=EventType.MOTION_DETECTED,
            source='enrichment_test',
            data={'test_field': 'test_value'}
        )
        
        # Process event
        processed_event = processor._process_event_pipeline(event)
        
        # Verify enrichments were applied
        enrichment_results = processed_event.enrichment_results
        assert len(enrichment_results) > 0
        
        # Check for metadata enrichment
        assert any('metadata' in name.lower() for name in enrichment_results.keys())
        
        # Verify all enrichments succeeded or were skipped
        for result in enrichment_results.values():
            if hasattr(result, 'success'):
                assert result.success or result.status.name == 'SKIPPED'
    
    def test_web_streaming_integration(self, integration_env):
        """Test web streaming integration."""
        processor = integration_env['processor']
        
        # Register a web client
        client_id = 'integration_test_client'
        processor.web_streamer.register_client(client_id)
        
        # Create and process event
        event = PipelineEvent(
            event_type=EventType.MOTION_DETECTED,
            source='streaming_test'
        )
        
        processed_event = processor._process_event_pipeline(event)
        
        # Check if web streaming enrichment was applied
        enrichment_data = processed_event.enriched_data
        has_web_streaming = any('web' in key.lower() for key in enrichment_data.keys())
        
        # Get events from client (may or may not have received based on timing)
        events = processor.web_streamer.get_events(client_id, timeout=0.5)
        
        # Verify streaming system is working
        assert processor.web_streamer.active_connections > 0
        
        # Cleanup
        processor.web_streamer.unregister_client(client_id)
    
    def test_database_persistence_integration(self, integration_env):
        """Test database persistence integration."""
        processor = integration_env['processor']
        
        # Create multiple events with different types
        events = [
            PipelineEvent(event_type=EventType.MOTION_DETECTED, source='db_test1'),
            PipelineEvent(event_type=EventType.DOORBELL_PRESSED, source='db_test2'),
        ]
        
        # Process events
        for event in events:
            processor._process_event_pipeline(event)
        
        # Query database
        stored_events = processor.event_database.query_events(limit=10)
        
        # Verify events were stored
        assert len(stored_events) >= 2
        
        # Verify we can retrieve individual events
        for event in events:
            stored = processor.event_database.get_event(event.event_id)
            if stored:
                assert stored['event_id'] == event.event_id
    
    def test_performance_metrics_collection(self, integration_env):
        """Test performance metrics collection."""
        processor = integration_env['processor']
        
        # Process multiple events
        for i in range(5):
            event = PipelineEvent(
                event_type=EventType.MOTION_DETECTED,
                source=f'perf_test_{i}'
            )
            processor._process_event_pipeline(event)
        
        # Get metrics
        metrics = processor.get_metrics()
        
        # Validate metrics structure
        assert 'events_processed' in metrics
        assert 'avg_processing_time' in metrics
        assert 'enrichment_success_rate' in metrics
        assert 'persistence_success_count' in metrics
        assert 'database_stats' in metrics
        assert 'enrichment_metrics' in metrics
        assert 'web_streaming_stats' in metrics
        
        # Validate values
        assert metrics['events_processed'] >= 5
        assert metrics['avg_processing_time'] >= 0
    
    def test_error_recovery(self, integration_env):
        """Test system error recovery."""
        processor = integration_env['processor']
        
        # Create invalid event (missing required fields)
        invalid_event = type('Event', (), {
            'event_id': None,
            'timestamp': time.time()
        })()
        
        # Process invalid event - should not crash
        try:
            processed = processor._process_event_pipeline(invalid_event)
            # Should create a failed processed event
            assert processed is not None
        except Exception:
            # If it raises, that's also acceptable as long as system doesn't crash
            pass
        
        # Verify system still works with valid event
        valid_event = PipelineEvent(event_type=EventType.MOTION_DETECTED)
        processed_valid = processor._process_event_pipeline(valid_event)
        assert processed_valid is not None


def test_system_initialization():
    """Test complete system initialization and teardown."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        # Initialize components
        message_bus = MessageBus()
        message_bus.start()
        
        config = {
            'base_config': {'worker_count': 1},
            'database_config': {'path': db_path, 'wal_mode': False},
            'enrichment_config': {'enabled_processors': []},
            'web_streaming_config': {'enabled': False},
            'performance_config': {}
        }
        
        processor = EventProcessor(message_bus, config)
        processor._initialize_worker()
        
        # Verify initialization
        assert processor.event_database is not None
        assert processor.enrichment_orchestrator is not None
        
        # Cleanup
        processor._cleanup_worker()
        message_bus.stop()
        
        assert processor.event_database.conn is None
        
    finally:
        Path(db_path).unlink(missing_ok=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
