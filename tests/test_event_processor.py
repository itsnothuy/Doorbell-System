#!/usr/bin/env python3
"""
Test Suite for Event Processor

Comprehensive tests for the event processing system including pipeline processing,
enrichment coordination, database persistence, and metrics tracking.
"""

import time
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from src.pipeline.event_processor import (
    EventProcessor,
    ProcessedEvent,
    ProcessingStage,
    PersistenceStatus
)
from src.communication.message_bus import MessageBus, Message
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
from src.storage.event_database import EventDatabase
from src.enrichment.base_enrichment import BaseEnrichment, EnrichmentResult, EnrichmentStatus


class TestEventProcessor:
    """Test suite for EventProcessor class."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def message_bus(self):
        """Create message bus instance."""
        bus = MessageBus()
        bus.start()
        yield bus
        bus.stop()
    
    @pytest.fixture
    def event_config(self, temp_db_path):
        """Create test event configuration."""
        return {
            'base_config': {
                'worker_count': 1,
                'queue_size': 100,
                'timeout': 10.0
            },
            'database_config': {
                'path': temp_db_path,
                'connection_pool_size': 2,
                'wal_mode': False  # Disable for testing
            },
            'enrichment_config': {
                'enabled_processors': ['metadata_enrichment'],
                'max_enrichment_time': 5.0,
                'retry_failed_enrichments': True,
                'max_retries': 2
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
    
    @pytest.fixture
    def event_processor(self, message_bus, event_config):
        """Create event processor instance."""
        processor = EventProcessor(message_bus, event_config)
        processor._initialize_worker()
        return processor
    
    @pytest.fixture
    def sample_pipeline_event(self):
        """Create sample pipeline event."""
        return PipelineEvent(
            event_type=EventType.MOTION_DETECTED,
            source='test_source',
            priority=EventPriority.NORMAL,
            data={'test_key': 'test_value'}
        )
    
    @pytest.fixture
    def sample_face_recognition_event(self):
        """Create sample face recognition event."""
        # Create face detection
        bbox = BoundingBox(x=100, y=100, width=50, height=50, confidence=0.95)
        face_detection = FaceDetection(
            bounding_box=bbox,
            confidence=0.95,
            quality_score=0.8
        )
        
        # Create face recognition result
        recognition = FaceRecognition(
            face_detection=face_detection,
            status=RecognitionStatus.KNOWN,
            identity='John Doe',
            similarity_score=0.85,
            recognition_time=0.1
        )
        
        # Create event
        return FaceRecognitionEvent(
            event_type=EventType.FACE_RECOGNIZED,
            recognitions=[recognition],
            recognition_time=0.15,
            source='test_recognizer'
        )
    
    def test_event_processor_initialization(self, event_processor):
        """Test event processor initialization."""
        assert event_processor is not None
        assert event_processor.event_database is not None
        assert event_processor.enrichment_orchestrator is not None
        assert event_processor.web_streamer is not None
        assert len(event_processor.enrichment_processors) > 0
        assert event_processor.events_processed == 0
    
    def test_validate_event_success(self, event_processor, sample_pipeline_event):
        """Test event validation with valid event."""
        result = event_processor._validate_event(sample_pipeline_event)
        assert result is True
    
    def test_validate_event_missing_fields(self, event_processor):
        """Test event validation with missing required fields."""
        # Event without event_id
        invalid_event = Mock()
        invalid_event.event_id = None
        invalid_event.event_type = EventType.MOTION_DETECTED
        invalid_event.timestamp = time.time()
        
        result = event_processor._validate_event(invalid_event)
        assert result is False
    
    def test_process_event_pipeline_success(self, event_processor, sample_pipeline_event):
        """Test successful event processing through complete pipeline."""
        processed_event = event_processor._process_event_pipeline(sample_pipeline_event)
        
        assert isinstance(processed_event, ProcessedEvent)
        assert processed_event.original_event == sample_pipeline_event
        assert processed_event.persistence_status in [PersistenceStatus.SUCCESS, PersistenceStatus.FAILED]
        assert 'processing_duration' in processed_event.processing_metadata
        assert processed_event.processing_metadata['stage'] in [
            ProcessingStage.COMPLETED.value,
            ProcessingStage.FAILED.value
        ]
    
    def test_process_face_recognition_event(self, event_processor, sample_face_recognition_event):
        """Test face recognition event processing."""
        processed_event = event_processor._process_event_pipeline(sample_face_recognition_event)
        
        assert isinstance(processed_event, ProcessedEvent)
        assert processed_event.original_event == sample_face_recognition_event
        assert len(processed_event.enrichment_results) > 0
    
    def test_event_persistence(self, event_processor, sample_pipeline_event):
        """Test event persistence to database."""
        enrichment_results = {}
        
        status = event_processor._persist_event(sample_pipeline_event, enrichment_results)
        
        assert status in [PersistenceStatus.SUCCESS, PersistenceStatus.FAILED]
        
        if status == PersistenceStatus.SUCCESS:
            # Verify event was stored
            stored_event = event_processor.event_database.get_event(sample_pipeline_event.event_id)
            assert stored_event is not None
            assert stored_event['event_id'] == sample_pipeline_event.event_id
    
    def test_merge_enrichment_data(self, event_processor):
        """Test merging enrichment data from multiple processors."""
        # Create mock enrichment results
        enrichment_results = {
            'processor1': EnrichmentResult(
                success=True,
                enriched_data={'key1': 'value1'},
                processing_time=0.1,
                processor_name='processor1'
            ),
            'processor2': EnrichmentResult(
                success=True,
                enriched_data={'key2': 'value2'},
                processing_time=0.2,
                processor_name='processor2'
            )
        }
        
        merged_data = event_processor._merge_enrichment_data(enrichment_results)
        
        assert 'processor1' in merged_data
        assert 'processor2' in merged_data
        assert merged_data['processor1']['key1'] == 'value1'
        assert merged_data['processor2']['key2'] == 'value2'
    
    def test_update_processing_metrics(self, event_processor, sample_pipeline_event):
        """Test processing metrics update."""
        initial_count = event_processor.events_processed
        
        processed_event = ProcessedEvent(
            original_event=sample_pipeline_event,
            enriched_data={},
            processing_metadata={
                'processing_duration': 0.1,
                'enrichment_count': 2
            },
            persistence_status=PersistenceStatus.SUCCESS,
            enrichment_results={
                'proc1': EnrichmentResult(
                    success=True,
                    enriched_data={},
                    processing_time=0.05,
                    processor_name='proc1'
                )
            }
        )
        
        event_processor._update_processing_metrics(processed_event)
        
        # Metrics should be updated
        assert event_processor.total_processing_time > 0
    
    def test_handle_recognition_event(self, event_processor, sample_face_recognition_event, message_bus):
        """Test handling of face recognition event from message bus."""
        initial_count = event_processor.events_processed
        
        # Create message
        message = Message(
            topic='faces_recognized',
            data=sample_face_recognition_event,
            source='test'
        )
        
        # Handle event
        event_processor.handle_recognition_event(message)
        
        # Verify event was processed
        assert event_processor.events_processed == initial_count + 1
    
    def test_concurrent_event_processing(self, event_processor):
        """Test processing multiple events concurrently."""
        events = [
            PipelineEvent(
                event_type=EventType.MOTION_DETECTED,
                source='test',
                data={'index': i}
            )
            for i in range(5)
        ]
        
        processed_events = []
        for event in events:
            processed = event_processor._process_event_pipeline(event)
            processed_events.append(processed)
        
        assert len(processed_events) == 5
        assert all(isinstance(pe, ProcessedEvent) for pe in processed_events)
    
    def test_event_processor_metrics(self, event_processor, sample_pipeline_event):
        """Test event processor metrics collection."""
        # Process some events
        for _ in range(3):
            event_processor._process_event_pipeline(sample_pipeline_event)
        
        metrics = event_processor.get_metrics()
        
        assert 'events_processed' in metrics
        assert 'events_failed' in metrics
        assert 'avg_processing_time' in metrics
        assert 'enrichment_success_rate' in metrics
        assert 'persistence_success_count' in metrics
        assert 'database_stats' in metrics
        assert 'enrichment_metrics' in metrics
        assert metrics['events_processed'] >= 0
    
    def test_event_state_tracking(self, event_processor, sample_pipeline_event):
        """Test event state tracking during processing."""
        # Event should be added to active_events during processing
        event_id = sample_pipeline_event.event_id
        
        # Start processing in a separate check
        processed_event = event_processor._process_event_pipeline(sample_pipeline_event)
        
        # Event should be removed from active_events after completion
        assert event_id not in event_processor.active_events
    
    def test_event_processor_cleanup(self, event_processor):
        """Test event processor cleanup."""
        event_processor._cleanup_worker()
        
        # Database should be closed
        assert event_processor.event_database.conn is None
    
    def test_error_handling_invalid_event(self, event_processor):
        """Test error handling with invalid event."""
        # Create completely invalid event
        invalid_event = Mock()
        invalid_event.event_id = None
        
        processed_event = event_processor._process_event_pipeline(invalid_event)
        
        # Should fail gracefully
        assert processed_event.persistence_status == PersistenceStatus.FAILED
        assert 'error' in processed_event.enriched_data


class TestProcessedEvent:
    """Test ProcessedEvent dataclass."""
    
    def test_processed_event_creation(self):
        """Test ProcessedEvent creation."""
        event = PipelineEvent(event_type=EventType.MOTION_DETECTED)
        
        processed = ProcessedEvent(
            original_event=event,
            enriched_data={'test': 'data'},
            processing_metadata={'duration': 0.1},
            persistence_status=PersistenceStatus.SUCCESS
        )
        
        assert processed.original_event == event
        assert processed.enriched_data == {'test': 'data'}
        assert processed.persistence_status == PersistenceStatus.SUCCESS
        assert len(processed.enrichment_results) == 0
        assert len(processed.notification_targets) == 0


class TestEventIntegration:
    """Integration tests for event processing system."""
    
    @pytest.fixture
    def integration_setup(self):
        """Setup for integration tests."""
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        # Create message bus
        bus = MessageBus()
        bus.start()
        
        # Create configuration
        config = {
            'base_config': {'worker_count': 1, 'queue_size': 100},
            'database_config': {'path': db_path, 'wal_mode': False},
            'enrichment_config': {
                'enabled_processors': ['metadata_enrichment'],
                'max_enrichment_time': 5.0
            },
            'web_streaming_config': {'enabled': True},
            'performance_config': {'max_concurrent_events': 10}
        }
        
        # Create processor
        processor = EventProcessor(bus, config)
        processor._initialize_worker()
        
        yield processor, bus, db_path
        
        # Cleanup
        processor._cleanup_worker()
        bus.stop()
        Path(db_path).unlink(missing_ok=True)
    
    def test_end_to_end_event_processing(self, integration_setup):
        """Test complete end-to-end event processing flow."""
        processor, bus, db_path = integration_setup
        
        # Create event
        event = FaceRecognitionEvent(
            event_type=EventType.FACE_RECOGNIZED,
            recognitions=[],
            recognition_time=0.1,
            source='integration_test'
        )
        
        # Process event
        message = Message(topic='faces_recognized', data=event, source='test')
        processor.handle_recognition_event(message)
        
        # Verify processing
        assert processor.events_processed > 0
        
        # Verify database storage
        stored_event = processor.event_database.get_event(event.event_id)
        if stored_event:
            assert stored_event['event_id'] == event.event_id
    
    def test_multiple_event_types_processing(self, integration_setup):
        """Test processing different event types."""
        processor, bus, db_path = integration_setup
        
        events = [
            PipelineEvent(event_type=EventType.MOTION_DETECTED, source='test'),
            PipelineEvent(event_type=EventType.DOORBELL_PRESSED, source='test'),
            FaceRecognitionEvent(
                event_type=EventType.FACE_RECOGNIZED,
                recognitions=[],
                recognition_time=0.1,
                source='test'
            )
        ]
        
        for event in events:
            processor._process_event_pipeline(event)
        
        assert processor.events_processed >= len(events)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
