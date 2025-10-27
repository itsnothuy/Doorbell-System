#!/usr/bin/env python3
"""
Test Suite for Enrichment Processors

Tests for base enrichment framework, enrichment orchestrator, metadata enrichment,
and web events streaming.
"""

import time
import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from src.enrichment.base_enrichment import (
    BaseEnrichment,
    EnrichmentResult,
    EnrichmentStatus
)
from src.enrichment.enrichment_orchestrator import EnrichmentOrchestrator
from src.enrichment.metadata_enrichment import MetadataEnrichment
from src.enrichment.web_events import WebEventsEnrichment, WebEventStreamer
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


class MockEnrichment(BaseEnrichment):
    """Mock enrichment processor for testing."""
    
    def can_process(self, event: Any) -> bool:
        return isinstance(event, PipelineEvent)
    
    def enrich(self, event: Any) -> EnrichmentResult:
        return EnrichmentResult(
            success=True,
            enriched_data={'mock_key': 'mock_value'},
            processing_time=0.01,
            processor_name=self.name,
            status=EnrichmentStatus.SUCCESS
        )


class FailingEnrichment(BaseEnrichment):
    """Enrichment processor that always fails."""
    
    def can_process(self, event: Any) -> bool:
        return isinstance(event, PipelineEvent)
    
    def enrich(self, event: Any) -> EnrichmentResult:
        raise ValueError("Intentional failure for testing")


class DependentEnrichment(BaseEnrichment):
    """Enrichment processor with dependencies."""
    
    def can_process(self, event: Any) -> bool:
        return isinstance(event, PipelineEvent)
    
    def enrich(self, event: Any) -> EnrichmentResult:
        return EnrichmentResult(
            success=True,
            enriched_data={'dependent_key': 'dependent_value'},
            processing_time=0.01,
            processor_name=self.name,
            status=EnrichmentStatus.SUCCESS
        )
    
    def get_dependencies(self) -> list:
        return ['MockEnrichment']


class TestBaseEnrichment:
    """Test suite for BaseEnrichment class."""
    
    @pytest.fixture
    def mock_enrichment(self):
        """Create mock enrichment processor."""
        config = {
            'priority': 5,
            'enabled': True,
            'timeout': 5.0
        }
        return MockEnrichment(config)
    
    @pytest.fixture
    def sample_event(self):
        """Create sample pipeline event."""
        return PipelineEvent(
            event_type=EventType.MOTION_DETECTED,
            source='test',
            data={'test': 'data'}
        )
    
    def test_base_enrichment_initialization(self, mock_enrichment):
        """Test base enrichment initialization."""
        assert mock_enrichment.name == 'MockEnrichment'
        assert mock_enrichment.priority == 5
        assert mock_enrichment.enabled is True
        assert mock_enrichment.timeout == 5.0
        assert mock_enrichment.processed_count == 0
    
    def test_process_event_success(self, mock_enrichment, sample_event):
        """Test successful event processing."""
        result = mock_enrichment.process_event(sample_event)
        
        assert isinstance(result, EnrichmentResult)
        assert result.success is True
        assert result.processor_name == 'MockEnrichment'
        assert result.status == EnrichmentStatus.SUCCESS
        assert 'mock_key' in result.enriched_data
        assert mock_enrichment.processed_count == 1
        assert mock_enrichment.success_count == 1
    
    def test_process_event_disabled(self):
        """Test processing with disabled processor."""
        config = {'priority': 5, 'enabled': False}
        processor = MockEnrichment(config)
        event = PipelineEvent(event_type=EventType.MOTION_DETECTED)
        
        result = processor.process_event(event)
        
        assert result.success is True
        assert result.status == EnrichmentStatus.SKIPPED
        assert result.metadata['reason'] == 'processor_disabled'
    
    def test_process_event_failure(self, sample_event):
        """Test processing with failing enrichment."""
        config = {'priority': 5, 'enabled': True}
        processor = FailingEnrichment(config)
        
        result = processor.process_event(sample_event)
        
        assert result.success is False
        assert result.status == EnrichmentStatus.FAILED
        assert result.error_message is not None
        assert processor.failure_count == 1
    
    def test_get_metrics(self, mock_enrichment, sample_event):
        """Test metrics collection."""
        # Process some events
        for _ in range(3):
            mock_enrichment.process_event(sample_event)
        
        metrics = mock_enrichment.get_metrics()
        
        assert metrics['processed_count'] == 3
        assert metrics['success_count'] == 3
        assert metrics['failure_count'] == 0
        assert metrics['success_rate'] == 1.0
        assert metrics['avg_processing_time'] > 0
    
    def test_reset_metrics(self, mock_enrichment, sample_event):
        """Test metrics reset."""
        mock_enrichment.process_event(sample_event)
        assert mock_enrichment.processed_count > 0
        
        mock_enrichment.reset_metrics()
        
        assert mock_enrichment.processed_count == 0
        assert mock_enrichment.success_count == 0
        assert mock_enrichment.failure_count == 0
    
    def test_get_dependencies(self, mock_enrichment):
        """Test getting processor dependencies."""
        dependencies = mock_enrichment.get_dependencies()
        assert isinstance(dependencies, list)
        assert len(dependencies) == 0


class TestEnrichmentOrchestrator:
    """Test suite for EnrichmentOrchestrator class."""
    
    @pytest.fixture
    def processors(self):
        """Create list of test processors."""
        return [
            MockEnrichment({'priority': 2, 'enabled': True}),
            DependentEnrichment({'priority': 5, 'enabled': True})
        ]
    
    @pytest.fixture
    def orchestrator_config(self):
        """Create orchestrator configuration."""
        return {
            'max_retries': 3,
            'retry_delay': 0.1,
            'timeout_per_processor': 5.0,
            'max_enrichment_time': 10.0,
            'retry_failed_enrichments': True
        }
    
    @pytest.fixture
    def orchestrator(self, processors, orchestrator_config):
        """Create enrichment orchestrator."""
        return EnrichmentOrchestrator(processors, orchestrator_config)
    
    @pytest.fixture
    def sample_event(self):
        """Create sample pipeline event."""
        event = PipelineEvent(
            event_type=EventType.MOTION_DETECTED,
            source='test'
        )
        event.enrichments = []  # Add enrichments tracking
        return event
    
    def test_orchestrator_initialization(self, orchestrator, processors):
        """Test orchestrator initialization."""
        assert len(orchestrator.processors) == len(processors)
        assert orchestrator.max_retries == 3
        assert orchestrator.retry_delay == 0.1
        assert len(orchestrator.dependency_graph) == len(processors)
    
    def test_process_event(self, orchestrator, sample_event):
        """Test event processing through orchestrator."""
        results = orchestrator.process_event(sample_event)
        
        assert isinstance(results, dict)
        assert len(results) > 0
        assert all(isinstance(r, EnrichmentResult) for r in results.values())
    
    def test_dependency_ordering(self, orchestrator):
        """Test processor ordering based on dependencies."""
        processing_order = orchestrator.processing_order
        
        # MockEnrichment should come before DependentEnrichment
        mock_index = next(i for i, p in enumerate(processing_order) if p.name == 'MockEnrichment')
        dep_index = next(i for i, p in enumerate(processing_order) if p.name == 'DependentEnrichment')
        
        assert mock_index < dep_index
    
    def test_get_metrics(self, orchestrator, sample_event):
        """Test orchestrator metrics."""
        orchestrator.process_event(sample_event)
        
        metrics = orchestrator.get_metrics()
        
        assert 'events_processed' in metrics
        assert 'avg_enrichment_time' in metrics
        assert 'processor_metrics' in metrics
        assert metrics['events_processed'] > 0
    
    def test_enable_disable_processor(self, orchestrator):
        """Test enabling/disabling processors."""
        processor_name = 'MockEnrichment'
        
        # Disable
        result = orchestrator.disable_processor(processor_name)
        assert result is True
        
        processor = orchestrator.get_processor(processor_name)
        assert processor.enabled is False
        
        # Enable
        result = orchestrator.enable_processor(processor_name)
        assert result is True
        assert processor.enabled is True
    
    def test_get_processor(self, orchestrator):
        """Test getting processor by name."""
        processor = orchestrator.get_processor('MockEnrichment')
        assert processor is not None
        assert processor.name == 'MockEnrichment'
        
        # Non-existent processor
        processor = orchestrator.get_processor('NonExistent')
        assert processor is None


class TestMetadataEnrichment:
    """Test suite for MetadataEnrichment class."""
    
    @pytest.fixture
    def metadata_enrichment(self):
        """Create metadata enrichment processor."""
        config = {
            'priority': 1,
            'enabled': True,
            'include_system_info': True,
            'include_timestamps': True,
            'include_processing_context': True
        }
        return MetadataEnrichment(config)
    
    @pytest.fixture
    def sample_event(self):
        """Create sample pipeline event."""
        return PipelineEvent(
            event_type=EventType.MOTION_DETECTED,
            source='test_source',
            priority=EventPriority.NORMAL
        )
    
    def test_metadata_enrichment_initialization(self, metadata_enrichment):
        """Test metadata enrichment initialization."""
        assert metadata_enrichment.name == 'MetadataEnrichment'
        assert metadata_enrichment.priority == 1
        assert metadata_enrichment.include_system_info is True
        assert metadata_enrichment.include_timestamps is True
    
    def test_can_process(self, metadata_enrichment, sample_event):
        """Test can_process method."""
        assert metadata_enrichment.can_process(sample_event) is True
        assert metadata_enrichment.can_process("not an event") is False
    
    def test_enrich_event(self, metadata_enrichment, sample_event):
        """Test event enrichment."""
        result = metadata_enrichment.enrich(sample_event)
        
        assert isinstance(result, EnrichmentResult)
        assert result.success is True
        assert result.status == EnrichmentStatus.SUCCESS
        assert 'timestamps' in result.enriched_data
        assert 'processing_context' in result.enriched_data
    
    def test_timestamp_enrichment(self, metadata_enrichment, sample_event):
        """Test timestamp information enrichment."""
        result = metadata_enrichment.enrich(sample_event)
        
        timestamps = result.enriched_data['timestamps']
        assert 'enrichment_timestamp' in timestamps
        assert 'event_timestamp' in timestamps
        assert 'age_seconds' in timestamps
    
    def test_processing_context_enrichment(self, metadata_enrichment, sample_event):
        """Test processing context enrichment."""
        result = metadata_enrichment.enrich(sample_event)
        
        context = result.enriched_data['processing_context']
        assert 'event_id' in context
        assert 'event_type' in context
        assert 'source' in context
        assert context['source'] == 'test_source'
    
    def test_face_recognition_enrichment(self, metadata_enrichment):
        """Test enrichment of face recognition event."""
        bbox = BoundingBox(x=100, y=100, width=50, height=50)
        face_detection = FaceDetection(bounding_box=bbox, confidence=0.9)
        recognition = FaceRecognition(
            face_detection=face_detection,
            status=RecognitionStatus.KNOWN,
            identity='Test Person',
            similarity_score=0.85
        )
        
        event = FaceRecognitionEvent(
            event_type=EventType.FACE_RECOGNIZED,
            recognitions=[recognition],
            recognition_time=0.1
        )
        
        result = metadata_enrichment.enrich(event)
        
        assert result.success is True
        assert 'type_specific' in result.enriched_data
        
        type_specific = result.enriched_data['type_specific']
        if 'recognition_summary' in type_specific:
            summary = type_specific['recognition_summary']
            assert summary['total_recognitions'] == 1
    
    def test_no_dependencies(self, metadata_enrichment):
        """Test that metadata enrichment has no dependencies."""
        dependencies = metadata_enrichment.get_dependencies()
        assert len(dependencies) == 0


class TestWebEventStreamer:
    """Test suite for WebEventStreamer class."""
    
    @pytest.fixture
    def streamer_config(self):
        """Create streamer configuration."""
        return {
            'enabled': True,
            'max_connections': 5,
            'buffer_size': 10,
            'heartbeat_interval': 30.0
        }
    
    @pytest.fixture
    def streamer(self, streamer_config):
        """Create web event streamer."""
        return WebEventStreamer(streamer_config)
    
    def test_streamer_initialization(self, streamer):
        """Test streamer initialization."""
        assert streamer.enabled is True
        assert streamer.max_connections == 5
        assert streamer.buffer_size == 10
        assert streamer.active_connections == 0
    
    def test_register_client(self, streamer):
        """Test client registration."""
        client_id = 'test_client_1'
        result = streamer.register_client(client_id)
        
        assert result is True
        assert streamer.active_connections == 1
        assert client_id in streamer.client_queues
    
    def test_register_multiple_clients(self, streamer):
        """Test registering multiple clients."""
        for i in range(3):
            result = streamer.register_client(f'client_{i}')
            assert result is True
        
        assert streamer.active_connections == 3
    
    def test_max_connections_limit(self, streamer):
        """Test max connections limit."""
        # Register up to max
        for i in range(streamer.max_connections):
            streamer.register_client(f'client_{i}')
        
        # Try to register one more
        result = streamer.register_client('excess_client')
        assert result is False
        assert streamer.active_connections == streamer.max_connections
    
    def test_unregister_client(self, streamer):
        """Test client unregistration."""
        client_id = 'test_client'
        streamer.register_client(client_id)
        assert streamer.active_connections == 1
        
        streamer.unregister_client(client_id)
        assert streamer.active_connections == 0
        assert client_id not in streamer.client_queues
    
    def test_stream_event(self, streamer):
        """Test event streaming."""
        # Register client
        client_id = 'test_client'
        streamer.register_client(client_id)
        
        # Stream event
        event_data = {'event_id': 'test_123', 'data': 'test'}
        delivered_count = streamer.stream_event(event_data)
        
        assert delivered_count == 1
        assert streamer.events_streamed == 1
    
    def test_get_events(self, streamer):
        """Test getting events for client."""
        client_id = 'test_client'
        streamer.register_client(client_id)
        
        # Stream some events
        for i in range(3):
            streamer.stream_event({'event_id': f'event_{i}'})
        
        # Get events
        events = streamer.get_events(client_id, timeout=0.5)
        
        assert len(events) == 3
    
    def test_get_stats(self, streamer):
        """Test getting streamer statistics."""
        streamer.register_client('client_1')
        streamer.stream_event({'test': 'event'})
        
        stats = streamer.get_stats()
        
        assert stats['enabled'] is True
        assert stats['active_connections'] == 1
        assert stats['events_streamed'] == 1


class TestWebEventsEnrichment:
    """Test suite for WebEventsEnrichment class."""
    
    @pytest.fixture
    def web_enrichment(self):
        """Create web events enrichment processor."""
        config = {
            'priority': 8,
            'enabled': True,
            'streaming': {
                'enabled': True,
                'max_connections': 10,
                'buffer_size': 50
            },
            'stream_all_events': True
        }
        return WebEventsEnrichment(config)
    
    @pytest.fixture
    def sample_event(self):
        """Create sample pipeline event."""
        return PipelineEvent(
            event_type=EventType.MOTION_DETECTED,
            source='test'
        )
    
    def test_web_enrichment_initialization(self, web_enrichment):
        """Test web enrichment initialization."""
        assert web_enrichment.name == 'WebEventsEnrichment'
        assert web_enrichment.priority == 8
        assert web_enrichment.streamer is not None
    
    def test_can_process(self, web_enrichment, sample_event):
        """Test can_process method."""
        assert web_enrichment.can_process(sample_event) is True
    
    def test_enrich_event(self, web_enrichment, sample_event):
        """Test event enrichment and streaming."""
        result = web_enrichment.enrich(sample_event)
        
        assert isinstance(result, EnrichmentResult)
        assert result.success is True
        assert 'web_streaming' in result.enriched_data
        assert result.enriched_data['web_streaming']['streamed'] is True
    
    def test_prepare_web_event(self, web_enrichment, sample_event):
        """Test preparing event for web streaming."""
        web_event = web_enrichment._prepare_web_event(sample_event)
        
        assert 'event_id' in web_event
        assert 'event_type' in web_event
        assert 'timestamp' in web_event
        assert 'data' in web_event
    
    def test_sanitize_data(self, web_enrichment):
        """Test data sanitization for web."""
        data = {
            'string': 'test',
            'number': 42,
            'binary': b'binary_data',
            'nested': {'key': 'value'}
        }
        
        sanitized = web_enrichment._sanitize_data(data)
        
        assert sanitized['string'] == 'test'
        assert sanitized['number'] == 42
        assert '<binary data' in sanitized['binary']
        assert sanitized['nested']['key'] == 'value'
    
    def test_get_dependencies(self, web_enrichment):
        """Test getting processor dependencies."""
        dependencies = web_enrichment.get_dependencies()
        assert 'MetadataEnrichment' in dependencies


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
