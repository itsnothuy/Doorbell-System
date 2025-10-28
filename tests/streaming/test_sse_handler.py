#!/usr/bin/env python3
"""
Tests for SSE Handler

Tests the Server-Sent Events streaming implementation.
"""

import pytest
import time
import json
from unittest.mock import Mock, MagicMock
from queue import Queue

from src.streaming.sse_handler import SSEHandler


class TestSSEHandler:
    """Test suite for SSE Handler."""
    
    @pytest.fixture
    def mock_web_event_streamer(self):
        """Create mock web event streamer."""
        streamer = Mock()
        streamer.register_client = Mock(return_value=True)
        streamer.unregister_client = Mock()
        streamer.client_queues = {}
        streamer.get_stats = Mock(return_value={
            'active_connections': 1,
            'events_streamed': 10,
            'buffer_size': 5
        })
        return streamer
    
    @pytest.fixture
    def sse_handler(self, mock_web_event_streamer):
        """Create SSE handler instance."""
        return SSEHandler(mock_web_event_streamer)
    
    def test_initialization(self, sse_handler):
        """Test SSE handler initializes correctly."""
        assert sse_handler.web_event_streamer is not None
        assert sse_handler.heartbeat_interval == 30.0
        assert sse_handler.max_queue_size == 100
        assert len(sse_handler.active_streams) == 0
    
    def test_format_sse_event(self, sse_handler):
        """Test SSE event formatting."""
        event_str = sse_handler._format_sse_event(
            event='test_event',
            data={'message': 'test'},
            event_id='123'
        )
        
        assert 'id: 123' in event_str
        assert 'event: test_event' in event_str
        assert 'data: {' in event_str
        assert '"message":"test"' in event_str
        assert event_str.endswith('\n\n')
    
    def test_format_sse_event_without_id(self, sse_handler):
        """Test SSE event formatting without event ID."""
        event_str = sse_handler._format_sse_event(
            event='test_event',
            data={'message': 'test'}
        )
        
        assert 'id:' not in event_str
        assert 'event: test_event' in event_str
        assert 'data: {' in event_str
    
    def test_get_system_status(self, sse_handler):
        """Test system status retrieval."""
        status = sse_handler._get_system_status()
        
        assert 'timestamp' in status
        assert 'streaming_status' in status
        assert 'active_connections' in status
        assert status['active_connections'] == 1
        assert status['events_streamed'] == 10
    
    def test_create_event_stream_max_connections(self, mock_web_event_streamer):
        """Test event stream creation when max connections reached."""
        mock_web_event_streamer.register_client = Mock(return_value=False)
        sse_handler = SSEHandler(mock_web_event_streamer)
        
        stream = sse_handler.create_event_stream('client_1')
        first_event = next(stream)
        
        assert 'event: error' in first_event
        assert 'Max connections reached' in first_event
    
    def test_create_event_stream_connection(self, sse_handler, mock_web_event_streamer):
        """Test successful event stream connection."""
        # Setup client queue
        client_queue = Queue()
        mock_web_event_streamer.client_queues['client_1'] = client_queue
        
        stream = sse_handler.create_event_stream('client_1')
        first_event = next(stream)
        
        assert 'event: connected' in first_event
        assert 'client_1' in first_event
        mock_web_event_streamer.register_client.assert_called_once_with('client_1')
    
    def test_get_next_event(self, sse_handler, mock_web_event_streamer):
        """Test retrieving next event from queue."""
        # Setup client queue with event
        client_queue = Queue()
        test_event = {'type': 'test', 'data': 'test_data'}
        client_queue.put(test_event)
        mock_web_event_streamer.client_queues['client_1'] = client_queue
        
        event = sse_handler._get_next_event('client_1')
        
        assert event == test_event
    
    def test_get_next_event_timeout(self, sse_handler, mock_web_event_streamer):
        """Test get next event with timeout."""
        # Setup empty client queue
        client_queue = Queue()
        mock_web_event_streamer.client_queues['client_1'] = client_queue
        
        # Should timeout and return None
        sse_handler.event_timeout = 0.1
        event = sse_handler._get_next_event('client_1')
        
        assert event is None
    
    def test_cleanup_client(self, sse_handler, mock_web_event_streamer):
        """Test client cleanup."""
        # Setup client
        sse_handler.active_streams['client_1'] = Mock()
        
        sse_handler._cleanup_client('client_1')
        
        assert 'client_1' not in sse_handler.active_streams
        mock_web_event_streamer.unregister_client.assert_called_once_with('client_1')
