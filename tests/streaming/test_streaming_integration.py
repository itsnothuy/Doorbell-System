#!/usr/bin/env python3
"""
Integration Tests for Streaming Components

Tests the complete streaming integration between SSE, WebSocket, and web_events.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from queue import Queue

from src.streaming.sse_handler import SSEHandler
from src.streaming.websocket_handler import WebSocketHandler
from src.enrichment.web_events import WebEventStreamer


class TestStreamingIntegration:
    """Integration tests for streaming components."""
    
    @pytest.fixture
    def web_event_streamer(self):
        """Create web event streamer instance."""
        config = {
            'enabled': True,
            'max_connections': 10,
            'buffer_size': 50,
            'heartbeat_interval': 30.0
        }
        return WebEventStreamer(config)
    
    @pytest.fixture
    def sse_handler(self, web_event_streamer):
        """Create SSE handler with web event streamer."""
        return SSEHandler(web_event_streamer)
    
    def test_sse_handler_integration_with_web_event_streamer(self, sse_handler, web_event_streamer):
        """Test SSE handler integrates correctly with web event streamer."""
        assert sse_handler.web_event_streamer == web_event_streamer
        
        # Test client registration
        client_id = 'test_client_1'
        registered = web_event_streamer.register_client(client_id)
        assert registered is True
        assert client_id in web_event_streamer.client_queues
        
        # Test event streaming
        test_event = {
            'event_id': '123',
            'event_type': 'test_event',
            'data': {'message': 'test'}
        }
        delivered = web_event_streamer.stream_event(test_event)
        assert delivered == 1
        
        # Test cleanup
        web_event_streamer.unregister_client(client_id)
        assert client_id not in web_event_streamer.client_queues
    
    def test_sse_event_formatting(self, sse_handler):
        """Test SSE event formatting matches spec."""
        event_str = sse_handler._format_sse_event(
            event='test',
            data={'key': 'value'},
            event_id='123'
        )
        
        # Check SSE format
        lines = event_str.strip().split('\n')
        assert lines[0] == 'id: 123'
        assert lines[1] == 'event: test'
        assert lines[2].startswith('data: ')
        assert event_str.endswith('\n\n')
    
    def test_websocket_handler_initialization(self):
        """Test WebSocket handler initializes correctly."""
        mock_socketio = Mock()
        mock_socketio.on = Mock(return_value=lambda f: f)
        
        handler = WebSocketHandler(mock_socketio)
        
        assert handler.socketio == mock_socketio
        assert len(handler.connected_clients) == 0
    
    def test_max_connections_limit(self, web_event_streamer):
        """Test max connections limit is enforced."""
        max_connections = web_event_streamer.max_connections
        
        # Register up to max
        clients = []
        for i in range(max_connections):
            client_id = f'client_{i}'
            registered = web_event_streamer.register_client(client_id)
            assert registered is True
            clients.append(client_id)
        
        # Try to register one more
        overflow_client = 'client_overflow'
        registered = web_event_streamer.register_client(overflow_client)
        assert registered is False
        
        # Cleanup
        for client_id in clients:
            web_event_streamer.unregister_client(client_id)
    
    def test_event_buffering(self, web_event_streamer):
        """Test event buffering for new clients."""
        # Stream events before client connects
        events = []
        for i in range(5):
            event = {
                'event_id': f'event_{i}',
                'event_type': 'test',
                'data': {'index': i}
            }
            web_event_streamer.stream_event(event)
            events.append(event)
        
        # Register new client
        client_id = 'late_client'
        web_event_streamer.register_client(client_id)
        
        # Client should receive buffered events
        client_queue = web_event_streamer.client_queues[client_id]
        assert client_queue.qsize() > 0
        
        # Cleanup
        web_event_streamer.unregister_client(client_id)
    
    def test_streaming_stats(self, web_event_streamer):
        """Test streaming statistics tracking."""
        # Register clients
        web_event_streamer.register_client('client_1')
        web_event_streamer.register_client('client_2')
        
        # Stream events
        for i in range(3):
            web_event_streamer.stream_event({'event_id': f'event_{i}'})
        
        # Check stats
        stats = web_event_streamer.get_stats()
        assert stats['active_connections'] == 2
        assert stats['events_streamed'] == 3
        assert stats['total_connections'] == 2
        
        # Cleanup
        web_event_streamer.unregister_client('client_1')
        web_event_streamer.unregister_client('client_2')
