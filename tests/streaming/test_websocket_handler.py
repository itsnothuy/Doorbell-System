#!/usr/bin/env python3
"""
Tests for WebSocket Handler

Tests the WebSocket bidirectional communication implementation.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch

from src.streaming.websocket_handler import WebSocketHandler


class TestWebSocketHandler:
    """Test suite for WebSocket Handler."""
    
    @pytest.fixture
    def mock_socketio(self):
        """Create mock SocketIO instance."""
        socketio = Mock()
        socketio.on = Mock(return_value=lambda f: f)
        socketio.emit = Mock()
        return socketio
    
    @pytest.fixture
    def websocket_handler(self, mock_socketio):
        """Create WebSocket handler instance."""
        return WebSocketHandler(mock_socketio)
    
    def test_initialization(self, websocket_handler, mock_socketio):
        """Test WebSocket handler initializes correctly."""
        assert websocket_handler.socketio == mock_socketio
        assert len(websocket_handler.connected_clients) == 0
        assert websocket_handler.pipeline_orchestrator is None
    
    def test_initialization_with_orchestrator(self, mock_socketio):
        """Test initialization with pipeline orchestrator."""
        mock_orchestrator = Mock()
        handler = WebSocketHandler(mock_socketio, mock_orchestrator)
        
        assert handler.pipeline_orchestrator == mock_orchestrator
    
    def test_get_client_permissions(self, websocket_handler):
        """Test getting client permissions."""
        permissions = websocket_handler._get_client_permissions()
        
        assert 'system_control' in permissions
        assert 'view_events' in permissions
        assert 'video_stream' in permissions
    
    def test_check_client_permissions_no_client(self, websocket_handler):
        """Test checking permissions for non-existent client."""
        result = websocket_handler._check_client_permissions('non_existent', 'system_control')
        
        assert result is False
    
    def test_check_client_permissions_success(self, websocket_handler):
        """Test checking permissions for existing client."""
        # Add client
        websocket_handler.connected_clients['client_1'] = {
            'permissions': {'system_control', 'view_events'}
        }
        
        result = websocket_handler._check_client_permissions('client_1', 'system_control')
        assert result is True
        
        result = websocket_handler._check_client_permissions('client_1', 'invalid_perm')
        assert result is False
    
    def test_execute_system_command_get_status(self, websocket_handler):
        """Test executing get_system_status command."""
        result = websocket_handler._execute_system_command('get_system_status', {})
        
        assert 'status' in result
        assert 'timestamp' in result
        assert result['status'] == 'running'
    
    def test_execute_system_command_trigger_doorbell(self, websocket_handler):
        """Test executing trigger_doorbell command."""
        result = websocket_handler._execute_system_command('trigger_doorbell', {})
        
        assert 'triggered' in result
        assert result['triggered'] is True
    
    def test_execute_system_command_unknown(self, websocket_handler):
        """Test executing unknown command."""
        with pytest.raises(Exception) as exc_info:
            websocket_handler._execute_system_command('unknown_command', {})
        
        assert 'Unknown command' in str(exc_info.value)
    
    def test_start_video_stream(self, websocket_handler):
        """Test starting video stream."""
        stream_url = websocket_handler._start_video_stream('client_1', 'medium')
        
        assert '/stream/video/client_1' in stream_url
        assert 'quality=medium' in stream_url
    
    def test_broadcast_event(self, websocket_handler, mock_socketio):
        """Test broadcasting event to clients."""
        event_type = 'doorbell_pressed'
        data = {'timestamp': time.time(), 'location': 'front_door'}
        
        websocket_handler.broadcast_event(event_type, data)
        
        mock_socketio.emit.assert_called_once()
        call_args = mock_socketio.emit.call_args
        assert call_args[0][0] == 'live_event'
        assert call_args[0][1]['event_type'] == event_type
        assert call_args[0][1]['data'] == data
