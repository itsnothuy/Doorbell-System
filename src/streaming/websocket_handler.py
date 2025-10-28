#!/usr/bin/env python3
"""
WebSocket Handler

Implements WebSocket endpoints for bidirectional real-time communication
with web clients for interactive controls and live updates.
"""

import json
import logging
import time
import threading
from typing import Dict, Any, Optional, Callable
from flask_socketio import SocketIO, emit, disconnect, join_room, leave_room

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """WebSocket handler for bidirectional communication."""
    
    def __init__(self, socketio: SocketIO, pipeline_orchestrator=None):
        """
        Initialize WebSocket handler.
        
        Args:
            socketio: Flask-SocketIO instance
            pipeline_orchestrator: Pipeline orchestrator for system control (optional)
        """
        self.socketio = socketio
        self.pipeline_orchestrator = pipeline_orchestrator
        
        # Connected clients
        self.connected_clients: Dict[str, Dict[str, Any]] = {}
        self.client_lock = threading.RLock()
        
        # Setup event handlers
        self._setup_event_handlers()
        
        logger.info("WebSocket Handler initialized")
    
    def _setup_event_handlers(self) -> None:
        """Setup WebSocket event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            from flask import request
            client_id = request.sid
            
            with self.client_lock:
                self.connected_clients[client_id] = {
                    'connected_at': time.time(),
                    'subscriptions': set(),
                    'permissions': self._get_client_permissions()
                }
            
            # Join default room
            join_room('general')
            
            # Send welcome message
            emit('connected', {
                'client_id': client_id,
                'server_capabilities': [
                    'system_control',
                    'live_events',
                    'video_stream',
                    'configuration'
                ]
            })
            
            logger.info(f"WebSocket client connected: {client_id}")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            from flask import request
            client_id = request.sid
            
            with self.client_lock:
                if client_id in self.connected_clients:
                    del self.connected_clients[client_id]
            
            logger.info(f"WebSocket client disconnected: {client_id}")
        
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            """Handle subscription to event types."""
            from flask import request
            client_id = request.sid
            event_types = data.get('event_types', [])
            
            with self.client_lock:
                if client_id in self.connected_clients:
                    self.connected_clients[client_id]['subscriptions'].update(event_types)
            
            # Join relevant rooms
            for event_type in event_types:
                join_room(f"events_{event_type}")
            
            emit('subscribed', {'event_types': event_types})
        
        @self.socketio.on('system_command')
        def handle_system_command(data):
            """Handle system control commands."""
            from flask import request
            client_id = request.sid
            command = data.get('command')
            params = data.get('params', {})
            
            # Check permissions
            if not self._check_client_permissions(client_id, 'system_control'):
                emit('error', {'message': 'Insufficient permissions'})
                return
            
            try:
                result = self._execute_system_command(command, params)
                emit('command_result', {
                    'command': command,
                    'success': True,
                    'result': result
                })
            except Exception as e:
                logger.error(f"System command error: {e}", exc_info=True)
                emit('command_result', {
                    'command': command,
                    'success': False,
                    'error': str(e)
                })
        
        @self.socketio.on('request_video_stream')
        def handle_video_stream_request(data):
            """Handle video stream requests."""
            from flask import request
            client_id = request.sid
            quality = data.get('quality', 'medium')
            
            try:
                stream_url = self._start_video_stream(client_id, quality)
                emit('video_stream_ready', {
                    'stream_url': stream_url,
                    'quality': quality
                })
            except Exception as e:
                logger.error(f"Video stream request error: {e}", exc_info=True)
                emit('error', {'message': f'Video stream failed: {e}'})
    
    def broadcast_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast event to subscribed clients."""
        room = f"events_{event_type}"
        self.socketio.emit('live_event', {
            'event_type': event_type,
            'data': data,
            'timestamp': time.time()
        }, room=room)
    
    def _check_client_permissions(self, client_id: str, permission: str) -> bool:
        """Check if client has required permission."""
        with self.client_lock:
            client_info = self.connected_clients.get(client_id)
            if not client_info:
                return False
            
            return permission in client_info.get('permissions', set())
    
    def _get_client_permissions(self) -> set:
        """Get permissions for new client (implement based on your auth system)."""
        # For now, grant all permissions - implement proper auth as needed
        return {'system_control', 'view_events', 'video_stream'}
    
    def _execute_system_command(self, command: str, params: Dict[str, Any]) -> Any:
        """Execute system control command."""
        if command == 'get_system_status':
            # Return basic status information
            return {
                'status': 'running',
                'timestamp': time.time()
            }
        elif command == 'trigger_doorbell':
            # Simulate doorbell trigger
            return {'triggered': True, 'timestamp': time.time()}
        else:
            raise Exception(f"Unknown command: {command}")
    
    def _start_video_stream(self, client_id: str, quality: str) -> str:
        """Start video stream for client."""
        # Return stream endpoint URL
        return f"/stream/video/{client_id}?quality={quality}"
