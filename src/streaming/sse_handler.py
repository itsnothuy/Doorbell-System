#!/usr/bin/env python3
"""
Server-Sent Events Handler

Implements SSE endpoints for real-time event streaming to web clients
with efficient connection management and event broadcasting.
"""

import json
import time
import logging
import threading
from typing import Dict, Any, Iterator, Optional
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class SSEHandler:
    """Server-Sent Events handler for real-time streaming."""
    
    def __init__(self, web_event_streamer):
        """
        Initialize SSE handler.
        
        Args:
            web_event_streamer: Web event streaming system instance
        """
        self.web_event_streamer = web_event_streamer
        self.active_streams: Dict[str, Queue] = {}
        self.stream_lock = threading.RLock()
        
        # Configuration
        self.heartbeat_interval = 30.0  # seconds
        self.max_queue_size = 100
        self.event_timeout = 5.0  # seconds
        
        logger.info("SSE Handler initialized")
    
    def create_event_stream(self, client_id: str) -> Iterator[str]:
        """
        Create SSE event stream for a client.
        
        Args:
            client_id: Unique client identifier
            
        Yields:
            SSE formatted event strings
        """
        try:
            # Register client with web event streamer
            if not self.web_event_streamer.register_client(client_id):
                yield self._format_sse_event(
                    event='error',
                    data={'error': 'Max connections reached'}
                )
                return
            
            # Create client queue
            client_queue = Queue(maxsize=self.max_queue_size)
            with self.stream_lock:
                self.active_streams[client_id] = client_queue
            
            # Send initial connection event
            yield self._format_sse_event(
                event='connected',
                data={
                    'client_id': client_id,
                    'server_time': time.time(),
                    'capabilities': ['events', 'system_status', 'controls']
                }
            )
            
            # Start streaming loop
            last_heartbeat = time.time()
            
            while True:
                try:
                    # Check for events from web event streamer
                    event_data = self._get_next_event(client_id)
                    
                    if event_data:
                        yield self._format_sse_event(
                            event=event_data.get('type', 'event'),
                            data=event_data
                        )
                    
                    # Send heartbeat if needed
                    current_time = time.time()
                    if current_time - last_heartbeat > self.heartbeat_interval:
                        yield self._format_sse_event(
                            event='heartbeat',
                            data={'timestamp': current_time}
                        )
                        last_heartbeat = current_time
                    
                    # Brief sleep to prevent busy waiting
                    time.sleep(0.1)
                    
                except GeneratorExit:
                    # Client disconnected
                    break
                except Exception as e:
                    logger.error(f"SSE streaming error for {client_id}: {e}")
                    yield self._format_sse_event(
                        event='error',
                        data={'error': str(e)}
                    )
                    break
        
        finally:
            self._cleanup_client(client_id)
    
    def create_system_status_stream(self, client_id: str) -> Iterator[str]:
        """
        Create SSE stream for system status updates.
        
        Args:
            client_id: Unique client identifier
            
        Yields:
            SSE formatted system status events
        """
        try:
            yield self._format_sse_event(
                event='status_connected',
                data={'client_id': client_id}
            )
            
            last_status_check = 0
            status_interval = 5.0  # Check status every 5 seconds
            
            while True:
                current_time = time.time()
                
                if current_time - last_status_check >= status_interval:
                    # Get system status
                    status_data = self._get_system_status()
                    
                    yield self._format_sse_event(
                        event='system_status',
                        data=status_data
                    )
                    
                    last_status_check = current_time
                
                time.sleep(1.0)
                
        except GeneratorExit:
            pass
        except Exception as e:
            logger.error(f"System status streaming error for {client_id}: {e}")
            yield self._format_sse_event(
                event='error',
                data={'error': str(e)}
            )
    
    def _get_next_event(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get next event for client from web event streamer."""
        # This integrates with the existing WebEventStreamer
        client_queue = self.web_event_streamer.client_queues.get(client_id)
        if not client_queue:
            return None
        
        try:
            return client_queue.get(timeout=self.event_timeout)
        except Empty:
            return None
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        # Get status from web event streamer
        stats = self.web_event_streamer.get_stats()
        
        return {
            'timestamp': time.time(),
            'streaming_status': 'running',
            'active_connections': stats.get('active_connections', 0),
            'events_streamed': stats.get('events_streamed', 0),
            'buffer_size': stats.get('buffer_size', 0)
        }
    
    def _format_sse_event(self, event: str, data: Dict[str, Any], 
                         event_id: Optional[str] = None) -> str:
        """Format data as SSE event string."""
        lines = []
        
        if event_id:
            lines.append(f"id: {event_id}")
        
        lines.append(f"event: {event}")
        
        # Format data as JSON
        json_data = json.dumps(data, separators=(',', ':'))
        lines.append(f"data: {json_data}")
        
        # Add empty line to terminate event
        lines.append("")
        
        return "\n".join(lines) + "\n"
    
    def _cleanup_client(self, client_id: str) -> None:
        """Clean up client resources."""
        with self.stream_lock:
            if client_id in self.active_streams:
                del self.active_streams[client_id]
        
        self.web_event_streamer.unregister_client(client_id)
        logger.info(f"Cleaned up SSE client: {client_id}")
