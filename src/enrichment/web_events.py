#!/usr/bin/env python3
"""
Web Events - Real-time Event Streaming to Web Interface

Streams security events to web interface in real-time using Server-Sent Events (SSE)
or WebSocket connections for live monitoring and notifications.
"""

import time
import json
import logging
import threading
from typing import Dict, Any, List, Set, Optional
from queue import Queue, Empty
from datetime import datetime

from src.enrichment.base_enrichment import BaseEnrichment, EnrichmentResult, EnrichmentStatus
from src.communication.events import PipelineEvent, FaceRecognitionEvent, EventType

logger = logging.getLogger(__name__)


class WebEventStreamer:
    """Manages real-time streaming of events to web clients."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize web event streamer.
        
        Args:
            config: Streaming configuration
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.max_connections = config.get('max_connections', 50)
        self.buffer_size = config.get('buffer_size', 100)
        self.heartbeat_interval = config.get('heartbeat_interval', 30.0)
        
        # Event queues for connected clients
        self.client_queues: Dict[str, Queue] = {}
        self.client_lock = threading.RLock()
        
        # Event buffer for new connections
        self.event_buffer: List[Dict[str, Any]] = []
        self.buffer_lock = threading.Lock()
        
        # Statistics
        self.events_streamed = 0
        self.active_connections = 0
        self.total_connections = 0
        
        logger.info(f"Initialized web event streamer (max_connections={self.max_connections})")
    
    def register_client(self, client_id: str) -> bool:
        """
        Register a new web client for event streaming.
        
        Args:
            client_id: Unique client identifier
            
        Returns:
            True if registration successful, False if max connections reached
        """
        with self.client_lock:
            if len(self.client_queues) >= self.max_connections:
                logger.warning(f"Max connections ({self.max_connections}) reached, rejecting client {client_id}")
                return False
            
            if client_id in self.client_queues:
                logger.warning(f"Client {client_id} already registered")
                return True
            
            # Create queue for this client
            self.client_queues[client_id] = Queue(maxsize=self.buffer_size)
            self.active_connections += 1
            self.total_connections += 1
            
            # Send buffered events to new client
            self._send_buffered_events(client_id)
            
            logger.info(f"Registered web client: {client_id} (active={self.active_connections})")
            return True
    
    def unregister_client(self, client_id: str) -> None:
        """
        Unregister a web client.
        
        Args:
            client_id: Client identifier to unregister
        """
        with self.client_lock:
            if client_id in self.client_queues:
                del self.client_queues[client_id]
                self.active_connections = max(0, self.active_connections - 1)
                logger.info(f"Unregistered web client: {client_id} (active={self.active_connections})")
    
    def stream_event(self, event_data: Dict[str, Any]) -> int:
        """
        Stream event to all connected clients.
        
        Args:
            event_data: Event data to stream
            
        Returns:
            Number of clients that received the event
        """
        if not self.enabled:
            return 0
        
        # Add to event buffer
        with self.buffer_lock:
            self.event_buffer.append(event_data)
            if len(self.event_buffer) > self.buffer_size:
                self.event_buffer.pop(0)
        
        # Stream to all connected clients
        delivered_count = 0
        
        with self.client_lock:
            dead_clients = []
            
            for client_id, client_queue in self.client_queues.items():
                try:
                    # Try to add to queue (non-blocking)
                    client_queue.put_nowait(event_data)
                    delivered_count += 1
                except Exception as e:
                    logger.warning(f"Failed to queue event for client {client_id}: {e}")
                    dead_clients.append(client_id)
            
            # Remove dead clients
            for client_id in dead_clients:
                self.unregister_client(client_id)
        
        self.events_streamed += 1
        
        if delivered_count > 0:
            logger.debug(f"Streamed event to {delivered_count} clients")
        
        return delivered_count
    
    def get_events(self, client_id: str, timeout: float = 1.0) -> List[Dict[str, Any]]:
        """
        Get pending events for a client.
        
        Args:
            client_id: Client identifier
            timeout: Maximum time to wait for events
            
        Returns:
            List of pending events
        """
        with self.client_lock:
            if client_id not in self.client_queues:
                logger.warning(f"Client {client_id} not registered")
                return []
            
            client_queue = self.client_queues[client_id]
        
        events = []
        deadline = time.time() + timeout
        
        try:
            # Get first event (blocking with timeout)
            remaining_time = deadline - time.time()
            if remaining_time > 0:
                event = client_queue.get(timeout=remaining_time)
                events.append(event)
            
            # Get any additional queued events (non-blocking)
            while True:
                try:
                    event = client_queue.get_nowait()
                    events.append(event)
                except Empty:
                    break
        except Empty:
            pass  # No events available
        
        return events
    
    def _send_buffered_events(self, client_id: str) -> None:
        """
        Send buffered events to a newly connected client.
        
        Args:
            client_id: Client to send buffered events to
        """
        with self.buffer_lock:
            buffered_events = self.event_buffer.copy()
        
        if not buffered_events:
            return
        
        with self.client_lock:
            if client_id not in self.client_queues:
                return
            
            client_queue = self.client_queues[client_id]
        
        for event in buffered_events:
            try:
                client_queue.put_nowait(event)
            except Exception as e:
                logger.warning(f"Failed to send buffered event to {client_id}: {e}")
                break
        
        logger.debug(f"Sent {len(buffered_events)} buffered events to {client_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        return {
            'enabled': self.enabled,
            'active_connections': self.active_connections,
            'total_connections': self.total_connections,
            'events_streamed': self.events_streamed,
            'buffer_size': len(self.event_buffer),
            'max_connections': self.max_connections
        }


class WebEventsEnrichment(BaseEnrichment):
    """Enrichment processor for web event streaming."""
    
    def __init__(self, config: Dict[str, Any], streamer: Optional[WebEventStreamer] = None):
        """
        Initialize web events enrichment processor.
        
        Args:
            config: Configuration dictionary
            streamer: Optional WebEventStreamer instance (creates one if not provided)
        """
        # Set default priority (run late in pipeline)
        if 'priority' not in config:
            config['priority'] = 8
        
        super().__init__(config)
        
        # Initialize or use provided streamer
        self.streamer = streamer or WebEventStreamer(config.get('streaming', {}))
        
        # Configuration
        self.stream_all_events = config.get('stream_all_events', True)
        self.stream_event_types = set(config.get('stream_event_types', []))
        
        logger.info(f"Initialized {self.name} enrichment processor")
    
    def can_process(self, event: Any) -> bool:
        """Check if this enrichment can process the given event."""
        if not self.streamer.enabled:
            return False
        
        # Can process any PipelineEvent
        if not isinstance(event, PipelineEvent):
            return False
        
        # Check event type filter
        if not self.stream_all_events and self.stream_event_types:
            event_type = event.event_type.name if hasattr(event, 'event_type') else None
            return event_type in self.stream_event_types
        
        return True
    
    def enrich(self, event: PipelineEvent) -> EnrichmentResult:
        """
        Enrich event by streaming it to web interface.
        
        Args:
            event: Event to stream
            
        Returns:
            EnrichmentResult with streaming status
        """
        start_time = time.time()
        
        try:
            # Prepare event for web streaming
            web_event = self._prepare_web_event(event)
            
            # Stream to connected clients
            delivered_count = self.streamer.stream_event(web_event)
            
            processing_time = time.time() - start_time
            
            return EnrichmentResult(
                success=True,
                enriched_data={
                    'web_streaming': {
                        'streamed': True,
                        'delivered_to': delivered_count,
                        'timestamp': datetime.now().isoformat()
                    }
                },
                processing_time=processing_time,
                processor_name=self.name,
                status=EnrichmentStatus.SUCCESS,
                metadata={
                    'delivered_count': delivered_count,
                    'active_connections': self.streamer.active_connections
                }
            )
            
        except Exception as e:
            logger.error(f"Web event streaming failed: {e}", exc_info=True)
            return EnrichmentResult(
                success=False,
                enriched_data={},
                processing_time=time.time() - start_time,
                processor_name=self.name,
                error_message=str(e),
                status=EnrichmentStatus.FAILED
            )
    
    def _prepare_web_event(self, event: PipelineEvent) -> Dict[str, Any]:
        """
        Prepare event data for web streaming.
        
        Args:
            event: Event to prepare
            
        Returns:
            Web-safe event data dictionary
        """
        # Convert event to dictionary
        event_dict = self._event_to_dict(event)
        
        # Sanitize for web consumption
        web_event = {
            'event_id': event_dict.get('event_id'),
            'event_type': event_dict.get('event_type'),
            'timestamp': event_dict.get('timestamp'),
            'source': event_dict.get('source'),
            'priority': event_dict.get('priority'),
            'data': self._sanitize_data(event_dict.get('data', {}))
        }
        
        # Add face recognition specific data
        if isinstance(event, FaceRecognitionEvent):
            web_event['recognition_summary'] = {
                'total': len(event.recognitions) if hasattr(event, 'recognitions') else 0,
                'known': getattr(event, 'known_count', 0),
                'unknown': getattr(event, 'unknown_count', 0),
                'blacklisted': getattr(event, 'blacklisted_count', 0)
            }
        
        return web_event
    
    def _event_to_dict(self, event: PipelineEvent) -> Dict[str, Any]:
        """Convert event to dictionary."""
        if hasattr(event, 'to_dict'):
            return event.to_dict()
        
        # Fallback: extract attributes
        return {
            'event_id': getattr(event, 'event_id', None),
            'event_type': getattr(event, 'event_type', None).name if hasattr(event, 'event_type') else None,
            'timestamp': getattr(event, 'timestamp', time.time()),
            'source': getattr(event, 'source', None),
            'priority': getattr(event, 'priority', None).name if hasattr(event, 'priority') else None,
            'data': getattr(event, 'data', {})
        }
    
    def _sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize event data for web consumption.
        
        Removes binary data, large objects, and sensitive information.
        
        Args:
            data: Data to sanitize
            
        Returns:
            Sanitized data dictionary
        """
        if not isinstance(data, dict):
            return {}
        
        sanitized = {}
        
        for key, value in data.items():
            # Skip binary data and large objects
            if isinstance(value, bytes):
                sanitized[key] = f"<binary data: {len(value)} bytes>"
            elif hasattr(value, '__array__'):  # NumPy arrays
                sanitized[key] = f"<array: {value.shape if hasattr(value, 'shape') else 'unknown'}>"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_data(value)
            elif isinstance(value, list) and len(value) > 10:
                sanitized[key] = f"<list: {len(value)} items>"
            else:
                # Keep simple types
                try:
                    json.dumps(value)  # Test if JSON serializable
                    sanitized[key] = value
                except (TypeError, ValueError):
                    sanitized[key] = str(value)
        
        return sanitized
    
    def get_dependencies(self) -> List[str]:
        """Depends on metadata enrichment."""
        return ['MetadataEnrichment']
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get enrichment and streaming metrics."""
        base_metrics = super().get_metrics()
        streaming_metrics = self.streamer.get_stats()
        
        return {
            **base_metrics,
            'streaming': streaming_metrics
        }
