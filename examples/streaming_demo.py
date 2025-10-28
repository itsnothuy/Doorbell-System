#!/usr/bin/env python3
"""
Streaming Integration Demo

Demonstrates the real-time streaming capabilities of the doorbell system.
This script shows how SSE, WebSocket, and video streaming work together.
"""

import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_sse_streaming():
    """Demonstrate SSE streaming functionality."""
    from src.enrichment.web_events import WebEventStreamer
    from src.streaming.sse_handler import SSEHandler
    
    logger.info("=== SSE Streaming Demo ===")
    
    # Create web event streamer
    config = {
        'enabled': True,
        'max_connections': 50,
        'buffer_size': 100,
        'heartbeat_interval': 30.0
    }
    web_event_streamer = WebEventStreamer(config)
    
    # Create SSE handler
    sse_handler = SSEHandler(web_event_streamer)
    
    # Register a client
    client_id = 'demo_client_1'
    registered = web_event_streamer.register_client(client_id)
    logger.info(f"Client registered: {registered}")
    
    # Stream some test events
    for i in range(3):
        event = {
            'event_id': f'event_{i}',
            'event_type': 'doorbell_pressed',
            'timestamp': time.time(),
            'data': {
                'location': 'front_door',
                'index': i
            }
        }
        delivered = web_event_streamer.stream_event(event)
        logger.info(f"Event {i} delivered to {delivered} clients")
    
    # Get stats
    stats = web_event_streamer.get_stats()
    logger.info(f"Streaming stats: {stats}")
    
    # Cleanup
    web_event_streamer.unregister_client(client_id)
    logger.info("Demo complete\n")


def demo_websocket_handler():
    """Demonstrate WebSocket handler functionality."""
    from unittest.mock import Mock
    from src.streaming.websocket_handler import WebSocketHandler
    
    logger.info("=== WebSocket Handler Demo ===")
    
    # Create mock SocketIO
    mock_socketio = Mock()
    mock_socketio.on = Mock(return_value=lambda f: f)
    mock_socketio.emit = Mock()
    
    # Create WebSocket handler
    ws_handler = WebSocketHandler(mock_socketio)
    
    # Demonstrate broadcasting
    ws_handler.broadcast_event(
        'face_detected',
        {
            'person': 'John Doe',
            'confidence': 0.95,
            'timestamp': time.time()
        }
    )
    logger.info("Event broadcasted via WebSocket")
    
    # Check permissions
    test_client_id = 'test_client'
    ws_handler.connected_clients[test_client_id] = {
        'permissions': {'system_control', 'view_events'}
    }
    
    has_permission = ws_handler._check_client_permissions(test_client_id, 'system_control')
    logger.info(f"Client has system_control permission: {has_permission}")
    
    logger.info("Demo complete\n")


def demo_video_streaming():
    """Demonstrate video streaming configuration."""
    from src.streaming.video_streamer import VideoStreamer
    from unittest.mock import Mock
    import numpy as np
    
    logger.info("=== Video Streaming Demo ===")
    
    # Create mock camera handler
    mock_camera = Mock()
    mock_camera.capture_image = Mock(return_value=np.zeros((480, 640, 3), dtype=np.uint8))
    
    # Create video streamer
    video_streamer = VideoStreamer(mock_camera)
    
    # Display quality settings
    logger.info("Quality settings:")
    for quality, settings in video_streamer.quality_settings.items():
        logger.info(f"  {quality}: {settings['width']}x{settings['height']} @ {settings['fps']}fps, quality={settings['quality']}")
    
    # Test stream URL generation
    stream_url = f"/stream/video/demo_client?quality=medium"
    logger.info(f"Stream URL: {stream_url}")
    
    logger.info("Demo complete\n")


def demo_complete_integration():
    """Demonstrate complete streaming integration."""
    from src.enrichment.web_events import WebEventStreamer, WebEventsEnrichment
    from src.communication.events import PipelineEvent, EventType
    
    logger.info("=== Complete Integration Demo ===")
    
    # Create web event streamer
    config = {
        'enabled': True,
        'max_connections': 50,
        'stream_all_events': True
    }
    web_event_streamer = WebEventStreamer(config)
    
    # Create enrichment processor
    enrichment_config = {
        'priority': 8,
        'streaming': config
    }
    web_enrichment = WebEventsEnrichment(enrichment_config, web_event_streamer)
    
    # Register a test client
    client_id = 'integration_test_client'
    web_event_streamer.register_client(client_id)
    
    # Create and process a test event
    test_event = PipelineEvent(
        event_type=EventType.DOORBELL_PRESSED,
        data={
            'timestamp': time.time(),
            'location': 'front_door',
            'triggered_by': 'button'
        },
        source='demo'
    )
    
    # Check if enrichment can process
    can_process = web_enrichment.can_process(test_event)
    logger.info(f"Enrichment can process event: {can_process}")
    
    # Process event
    if can_process:
        result = web_enrichment.enrich(test_event)
        logger.info(f"Enrichment result: success={result.success}, status={result.status}")
        logger.info(f"Event delivered to {result.metadata.get('delivered_count', 0)} clients")
    
    # Get metrics
    metrics = web_enrichment.get_metrics()
    logger.info(f"Enrichment metrics: {metrics}")
    
    # Cleanup
    web_event_streamer.unregister_client(client_id)
    logger.info("Demo complete\n")


def main():
    """Run all demos."""
    logger.info("Starting Streaming Integration Demos\n")
    
    try:
        demo_sse_streaming()
        demo_websocket_handler()
        demo_video_streaming()
        demo_complete_integration()
        
        logger.info("✅ All demos completed successfully!")
        logger.info("\nTo use streaming in production:")
        logger.info("1. Ensure Flask-SocketIO is installed: pip install flask-socketio")
        logger.info("2. Include streaming scripts in your HTML: event-stream.js, websocket-client.js")
        logger.info("3. Initialize streaming clients in your frontend JavaScript")
        logger.info("4. Access SSE endpoint: /stream/events")
        logger.info("5. Access video stream: /stream/video/<client_id>")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
