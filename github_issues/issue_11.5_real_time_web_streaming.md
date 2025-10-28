# Issue #11.5: Real-Time Web Streaming Integration with SSE and WebSocket Support

## 📋 **Overview**

Complete the real-time web streaming architecture by implementing Server-Sent Events (SSE) endpoints and WebSocket integration to connect the existing web event streaming system (`src/enrichment/web_events.py`) with the web interface for live security monitoring. This final integration component enables real-time event updates, live video streaming, and interactive security monitoring in the production-ready Frigate-inspired doorbell system.

## 🎯 **Objectives**

### **Primary Goals**
1. **Real-Time Event Streaming**: Implement SSE endpoints for live security event updates
2. **WebSocket Integration**: Add WebSocket support for bidirectional communication
3. **Live Video Streaming**: Real-time camera feed streaming to web interface
4. **Interactive Controls**: Real-time system control and configuration updates
5. **Performance Optimization**: Efficient streaming with minimal resource usage

### **Success Criteria**
- Real-time event updates with <500ms latency from trigger to web display
- Support for 50+ concurrent web clients without performance degradation
- Live video streaming at 15+ FPS to web interface
- Interactive controls with immediate system response
- Zero event loss during streaming operations
- Graceful handling of client connections and disconnections

## 🔍 **Gap Analysis**

### **✅ What's Already Implemented**
- **Web Event Streamer**: `src/enrichment/web_events.py` (413 lines) - Complete streaming infrastructure
- **Web Interface Backend**: `src/web_interface.py` (632 lines) - Flask application with API endpoints
- **Dashboard Frontend**: `templates/dashboard.html` - Complete UI with event display components
- **Event Processing**: Full pipeline integration with enrichment system
- **Storage Integration**: Event persistence and retrieval systems

### **❌ What's Missing**
- **SSE Endpoints**: `/stream/events`, `/stream/system-status` routes
- **WebSocket Handlers**: Bidirectional communication for controls
- **Live Video Endpoints**: `/stream/video`, `/stream/camera` routes
- **Frontend Integration**: JavaScript EventSource and WebSocket clients
- **Connection Management**: Client registration, heartbeat, cleanup
- **Streaming Optimization**: Compression, buffering, quality adaptation

## 🔧 **Implementation Specifications**

### **Files to Create/Modify**

#### **New Files**
```
src/streaming/                           # Real-time streaming module
    __init__.py
    sse_handler.py                       # Server-Sent Events implementation
    websocket_handler.py                 # WebSocket handler
    video_streamer.py                    # Live video streaming
    stream_manager.py                    # Streaming connection management
    
static/js/                              # Frontend streaming clients
    event-stream.js                      # SSE client implementation
    websocket-client.js                  # WebSocket client
    video-stream.js                      # Live video streaming client
    
tests/streaming/                        # Streaming tests
    test_sse_handler.py                 # SSE endpoint tests
    test_websocket_handler.py           # WebSocket tests
    test_stream_integration.py          # Integration tests
```

#### **Modified Files**
```
src/web_interface.py                    # Add streaming routes
templates/dashboard.html                # Integrate streaming clients
config/pipeline_config.py              # Streaming configuration
requirements.txt                       # Add streaming dependencies
```

### **Core Implementation**

#### **Server-Sent Events Handler**
```python
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
from flask import Response, request, current_app
from queue import Queue, Empty

from src.enrichment.web_events import WebEventStreamer
from src.communication.events import PipelineEvent, EventType

logger = logging.getLogger(__name__)


class SSEHandler:
    """Server-Sent Events handler for real-time streaming."""
    
    def __init__(self, web_event_streamer: WebEventStreamer):
        """
        Initialize SSE handler.
        
        Args:
            web_event_streamer: Web event streaming system
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
                    # Get system status (implement based on your system)
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
        # This integrates with your existing WebEventStreamer
        client_queue = self.web_event_streamer.client_queues.get(client_id)
        if not client_queue:
            return None
        
        try:
            return client_queue.get(timeout=self.event_timeout)
        except Empty:
            return None
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        # Implement based on your pipeline orchestrator
        return {
            'timestamp': time.time(),
            'pipeline_status': 'running',  # Get from orchestrator
            'active_workers': 5,  # Get from orchestrator
            'events_processed': 100,  # Get from event processor
            'memory_usage': 150.5,  # MB
            'cpu_usage': 25.3  # %
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
        
        return "\\n".join(lines) + "\\n"
    
    def _cleanup_client(self, client_id: str) -> None:
        """Clean up client resources."""
        with self.stream_lock:
            if client_id in self.active_streams:
                del self.active_streams[client_id]
        
        self.web_event_streamer.unregister_client(client_id)
        logger.info(f"Cleaned up SSE client: {client_id}")
```

#### **WebSocket Handler**
```python
#!/usr/bin/env python3
"""
WebSocket Handler

Implements WebSocket endpoints for bidirectional real-time communication
with web clients for interactive controls and live updates.
"""

import json
import logging
import threading
from typing import Dict, Any, Callable, Optional
from flask_socketio import SocketIO, emit, disconnect, join_room, leave_room

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """WebSocket handler for bidirectional communication."""
    
    def __init__(self, socketio: SocketIO, pipeline_orchestrator=None):
        """
        Initialize WebSocket handler.
        
        Args:
            socketio: Flask-SocketIO instance
            pipeline_orchestrator: Pipeline orchestrator for system control
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
            client_id = request.sid
            
            with self.client_lock:
                if client_id in self.connected_clients:
                    del self.connected_clients[client_id]
            
            logger.info(f"WebSocket client disconnected: {client_id}")
        
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            """Handle subscription to event types."""
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
                emit('command_result', {
                    'command': command,
                    'success': False,
                    'error': str(e)
                })
        
        @self.socketio.on('request_video_stream')
        def handle_video_stream_request(data):
            """Handle video stream requests."""
            client_id = request.sid
            quality = data.get('quality', 'medium')
            
            try:
                stream_url = self._start_video_stream(client_id, quality)
                emit('video_stream_ready', {
                    'stream_url': stream_url,
                    'quality': quality
                })
            except Exception as e:
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
        if not self.pipeline_orchestrator:
            raise Exception("Pipeline orchestrator not available")
        
        if command == 'trigger_doorbell':
            # Simulate doorbell press
            return self.pipeline_orchestrator.trigger_doorbell_simulation()
        elif command == 'get_system_status':
            return self.pipeline_orchestrator.get_system_status()
        elif command == 'restart_pipeline':
            return self.pipeline_orchestrator.restart_pipeline()
        else:
            raise Exception(f"Unknown command: {command}")
    
    def _start_video_stream(self, client_id: str, quality: str) -> str:
        """Start video stream for client."""
        # Return stream endpoint URL
        return f"/stream/video/{client_id}?quality={quality}"
```

#### **Live Video Streaming**
```python
#!/usr/bin/env python3
"""
Live Video Streaming

Implements real-time video streaming endpoints for web interface
with quality adaptation and efficient frame delivery.
"""

import cv2
import time
import logging
import threading
from typing import Dict, Any, Iterator, Optional
from flask import Response, request

from src.hardware.camera_handler import CameraHandler

logger = logging.getLogger(__name__)


class VideoStreamer:
    """Live video streaming for web interface."""
    
    def __init__(self, camera_handler: CameraHandler):
        """
        Initialize video streamer.
        
        Args:
            camera_handler: Camera handler instance
        """
        self.camera_handler = camera_handler
        self.active_streams: Dict[str, bool] = {}
        self.stream_lock = threading.RLock()
        
        # Quality settings
        self.quality_settings = {
            'low': {'width': 320, 'height': 240, 'fps': 10, 'quality': 70},
            'medium': {'width': 640, 'height': 480, 'fps': 15, 'quality': 80},
            'high': {'width': 1280, 'height': 720, 'fps': 20, 'quality': 90}
        }
        
        logger.info("Video Streamer initialized")
    
    def create_video_stream(self, client_id: str, quality: str = 'medium') -> Iterator[bytes]:
        """
        Create video stream for client.
        
        Args:
            client_id: Unique client identifier
            quality: Video quality setting
            
        Yields:
            MJPEG video frames
        """
        try:
            # Register client stream
            with self.stream_lock:
                self.active_streams[client_id] = True
            
            # Get quality settings
            settings = self.quality_settings.get(quality, self.quality_settings['medium'])
            frame_interval = 1.0 / settings['fps']
            
            logger.info(f"Starting video stream for {client_id} (quality: {quality})")
            
            last_frame_time = 0
            
            while self.active_streams.get(client_id, False):
                current_time = time.time()
                
                # Respect frame rate limit
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.01)
                    continue
                
                # Capture frame
                frame = self.camera_handler.capture_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Resize frame based on quality
                resized_frame = cv2.resize(
                    frame,
                    (settings['width'], settings['height'])
                )
                
                # Encode as JPEG
                ret, jpeg_frame = cv2.imencode(
                    '.jpg',
                    resized_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, settings['quality']]
                )
                
                if ret:
                    # Yield frame in MJPEG format
                    yield (b'--frame\\r\\n'
                           b'Content-Type: image/jpeg\\r\\n\\r\\n' +
                           jpeg_frame.tobytes() + b'\\r\\n')
                    
                    last_frame_time = current_time
                else:
                    logger.warning("Failed to encode video frame")
        
        except Exception as e:
            logger.error(f"Video streaming error for {client_id}: {e}")
        
        finally:
            self._cleanup_stream(client_id)
    
    def stop_stream(self, client_id: str) -> None:
        """Stop video stream for client."""
        with self.stream_lock:
            if client_id in self.active_streams:
                self.active_streams[client_id] = False
    
    def _cleanup_stream(self, client_id: str) -> None:
        """Clean up stream resources."""
        with self.stream_lock:
            if client_id in self.active_streams:
                del self.active_streams[client_id]
        
        logger.info(f"Cleaned up video stream for {client_id}")
```

### **Web Interface Integration**

#### **Modified Flask Routes** (Add to `src/web_interface.py`)
```python
# Add these imports at the top
from flask_socketio import SocketIO
from src.streaming.sse_handler import SSEHandler
from src.streaming.websocket_handler import WebSocketHandler
from src.streaming.video_streamer import VideoStreamer
import uuid

class WebInterface:
    def __init__(self, doorbell_system=None):
        # ... existing code ...
        
        # Add SocketIO support
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Initialize streaming components
        self.sse_handler = SSEHandler(doorbell_system.web_event_streamer)
        self.websocket_handler = WebSocketHandler(self.socketio, doorbell_system.pipeline_orchestrator)
        self.video_streamer = VideoStreamer(doorbell_system.camera_handler)
        
        # ... rest of existing code ...

    def _init_routes(self):
        # ... existing routes ...
        
        # SSE Endpoints
        @self.app.route('/stream/events')
        def stream_events():
            """Server-Sent Events stream for real-time events."""
            client_id = request.args.get('client_id', str(uuid.uuid4()))
            
            return Response(
                self.sse_handler.create_event_stream(client_id),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Access-Control-Allow-Origin': '*'
                }
            )
        
        @self.app.route('/stream/system-status')
        def stream_system_status():
            """SSE stream for system status updates."""
            client_id = request.args.get('client_id', str(uuid.uuid4()))
            
            return Response(
                self.sse_handler.create_system_status_stream(client_id),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive'
                }
            )
        
        # Video Streaming
        @self.app.route('/stream/video/<client_id>')
        def stream_video(client_id):
            """Live video stream endpoint."""
            quality = request.args.get('quality', 'medium')
            
            return Response(
                self.video_streamer.create_video_stream(client_id, quality),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
```

### **Frontend Integration** (Add to `templates/dashboard.html`)

#### **Enhanced JavaScript Clients**
```javascript
// Add to the <script> section of dashboard.html

class EventStreamManager {
    constructor() {
        this.eventSource = null;
        this.clientId = this.generateClientId();
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }
    
    generateClientId() {
        return 'client_' + Math.random().toString(36).substr(2, 9);
    }
    
    connect() {
        if (this.eventSource) {
            this.disconnect();
        }
        
        const url = `/stream/events?client_id=${this.clientId}`;
        this.eventSource = new EventSource(url);
        
        this.eventSource.onopen = () => {
            console.log('Event stream connected');
            this.reconnectAttempts = 0;
            this.updateConnectionStatus('connected');
        };
        
        this.eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleEvent(data);
        };
        
        this.eventSource.addEventListener('face_recognized', (event) => {
            const data = JSON.parse(event.data);
            this.displayFaceRecognitionEvent(data);
        });
        
        this.eventSource.addEventListener('system_status', (event) => {
            const data = JSON.parse(event.data);
            this.updateSystemStatus(data);
        });
        
        this.eventSource.onerror = () => {
            console.error('Event stream error');
            this.updateConnectionStatus('error');
            this.attemptReconnect();
        };
    }
    
    handleEvent(data) {
        console.log('Received event:', data);
        
        // Update events list in real-time
        this.addEventToList(data);
        
        // Show notification
        this.showNotification(data);
    }
    
    addEventToList(eventData) {
        const eventsList = document.getElementById('events-list');
        const eventElement = document.createElement('div');
        eventElement.className = 'event-item';
        
        const timestamp = new Date(eventData.timestamp * 1000).toLocaleString();
        eventElement.innerHTML = `
            <div class="event-time">${timestamp}</div>
            <div class="event-type">${eventData.event_type}</div>
            <div class="event-details">${JSON.stringify(eventData.data, null, 2)}</div>
        `;
        
        eventsList.insertBefore(eventElement, eventsList.firstChild);
        
        // Limit to last 20 events
        while (eventsList.children.length > 20) {
            eventsList.removeChild(eventsList.lastChild);
        }
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            setTimeout(() => {
                console.log(`Reconnecting... (attempt ${this.reconnectAttempts})`);
                this.connect();
            }, 2000 * this.reconnectAttempts);
        }
    }
    
    disconnect() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
    }
}

class WebSocketManager {
    constructor() {
        this.socket = null;
        this.connected = false;
    }
    
    connect() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('WebSocket connected');
            this.connected = true;
            
            // Subscribe to events
            this.socket.emit('subscribe', {
                event_types: ['face_recognized', 'doorbell_pressed', 'system_alert']
            });
        });
        
        this.socket.on('live_event', (data) => {
            console.log('Live event:', data);
            this.handleLiveEvent(data);
        });
        
        this.socket.on('command_result', (data) => {
            console.log('Command result:', data);
            this.showCommandResult(data);
        });
        
        this.socket.on('disconnect', () => {
            console.log('WebSocket disconnected');
            this.connected = false;
        });
    }
    
    sendCommand(command, params = {}) {
        if (this.connected) {
            this.socket.emit('system_command', {
                command: command,
                params: params
            });
        }
    }
    
    handleLiveEvent(data) {
        // Handle real-time events
        this.updateEventDisplay(data);
    }
}

// Initialize streaming managers
const eventStreamManager = new EventStreamManager();
const webSocketManager = new WebSocketManager();

// Start connections when page loads
document.addEventListener('DOMContentLoaded', () => {
    eventStreamManager.connect();
    webSocketManager.connect();
    
    // Add disconnect handlers for page unload
    window.addEventListener('beforeunload', () => {
        eventStreamManager.disconnect();
        webSocketManager.disconnect();
    });
});

// Enhanced doorbell trigger with real-time feedback
async function triggerDoorbell() {
    // Use WebSocket for immediate feedback
    webSocketManager.sendCommand('trigger_doorbell');
    
    // Also call the REST API as backup
    const result = await apiCall('trigger-doorbell', 'POST');
    console.log('Doorbell triggered:', result);
}
```

## 🧪 **Testing Requirements**

### **Unit Tests**
```python
class TestSSEHandler:
    """Test Server-Sent Events functionality."""
    
    def test_event_stream_creation(self):
        """Test SSE stream creation and formatting."""
        
    def test_client_connection_management(self):
        """Test client registration and cleanup."""
        
    def test_event_broadcasting(self):
        """Test event broadcasting to multiple clients."""

class TestWebSocketHandler:
    """Test WebSocket functionality."""
    
    def test_client_connection(self):
        """Test WebSocket client connection."""
        
    def test_bidirectional_communication(self):
        """Test sending and receiving messages."""
        
    def test_system_commands(self):
        """Test system control commands."""

class TestVideoStreamer:
    """Test live video streaming."""
    
    def test_video_stream_creation(self):
        """Test video stream setup."""
        
    def test_quality_adaptation(self):
        """Test different quality settings."""
```

### **Integration Tests**
```python
def test_end_to_end_streaming():
    """Test complete streaming pipeline from event to web display."""
    
def test_concurrent_clients():
    """Test multiple simultaneous streaming clients."""
    
def test_stream_performance():
    """Test streaming performance under load."""
```

## 📊 **Performance Targets**

### **Streaming Performance**
- **Event Latency**: <500ms from trigger to web display
- **Video Latency**: <1 second for live video
- **Concurrent Clients**: Support 50+ simultaneous connections
- **Memory Usage**: <100MB additional for streaming
- **CPU Usage**: <30% additional under full load

### **Reliability Targets**
- **Connection Stability**: >99.5% uptime for streaming connections
- **Event Delivery**: Zero event loss during normal operation
- **Recovery Time**: <5 seconds for automatic reconnection
- **Error Rate**: <0.1% for streaming operations

## 📁 **File Structure**
```
src/streaming/                          # Real-time streaming module
├── __init__.py
├── sse_handler.py                      # Server-Sent Events
├── websocket_handler.py                # WebSocket handler  
├── video_streamer.py                   # Live video streaming
└── stream_manager.py                   # Connection management

static/js/                              # Frontend clients
├── event-stream.js                     # SSE client
├── websocket-client.js                 # WebSocket client
└── video-stream.js                     # Video streaming

tests/streaming/                        # Streaming tests
├── test_sse_handler.py
├── test_websocket_handler.py
├── test_video_streamer.py
└── test_stream_integration.py
```

## ⚡ **Implementation Timeline**
- **Phase 1** (Days 1-2): SSE Handler Implementation
- **Phase 2** (Days 3-4): WebSocket Handler Implementation  
- **Phase 3** (Days 5-6): Live Video Streaming
- **Phase 4** (Days 7-8): Frontend Integration
- **Phase 5** (Days 9-10): Testing & Performance Optimization

## 🎯 **Definition of Done**
- [ ] SSE endpoints operational with real-time event streaming
- [ ] WebSocket bidirectional communication functional
- [ ] Live video streaming to web interface working
- [ ] Frontend clients integrated and responsive
- [ ] Performance targets achieved consistently
- [ ] All tests pass with >95% coverage
- [ ] Documentation updated with streaming examples
- [ ] Production deployment tested and verified

## 🔗 **Integration Points**
- **Connects**: Existing `WebEventStreamer` with web interface
- **Extends**: Current Flask web interface with real-time capabilities  
- **Completes**: Real-time monitoring and control system
- **Enables**: Production-ready live security monitoring

## 📚 **References**
- [Server-Sent Events Specification](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [WebSocket Protocol](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
- [Flask-SocketIO Documentation](https://flask-socketio.readthedocs.io/)
- [Real-time Web Applications Best Practices](docs/IMPLEMENTATION_GUIDE.md)

---

**This issue completes the real-time web streaming integration, connecting all existing components into a cohesive live monitoring system for the production-ready Frigate-inspired doorbell security architecture.**