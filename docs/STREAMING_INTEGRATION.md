# Real-Time Web Streaming Integration

## Overview

This document describes the real-time streaming architecture for the Doorbell Security System, including Server-Sent Events (SSE), WebSocket, and live video streaming capabilities.

## Architecture

### Components

1. **SSE Handler** (`src/streaming/sse_handler.py`)
   - Manages Server-Sent Events for one-way real-time updates
   - Handles client connections and heartbeats
   - Integrates with WebEventStreamer for event distribution

2. **WebSocket Handler** (`src/streaming/websocket_handler.py`)
   - Provides bidirectional communication
   - Handles system commands and control
   - Broadcasts events to subscribed clients

3. **Video Streamer** (`src/streaming/video_streamer.py`)
   - Streams live camera feed to web clients
   - Supports multiple quality levels (low/medium/high)
   - Handles multiple concurrent streams

4. **Web Event Streamer** (`src/enrichment/web_events.py`)
   - Core event distribution system (already implemented)
   - Manages client queues and event buffering
   - Tracks streaming statistics

### Flow Diagram

```
┌─────────────────┐
│  Pipeline Event │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ WebEventsEnrichment │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐       ┌──────────────┐
│ WebEventStreamer    │◄──────┤  SSE Handler │
│                     │       └──────┬───────┘
│ - Client Queues     │              │
│ - Event Buffer      │              ▼
│ - Connection Mgmt   │       ┌──────────────┐
└────────┬────────────┘       │  Web Client  │
         │                     │  (Browser)   │
         ▼                     └──────────────┘
┌─────────────────────┐
│ WebSocket Handler   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Video Streamer    │
└─────────────────────┘
```

## API Endpoints

### SSE Endpoints

#### `/stream/events`
Real-time event stream for security events.

**Query Parameters:**
- `client_id` (optional): Unique client identifier

**Response Format:**
```
event: connected
data: {"client_id": "client_123", "server_time": 1234567890}

event: event
data: {"event_id": "evt_1", "event_type": "DOORBELL_PRESSED", "timestamp": 1234567890}

event: heartbeat
data: {"timestamp": 1234567890}
```

#### `/stream/system-status`
System status updates stream.

**Query Parameters:**
- `client_id` (optional): Unique client identifier

**Response Format:**
```
event: system_status
data: {"timestamp": 1234567890, "streaming_status": "running", "active_connections": 5}
```

### Video Streaming

#### `/stream/video/<client_id>`
Live MJPEG video stream.

**Query Parameters:**
- `quality`: Video quality (low/medium/high)

**Quality Settings:**
- **Low**: 320x240 @ 10fps, JPEG quality 70
- **Medium**: 640x480 @ 15fps, JPEG quality 80
- **High**: 1280x720 @ 20fps, JPEG quality 90

### WebSocket Events

#### Client → Server

**`subscribe`**
Subscribe to event types.
```json
{
  "event_types": ["doorbell_pressed", "face_detected"]
}
```

**`system_command`**
Execute system command.
```json
{
  "command": "trigger_doorbell",
  "params": {}
}
```

**`request_video_stream`**
Request video stream URL.
```json
{
  "quality": "medium"
}
```

#### Server → Client

**`connected`**
Connection acknowledgment.
```json
{
  "client_id": "client_123",
  "server_capabilities": ["system_control", "live_events", "video_stream"]
}
```

**`live_event`**
Real-time event broadcast.
```json
{
  "event_type": "doorbell_pressed",
  "data": {"timestamp": 1234567890},
  "timestamp": 1234567890
}
```

**`command_result`**
Command execution result.
```json
{
  "command": "trigger_doorbell",
  "success": true,
  "result": {"triggered": true}
}
```

## Frontend Integration

### Including Scripts

Add to your HTML `<head>` or before closing `</body>`:

```html
<!-- Socket.IO Client (for WebSocket) -->
<script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>

<!-- Streaming Clients -->
<script src="/static/js/event-stream.js"></script>
<script src="/static/js/websocket-client.js"></script>
<script src="/static/js/video-stream.js"></script>
```

### JavaScript Usage

#### SSE Event Stream

```javascript
// Initialize event stream client
const eventClient = new EventStreamClient();

// Handle events
eventClient.on('connected', (data) => {
    console.log('Connected:', data);
});

eventClient.on('event', (data) => {
    console.log('Event received:', data);
    updateUI(data);
});

eventClient.on('status', (data) => {
    console.log('System status:', data);
    updateStatusBar(data);
});

// Connect
eventClient.connect();

// Later: disconnect
eventClient.disconnect();
```

#### WebSocket Client

```javascript
// Initialize WebSocket client
const wsClient = new WebSocketClient();

// Handle events
wsClient.on('connect', () => {
    console.log('WebSocket connected');
});

wsClient.on('live_event', (data) => {
    console.log('Live event:', data);
});

// Connect
wsClient.connect();

// Subscribe to events
wsClient.subscribe(['doorbell_pressed', 'face_detected']);

// Send command
wsClient.sendCommand('trigger_doorbell', {})
    .then(result => console.log('Result:', result))
    .catch(error => console.error('Error:', error));

// Request video stream
wsClient.requestVideoStream('medium')
    .then(data => {
        console.log('Stream URL:', data.stream_url);
        startVideo(data.stream_url);
    });
```

#### Video Stream

```javascript
// Get video element
const videoElement = document.getElementById('live-video');

// Initialize video stream client
const videoClient = new VideoStreamClient(videoElement);

// Start stream
videoClient.startStream('medium')
    .then(() => console.log('Video streaming started'))
    .catch(error => console.error('Failed to start video:', error));

// Change quality
videoClient.changeQuality('high');

// Stop stream
videoClient.stopStream();
```

## Configuration

### Backend Configuration

In your pipeline configuration:

```python
streaming_config = {
    'enabled': True,
    'max_connections': 50,
    'buffer_size': 100,
    'heartbeat_interval': 30.0
}
```

### WebInterface Initialization

```python
from src.web_interface import WebInterface

# Initialize with doorbell system
web_interface = WebInterface(doorbell_system)

# Run with SocketIO support
web_interface.run(host='0.0.0.0', port=5000, debug=False)
```

## Performance Considerations

### Connection Limits

- **SSE**: Up to 50 concurrent connections (configurable)
- **WebSocket**: Up to 50 concurrent connections (configurable)
- **Video**: Recommended max 10 concurrent streams

### Resource Usage

**Per SSE Connection:**
- Memory: ~1-2 MB per connection
- CPU: <1% per connection

**Per Video Stream (Medium Quality):**
- Bandwidth: ~300-500 KB/s per stream
- CPU: 5-10% per stream (varies by quality)

### Optimization Tips

1. **Use SSE for one-way updates** - Lower overhead than WebSocket
2. **Limit video quality** - Use 'low' or 'medium' for battery devices
3. **Implement backpressure** - Queue size limits prevent memory issues
4. **Monitor connections** - Track active streams and cleanup stale ones
5. **Use CDN for static assets** - Reduce load on main server

## Error Handling

### Connection Failures

All clients implement automatic reconnection with exponential backoff:
- Initial delay: 3 seconds
- Max delay: 30 seconds
- Automatic retry on disconnect

### Event Loss Prevention

- **Event buffering**: Last 100 events kept in memory
- **Queue overflow handling**: Events dropped with warning log
- **Heartbeat mechanism**: Detects stale connections

### Video Stream Errors

- Automatic frame skip on encoding errors
- Graceful degradation on camera failures
- Client timeout after 30 seconds of inactivity

## Security Considerations

### Authentication

Current implementation provides basic functionality. For production:

1. Implement token-based authentication
2. Add CORS restrictions
3. Use HTTPS/WSS for secure connections
4. Implement rate limiting

### Example Authentication

```python
from functools import wraps
from flask import request, jsonify

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.args.get('token')
        if not verify_token(token):
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/stream/events')
@require_auth
def stream_events():
    # ... existing code
```

## Testing

### Running Tests

```bash
# Run all streaming tests
pytest tests/streaming/ -v

# Run specific test
pytest tests/streaming/test_sse_handler.py -v

# Run integration tests
pytest tests/streaming/test_streaming_integration.py -v
```

### Manual Testing

1. **Start the web interface:**
   ```bash
   python app.py
   ```

2. **Open browser console** at http://localhost:5000

3. **Check streaming initialization:**
   ```javascript
   // Should see console messages:
   // "Initializing real-time streaming..."
   // "Event stream connected: {...}"
   ```

4. **Trigger an event:**
   ```bash
   curl -X POST http://localhost:5000/api/trigger-doorbell
   ```

5. **Verify event received** in browser console

## Troubleshooting

### SSE Connection Issues

**Problem:** Events not received
- **Check:** Browser console for connection errors
- **Verify:** `/stream/events` endpoint is accessible
- **Solution:** Check firewall and CORS settings

**Problem:** Connection drops frequently
- **Check:** Server logs for errors
- **Verify:** Heartbeat interval is appropriate
- **Solution:** Increase heartbeat interval or check network stability

### WebSocket Issues

**Problem:** Socket.IO not connecting
- **Check:** Socket.IO client is loaded
- **Verify:** WebSocket transport is enabled
- **Solution:** Fall back to polling if WebSocket is blocked

### Video Stream Issues

**Problem:** No video displayed
- **Check:** Camera handler is initialized
- **Verify:** Video element has correct src
- **Solution:** Check camera permissions and availability

**Problem:** Low frame rate
- **Check:** CPU usage on server
- **Verify:** Network bandwidth
- **Solution:** Reduce quality setting or limit concurrent streams

## Examples

See `examples/streaming_demo.py` for a complete demonstration of:
- SSE streaming setup
- WebSocket handler usage
- Video streaming configuration
- Complete integration flow

Run the demo:
```bash
python examples/streaming_demo.py
```

## Future Enhancements

- [ ] Add authentication and authorization
- [ ] Implement rate limiting per client
- [ ] Add metrics and monitoring dashboard
- [ ] Support WebRTC for lower latency video
- [ ] Add event filtering and subscriptions
- [ ] Implement client preferences storage
- [ ] Add compression for event data
- [ ] Support multiple camera streams

## References

- [Server-Sent Events Specification](https://html.spec.whatwg.org/multipage/server-sent-events.html)
- [Socket.IO Documentation](https://socket.io/docs/v4/)
- [Flask-SocketIO Documentation](https://flask-socketio.readthedocs.io/)
- [MJPEG Streaming](https://en.wikipedia.org/wiki/Motion_JPEG)
