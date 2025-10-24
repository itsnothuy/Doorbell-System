#!/usr/bin/env python3
"""
Integration test for Frame Capture Worker

Tests the complete frame capture workflow with message bus.
"""

import sys
import time
import threading
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pipeline.frame_capture import FrameCaptureWorker
from src.communication.message_bus import MessageBus
from src.communication.events import DoorbellEvent, EventType

# Mock camera handler for tests
class SimpleMockCamera:
    def __init__(self):
        self.is_initialized = False
        self.capture_count = 0
    
    def initialize(self):
        self.is_initialized = True
        return True
    
    def capture_frame(self):
        self.capture_count += 1
        # Return a numpy array or compatible mock
        import numpy as np
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def get_status(self):
        return 'mock_active'
    
    def cleanup(self):
        self.is_initialized = False


def test_end_to_end_workflow():
    """Test complete frame capture workflow with doorbell event."""
    print("Starting end-to-end frame capture test...")
    
    # Setup
    message_bus = MessageBus()
    message_bus.start()
    
    camera = SimpleMockCamera()
    camera.initialize()
    
    config = {
        'buffer_size': 10,
        'capture_fps': 10,
        'burst_count': 3,
        'burst_interval': 0.1
    }
    
    # Track published frames
    frames_received = []
    
    def frame_handler(message):
        frames_received.append(message.data)
        print(f"  Received frame event: {message.data.event_type}")
    
    message_bus.subscribe('frame_captured', frame_handler, 'test_subscriber')
    
    # Create and start worker
    worker = FrameCaptureWorker(camera, message_bus, config)
    
    worker_thread = threading.Thread(target=worker.start)
    worker_thread.start()
    
    # Wait for initialization
    time.sleep(0.5)
    
    print("Worker started, sending doorbell event...")
    
    # Simulate doorbell press
    doorbell_event = DoorbellEvent(
        event_type=EventType.DOORBELL_PRESSED,
        channel=18
    )
    
    message_bus.publish('doorbell_pressed', doorbell_event)
    
    # Wait for processing
    time.sleep(1.0)
    
    # Cleanup
    worker.stop()
    worker_thread.join(timeout=2.0)
    message_bus.stop()
    
    # Verify
    print(f"\nResults:")
    print(f"  Frames captured total: {worker.frames_captured}")
    print(f"  Ring buffer size: {len(worker.ring_buffer)}")
    print(f"  Frame events published: {len(frames_received)}")
    print(f"  Camera capture count: {camera.capture_count}")
    
    # Assertions
    assert worker.frames_captured > 0, "Should have captured some frames"
    assert len(frames_received) == config['burst_count'], f"Should have received {config['burst_count']} frame events"
    
    print("\n✅ End-to-end test passed!")
    return True


if __name__ == '__main__':
    try:
        success = test_end_to_end_workflow()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
