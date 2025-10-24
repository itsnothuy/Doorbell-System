#!/usr/bin/env python3
"""
Frame Capture Worker Demo

Demonstrates the frame capture worker with ring buffer and event handling.
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


def create_mock_camera():
    """Create a simple mock camera for demonstration."""
    class MockCamera:
        def __init__(self):
            self.is_initialized = False
        
        def initialize(self):
            print("  📷 Initializing mock camera...")
            self.is_initialized = True
            return True
        
        def capture_frame(self):
            import numpy as np
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        def get_status(self):
            return 'mock_active'
        
        def cleanup(self):
            print("  📷 Cleaning up camera...")
            self.is_initialized = False
    
    return MockCamera()


def demo_continuous_capture():
    """Demonstrate continuous frame capture."""
    print("\n" + "="*60)
    print("Demo: Continuous Frame Capture with Ring Buffer")
    print("="*60)
    
    # Setup
    message_bus = MessageBus()
    message_bus.start()
    
    camera = create_mock_camera()
    camera.initialize()
    
    config = {
        'buffer_size': 10,
        'capture_fps': 15,
        'burst_count': 3,
        'burst_interval': 0.2
    }
    
    # Create worker
    worker = FrameCaptureWorker(camera, message_bus, config)
    
    # Start worker
    print("\n▶️  Starting frame capture worker...")
    worker_thread = threading.Thread(target=worker.start, daemon=True)
    worker_thread.start()
    
    # Let it capture for a few seconds
    print("  Capturing frames...")
    for i in range(5):
        time.sleep(1)
        metrics = worker.get_metrics()
        print(f"  [{i+1}s] Captured: {metrics['frames_captured']} frames, "
              f"Buffer: {metrics['buffer_size']}/{metrics['buffer_capacity']}")
    
    # Show final metrics
    metrics = worker.get_metrics()
    print(f"\n📊 Final Stats:")
    print(f"  Total frames captured: {metrics['frames_captured']}")
    print(f"  Actual FPS: {metrics['frames_captured'] / metrics['uptime_seconds']:.2f}")
    print(f"  Error count: {metrics['capture_errors']}")
    
    # Cleanup
    print("\n⏸️  Stopping worker...")
    worker.stop()
    worker_thread.join(timeout=2.0)
    message_bus.stop()
    
    print("✅ Demo complete!\n")


def main():
    """Run demonstration."""
    print("\n" + "🎬 "*20)
    print("Frame Capture Worker Demonstration")
    print("🎬 "*20)
    
    try:
        demo_continuous_capture()
        
        print("\n" + "="*60)
        print("✅ Demonstration completed successfully!")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
