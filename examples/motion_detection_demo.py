#!/usr/bin/env python3
"""
Motion Detection Worker Demo

This example demonstrates how to use the motion detection worker
with the message bus for pipeline integration.
"""

import sys
import time
import threading
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import numpy as np
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("WARNING: OpenCV not available. Using mock data.")

from src.communication.message_bus import MessageBus, Message
from src.communication.events import FrameEvent, EventType
from config.motion_config import MotionConfig, create_default_config

if OPENCV_AVAILABLE:
    from src.pipeline.motion_detector import MotionDetector


def create_test_frame(width=640, height=480, add_motion=False):
    """Create a test frame with optional motion."""
    if not OPENCV_AVAILABLE:
        return None
    
    # Create base frame (static)
    frame = np.ones((height, width, 3), dtype=np.uint8) * 128
    
    if add_motion:
        # Add a moving rectangle to simulate motion
        x = int(time.time() * 50) % width
        cv2.rectangle(frame, (x, 100), (x+50, 200), (255, 255, 255), -1)
    
    return frame


def motion_event_handler(message: Message):
    """Handle motion analyzed events."""
    frame_event = message.data
    
    if hasattr(frame_event, 'data') and isinstance(frame_event.data, dict):
        motion_detected = frame_event.data.get('motion_detected', False)
        motion_score = frame_event.data.get('motion_score', 0.0)
        
        status = "MOTION" if motion_detected else "STATIC"
        print(f"[{status}] Frame analyzed - Motion score: {motion_score:.2f}%")
        
        if motion_detected:
            motion_data = frame_event.data.get('motion_data', {})
            contour_count = motion_data.get('contour_count', 0)
            print(f"  → {contour_count} motion regions detected")


def run_basic_demo():
    """Run basic motion detection demo."""
    print("=" * 60)
    print("Motion Detection Worker - Basic Demo")
    print("=" * 60)
    
    if not OPENCV_AVAILABLE:
        print("\nERROR: OpenCV is required for this demo.")
        print("Install with: pip install opencv-python numpy")
        return
    
    # Create message bus
    print("\n[1] Creating message bus...")
    message_bus = MessageBus()
    message_bus.start()
    
    # Subscribe to motion analyzed events
    print("[2] Subscribing to motion events...")
    subscriber_id = f'demo_subscriber_{int(time.time() * 1000)}'
    message_bus.subscribe('motion_analyzed', motion_event_handler, subscriber_id)
    
    # Create motion detector with default config
    print("[3] Creating motion detector...")
    config = create_default_config()
    print(f"    - Motion threshold: {config.motion_threshold}")
    print(f"    - Min contour area: {config.min_contour_area}")
    print(f"    - Background subtractor: {config.bg_subtractor_type}")
    
    detector = MotionDetector(message_bus, config)
    
    # Start detector in background thread
    print("[4] Starting motion detector...")
    detector_thread = threading.Thread(target=detector.start, daemon=True)
    detector_thread.start()
    
    # Wait for initialization
    time.sleep(0.5)
    
    # Publish test frames
    print("\n[5] Publishing test frames...")
    print("-" * 60)
    
    for i in range(10):
        # Alternate between static and motion frames (motion every 3rd frame)
        has_motion = i % 3 == 0
        frame = create_test_frame(add_motion=has_motion)
        
        # Create frame event
        # Note: resolution is (width, height), frame.shape is (height, width, channels)
        frame_event = FrameEvent(
            event_type=EventType.FRAME_CAPTURED,
            frame_data=frame,
            resolution=(frame.shape[1], frame.shape[0])  # (width, height)
        )
        
        # Publish frame
        message_bus.publish('frame_captured', frame_event)
        
        # Wait a bit
        time.sleep(0.3)
    
    # Wait for processing to complete
    print("\n[6] Waiting for processing to complete...")
    time.sleep(1.0)
    
    # Get and display metrics
    print("\n[7] Motion detector metrics:")
    print("-" * 60)
    metrics = detector.get_metrics()
    print(f"Frames processed: {metrics['frames_processed']}")
    print(f"Frames forwarded: {metrics['frames_forwarded']}")
    print(f"Motion events: {metrics['motion_events']}")
    print(f"Forward ratio: {metrics['forward_ratio']:.2%}")
    print(f"Motion event ratio: {metrics['motion_event_ratio']:.2%}")
    print(f"Motion trend: {metrics['motion_trend']}")
    
    # Cleanup
    print("\n[8] Cleaning up...")
    detector.stop()
    message_bus.stop()
    
    print("\n✓ Demo completed successfully!")
    print("=" * 60)


def run_custom_config_demo():
    """Run demo with custom configuration."""
    print("\n" + "=" * 60)
    print("Motion Detection Worker - Custom Configuration Demo")
    print("=" * 60)
    
    if not OPENCV_AVAILABLE:
        print("\nERROR: OpenCV is required for this demo.")
        return
    
    # Create custom configuration
    print("\n[1] Creating custom configuration...")
    config = MotionConfig.from_dict({
        'enabled': True,
        'motion_threshold': 30.0,  # Higher threshold
        'min_contour_area': 800,   # Larger contours only
        'bg_subtractor_type': 'MOG2',
        'frame_resize_factor': 0.6,
        'max_static_duration': 20.0
    })
    
    print(f"    - Motion threshold: {config.motion_threshold}")
    print(f"    - Min contour area: {config.min_contour_area}")
    print(f"    - Frame resize factor: {config.frame_resize_factor}")
    
    # Create message bus and detector
    message_bus = MessageBus()
    message_bus.start()
    
    detector = MotionDetector(message_bus, config)
    detector_thread = threading.Thread(target=detector.start, daemon=True)
    detector_thread.start()
    
    time.sleep(0.5)
    
    # Test with different motion levels
    print("\n[2] Testing with different motion levels...")
    
    for level in ['low', 'medium', 'high']:
        print(f"\n  Testing {level} motion:")
        
        for i in range(3):
            if level == 'low':
                frame = create_test_frame(add_motion=i == 0)
            elif level == 'medium':
                frame = create_test_frame(add_motion=i < 2)
            else:  # high
                frame = create_test_frame(add_motion=True)
            
            frame_event = FrameEvent(
                event_type=EventType.FRAME_CAPTURED,
                frame_data=frame,
                resolution=(frame.shape[1], frame.shape[0])  # (width, height)
            )
            
            message_bus.publish('frame_captured', frame_event)
            time.sleep(0.2)
    
    time.sleep(1.0)
    
    # Show results
    print("\n[3] Results:")
    print("-" * 60)
    metrics = detector.get_metrics()
    print(f"Frames processed: {metrics['frames_processed']}")
    print(f"Frames forwarded: {metrics['frames_forwarded']}")
    print(f"Motion events: {metrics['motion_events']}")
    
    # Cleanup
    detector.stop()
    message_bus.stop()
    
    print("\n✓ Custom config demo completed!")


def main():
    """Main demo entry point."""
    print("\nMotion Detection Worker Demo")
    print("Choose a demo to run:")
    print("  1. Basic demo")
    print("  2. Custom configuration demo")
    print("  3. Run both")
    
    choice = input("\nEnter choice (1-3) or press Enter for basic demo: ").strip()
    
    if not choice:
        choice = '1'
    
    try:
        if choice == '1':
            run_basic_demo()
        elif choice == '2':
            run_custom_config_demo()
        elif choice == '3':
            run_basic_demo()
            time.sleep(1)
            run_custom_config_demo()
        else:
            print("Invalid choice. Running basic demo...")
            run_basic_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError running demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
