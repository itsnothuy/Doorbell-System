#!/usr/bin/env python3
"""
Face Detection Worker Pool - Integration Example

This example demonstrates the face detection worker pool in action with
the complete pipeline architecture.
"""

import sys
import time
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_face_detection_example():
    """Run face detection worker pool example."""
    
    try:
        import numpy as np
    except ImportError:
        logger.error("numpy is required for this example. Install with: pip install numpy")
        return
    
    # Import required modules
    from src.communication.message_bus import MessageBus
    from src.communication.events import FrameEvent, EventType
    from src.pipeline.face_detector import FaceDetectionWorker
    from src.detectors.detector_factory import DetectorFactory
    
    logger.info("=== Face Detection Worker Pool Example ===")
    
    # Check available detectors
    logger.info("\n1. Checking available detectors...")
    available_detectors = DetectorFactory.list_detectors()
    for detector_type, is_available in available_detectors.items():
        status = "✓ Available" if is_available else "✗ Not available"
        logger.info(f"   {detector_type}: {status}")
    
    # Auto-detect best detector
    best_detector = DetectorFactory.auto_detect_best_detector()
    logger.info(f"\n   Best detector: {best_detector}")
    
    # Initialize message bus
    logger.info("\n2. Initializing message bus...")
    message_bus = MessageBus()
    message_bus.start()
    
    # Configure face detection worker
    config = {
        'worker_count': 2,
        'detector_type': best_detector,
        'max_queue_size': 10,
        'job_timeout': 30.0,
        'model': 'hog',
        'number_of_times_to_upsample': 1,
        'confidence_threshold': 0.5,
        'min_face_size': (30, 30)
    }
    
    # Create face detection worker
    logger.info("\n3. Creating face detection worker pool...")
    face_detector = FaceDetectionWorker(message_bus, config)
    
    # Subscribe to detection results
    detection_results = []
    
    def handle_detection_result(message):
        """Handle face detection results."""
        event = message.data
        face_count = event.data.get('face_count', 0)
        processing_time = event.data.get('processing_time_ms', 0)
        
        logger.info(
            f"   Detection result: {face_count} faces found "
            f"(processing time: {processing_time:.2f}ms)"
        )
        detection_results.append(event)
    
    message_bus.subscribe('faces_detected', handle_detection_result, 'example_listener')
    
    # Initialize worker
    logger.info("\n4. Initializing worker pool...")
    try:
        face_detector._initialize_worker()
        logger.info(f"   Worker pool initialized with {config['worker_count']} workers")
    except Exception as e:
        logger.error(f"   Failed to initialize worker pool: {e}")
        message_bus.stop()
        return
    
    # Simulate frame capture events
    logger.info("\n5. Simulating frame capture events...")
    
    for i in range(3):
        # Create test frame (blank image)
        frame_data = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create frame event
        frame_event = FrameEvent(
            event_type=EventType.FRAME_CAPTURED,
            frame_data=frame_data,
            event_id=f'test_frame_{i}'
        )
        
        # Mark first frame as doorbell event (higher priority)
        if i == 0:
            frame_event.data['source'] = 'doorbell'
            logger.info(f"   Sending frame {i} (doorbell event - high priority)")
        else:
            logger.info(f"   Sending frame {i} (normal priority)")
        
        # Publish frame event
        message_bus.publish('frame_captured', frame_event)
        
        # Small delay between frames
        time.sleep(0.5)
    
    # Wait for processing
    logger.info("\n6. Waiting for detection results...")
    time.sleep(3)
    
    # Get worker metrics
    logger.info("\n7. Worker metrics:")
    metrics = face_detector.get_metrics()
    logger.info(f"   Detection count: {metrics['detection_count']}")
    logger.info(f"   Detection errors: {metrics['detection_errors']}")
    logger.info(f"   Queue size: {metrics['queue_size']}")
    logger.info(f"   Pending jobs: {metrics['pending_jobs']}")
    logger.info(f"   Worker count: {metrics['worker_count']}")
    logger.info(f"   Detector type: {metrics['detector_type']}")
    
    if metrics['detection_count'] > 0:
        logger.info(f"   Avg detection time: {metrics['avg_detection_time']:.3f}s")
        logger.info(f"   Detection rate: {metrics['detection_rate']:.2f} detections/sec")
    
    # Cleanup
    logger.info("\n8. Cleaning up...")
    face_detector._cleanup_worker()
    message_bus.stop()
    
    logger.info("\n=== Example completed successfully! ===")
    logger.info(f"Processed {len(detection_results)} detection results")


if __name__ == '__main__':
    try:
        run_face_detection_example()
    except KeyboardInterrupt:
        logger.info("\nExample interrupted by user")
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
