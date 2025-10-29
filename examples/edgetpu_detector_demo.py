#!/usr/bin/env python3
"""
EdgeTPU Detector Demo

Demonstrates usage of the EdgeTPU face detector with model management,
inference, and performance monitoring.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.detectors.edgetpu_detector import (
    EdgeTPUDetector,
    EdgeTPUModelManager,
    detect_edgetpu_devices,
    is_edgetpu_available
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_hardware_detection():
    """Demonstrate EdgeTPU hardware detection."""
    logger.info("=" * 60)
    logger.info("EdgeTPU Hardware Detection Demo")
    logger.info("=" * 60)
    
    # Check if EdgeTPU is available
    available = is_edgetpu_available()
    logger.info(f"EdgeTPU Available: {available}")
    
    if available:
        # List EdgeTPU devices
        devices = detect_edgetpu_devices()
        logger.info(f"Found {len(devices)} EdgeTPU device(s):")
        for device in devices:
            logger.info(f"  - Type: {device['type']}, Path: {device['path']}")
    else:
        logger.warning("No EdgeTPU devices found")
    
    logger.info("")


def demo_model_management():
    """Demonstrate EdgeTPU model management."""
    logger.info("=" * 60)
    logger.info("EdgeTPU Model Management Demo")
    logger.info("=" * 60)
    
    # Create model manager
    model_manager = EdgeTPUModelManager()
    
    # List available models
    models = model_manager.list_available_models()
    logger.info(f"Available models: {len(models)}")
    for model_name in models:
        model_info = model_manager.get_model_info(model_name)
        logger.info(f"  - {model_info.name}")
        logger.info(f"    File: {model_info.file_path}")
        logger.info(f"    Input size: {model_info.input_size}")
        logger.info(f"    URL: {model_info.url or 'N/A'}")
    
    logger.info("")


def demo_detector_creation():
    """Demonstrate EdgeTPU detector creation."""
    logger.info("=" * 60)
    logger.info("EdgeTPU Detector Creation Demo")
    logger.info("=" * 60)
    
    # Check availability
    if not EdgeTPUDetector.is_available():
        logger.warning("EdgeTPU detector is not available on this system")
        logger.info("This demo requires:")
        logger.info("  1. Google Coral EdgeTPU device (USB Accelerator or Dev Board)")
        logger.info("  2. EdgeTPU runtime installed")
        logger.info("  3. pycoral and tflite-runtime libraries")
        return
    
    # Create detector with configuration
    config = {
        'model': 'mobilenet_face',
        'confidence_threshold': 0.6,
        'min_face_size': (30, 30),
        'max_face_size': (1000, 1000),
        'auto_download': True,  # Enable model auto-download
        'enable_monitoring': True,
        'temperature_limit': 85.0
    }
    
    try:
        logger.info("Creating EdgeTPU detector...")
        detector = EdgeTPUDetector(config)
        logger.info(f"Detector created successfully: {detector.detector_type}")
        
        # Get model info
        model_info = detector.get_model_info()
        logger.info(f"Model: {model_info.get('name', 'N/A')}")
        logger.info(f"Input size: {model_info.get('input_size', 'N/A')}")
        
        # Cleanup
        detector.cleanup()
        logger.info("Detector cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Failed to create detector: {e}")
    
    logger.info("")


def demo_inference():
    """Demonstrate face detection inference."""
    logger.info("=" * 60)
    logger.info("EdgeTPU Face Detection Inference Demo")
    logger.info("=" * 60)
    
    # Check availability
    if not EdgeTPUDetector.is_available():
        logger.warning("EdgeTPU detector is not available on this system")
        return
    
    # Create detector
    config = {
        'model': 'mobilenet_face',
        'confidence_threshold': 0.5,
        'auto_download': True
    }
    
    try:
        logger.info("Creating detector...")
        detector = EdgeTPUDetector(config)
        
        # Create test image (random noise for demo)
        logger.info("Creating test image (320x240 RGB)...")
        test_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        
        # Run detection
        logger.info("Running face detection...")
        detections, metrics = detector.detect_faces(test_image)
        
        # Display results
        logger.info(f"Detection complete:")
        logger.info(f"  - Faces detected: {len(detections)}")
        logger.info(f"  - Inference time: {metrics.inference_time*1000:.2f}ms")
        logger.info(f"  - Total time: {metrics.total_time*1000:.2f}ms")
        logger.info(f"  - FPS: {1.0/metrics.total_time:.1f}")
        
        # Display detection details
        for i, detection in enumerate(detections):
            logger.info(f"  Face {i+1}:")
            logger.info(f"    - Bounding box: {detection.bounding_box}")
            logger.info(f"    - Confidence: {detection.confidence:.3f}")
            logger.info(f"    - Quality score: {detection.quality_score:.3f}")
        
        # Get performance metrics
        perf_metrics = detector.get_performance_metrics()
        logger.info(f"Performance metrics:")
        logger.info(f"  - Backend: {perf_metrics.get('backend', 'N/A')}")
        logger.info(f"  - Average inference time: {perf_metrics.get('average_inference_time_ms', 0):.2f}ms")
        logger.info(f"  - Theoretical FPS: {perf_metrics.get('fps', 0):.1f}")
        
        # Cleanup
        detector.cleanup()
        
    except Exception as e:
        logger.error(f"Inference demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("")


def demo_benchmarking():
    """Demonstrate detector benchmarking."""
    logger.info("=" * 60)
    logger.info("EdgeTPU Detector Benchmarking Demo")
    logger.info("=" * 60)
    
    # Check availability
    if not EdgeTPUDetector.is_available():
        logger.warning("EdgeTPU detector is not available on this system")
        return
    
    # Create detector
    config = {
        'model': 'mobilenet_face',
        'confidence_threshold': 0.5,
        'auto_download': True
    }
    
    try:
        logger.info("Creating detector...")
        detector = EdgeTPUDetector(config)
        
        # Create test images
        logger.info("Creating test images...")
        test_images = [
            np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        
        # Run benchmark
        logger.info("Running benchmark (3 images x 5 iterations)...")
        results = detector.benchmark(test_images, iterations=5)
        
        # Display results
        stats = results.get('statistics', {})
        logger.info(f"Benchmark results:")
        logger.info(f"  - Total iterations: {results.get('total_iterations', 0)}")
        logger.info(f"  - Mean inference time: {stats.get('mean_inference_time', 0)*1000:.2f}ms")
        logger.info(f"  - Min inference time: {stats.get('min_inference_time', 0)*1000:.2f}ms")
        logger.info(f"  - Max inference time: {stats.get('max_inference_time', 0)*1000:.2f}ms")
        logger.info(f"  - Std deviation: {stats.get('std_inference_time', 0)*1000:.2f}ms")
        logger.info(f"  - Mean FPS: {stats.get('mean_fps', 0):.1f}")
        logger.info(f"  - Mean detections: {stats.get('mean_detections', 0):.1f}")
        logger.info(f"  - Error rate: {stats.get('error_rate', 0)*100:.1f}%")
        
        # Cleanup
        detector.cleanup()
        
    except Exception as e:
        logger.error(f"Benchmarking demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("")


def main():
    """Run all demos."""
    logger.info("EdgeTPU Detector Demonstration")
    logger.info("=" * 60)
    logger.info("")
    
    # Run demos
    demo_hardware_detection()
    demo_model_management()
    demo_detector_creation()
    demo_inference()
    demo_benchmarking()
    
    logger.info("=" * 60)
    logger.info("Demo complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
