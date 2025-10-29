#!/usr/bin/env python3
"""
Production Google Coral EdgeTPU Face Detection Implementation

Provides ultra-fast face detection using Google Coral EdgeTPU accelerator,
optimized for real-time doorbell security applications.
"""

import os
import cv2
import numpy as np
import logging
import time
import threading
import hashlib
import requests
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

# EdgeTPU specific imports
try:
    import tflite_runtime.interpreter as tflite
    from pycoral.utils import edgetpu
    from pycoral.utils import dataset
    from pycoral.adapters import common
    from pycoral.adapters import detect
    EDGETPU_AVAILABLE = True
except ImportError:
    EDGETPU_AVAILABLE = False
    logging.warning("EdgeTPU libraries not available - falling back to CPU")

from src.detectors.base_detector import (
    BaseDetector, 
    DetectorType, 
    ModelType, 
    FaceDetectionResult
)

logger = logging.getLogger(__name__)


@dataclass
class EdgeTPUModelInfo:
    """EdgeTPU model configuration and metadata."""
    name: str
    file_path: str
    url: Optional[str] = None
    checksum: Optional[str] = None
    input_size: Tuple[int, int] = (320, 320)
    score_threshold: float = 0.5
    iou_threshold: float = 0.4
    max_detections: int = 10
    preprocessing_mean: Tuple[float, float, float] = (127.5, 127.5, 127.5)
    preprocessing_std: Tuple[float, float, float] = (127.5, 127.5, 127.5)


@dataclass
class DetectionResult:
    """Face detection result with confidence and coordinates."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    landmarks: Optional[List[Tuple[int, int]]] = None
    inference_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'landmarks': self.landmarks,
            'inference_time': self.inference_time
        }


class EdgeTPUModelManager:
    """Manages EdgeTPU model downloading, validation, and caching."""
    
    # Pre-trained face detection models optimized for EdgeTPU
    AVAILABLE_MODELS = {
        'mobilenet_ssd_v2_face': EdgeTPUModelInfo(
            name='MobileNet SSD v2 Face',
            file_path='models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite',
            url='https://github.com/google-coral/test_data/raw/master/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite',
            checksum='7448076066ac85d0beb3100ae3e6d7bb7ad3c000da0e3d4b90b993a1da4cbe93',
            input_size=(320, 320),
            score_threshold=0.5,
            max_detections=10
        ),
        'ssd_mobilenet_v1_face': EdgeTPUModelInfo(
            name='SSD MobileNet v1 Face',
            file_path='models/ssd_mobilenet_v1_face_quant_postprocess_edgetpu.tflite',
            url='https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v1_face_quant_postprocess_edgetpu.tflite',
            checksum='171d8b6115e81421bdbc8175dd09de6cf8b9edc14d09a91d1d5b98c7c8084f6b',
            input_size=(300, 300),
            score_threshold=0.6,
            max_detections=10
        ),
        'mobilenet_face': EdgeTPUModelInfo(
            name='MobileNet Face (Legacy)',
            file_path='models/mobilenet_face_edgetpu.tflite',
            url=None,  # Legacy model, no download URL
            checksum=None,
            input_size=(224, 224),
            score_threshold=0.6,
            max_detections=10
        ),
        'efficientdet_face': EdgeTPUModelInfo(
            name='EfficientDet Face',
            file_path='models/efficientdet_face_edgetpu.tflite',
            url=None,  # No public URL available
            checksum=None,
            input_size=(320, 320),
            score_threshold=0.7,
            max_detections=10
        ),
        'blazeface': EdgeTPUModelInfo(
            name='BlazeFace',
            file_path='models/blazeface_edgetpu.tflite',
            url=None,  # No public URL available
            checksum=None,
            input_size=(128, 128),
            score_threshold=0.5,
            max_detections=10
        ),
    }
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}
        
        logger.info(f"EdgeTPU model manager initialized with {len(self.AVAILABLE_MODELS)} available models")
    
    def get_model_info(self, model_name: str) -> Optional[EdgeTPUModelInfo]:
        """Get model information by name."""
        return self.AVAILABLE_MODELS.get(model_name)
    
    def list_available_models(self) -> List[str]:
        """List all available model names."""
        return list(self.AVAILABLE_MODELS.keys())
    
    def download_model(self, model_name: str, force_download: bool = False) -> bool:
        """Download EdgeTPU model if not present or if forced."""
        model_info = self.get_model_info(model_name)
        if not model_info:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        model_path = self.models_dir / model_info.file_path
        
        # Check if model already exists and is valid
        if model_path.exists() and not force_download:
            if self._verify_model_checksum(model_path, model_info.checksum):
                logger.info(f"Model {model_name} already exists and is valid")
                return True
            else:
                logger.warning(f"Model {model_name} exists but checksum mismatch, re-downloading")
        
        # Download model
        if not model_info.url:
            logger.error(f"No download URL available for model {model_name}")
            return False
        
        try:
            logger.info(f"Downloading EdgeTPU model: {model_name}")
            logger.info(f"URL: {model_info.url}")
            
            # Create model directory
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress
            response = requests.get(model_info.url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            logger.debug(f"Download progress: {progress:.1f}%")
            
            # Verify checksum if provided
            if model_info.checksum:
                if self._verify_model_checksum(model_path, model_info.checksum):
                    logger.info(f"Model {model_name} downloaded and verified successfully")
                    return True
                else:
                    logger.error(f"Model {model_name} download failed checksum verification")
                    model_path.unlink()  # Delete invalid file
                    return False
            else:
                logger.info(f"Model {model_name} downloaded (no checksum verification)")
                return True
                
        except Exception as e:
            logger.error(f"Failed to download model {model_name}: {e}")
            if model_path.exists():
                model_path.unlink()  # Clean up partial download
            return False
    
    def _verify_model_checksum(self, model_path: Path, expected_checksum: Optional[str]) -> bool:
        """Verify model file checksum."""
        if not expected_checksum:
            return True  # No checksum to verify
        
        try:
            with open(model_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            if file_hash == expected_checksum:
                logger.debug(f"Checksum verification passed for {model_path.name}")
                return True
            else:
                logger.error(f"Checksum mismatch for {model_path.name}")
                logger.error(f"Expected: {expected_checksum}")
                logger.error(f"Got: {file_hash}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to verify checksum for {model_path.name}: {e}")
            return False
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get local path to model file."""
        model_info = self.get_model_info(model_name)
        if not model_info:
            return None
        
        model_path = self.models_dir / model_info.file_path
        return model_path if model_path.exists() else None


class EdgeTPUInferenceEngine:
    """High-performance EdgeTPU inference engine with optimizations."""
    
    def __init__(self, model_path: str, model_info: EdgeTPUModelInfo):
        self.model_path = model_path
        self.model_info = model_info
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        # Performance metrics
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.last_inference_time = 0.0
        
        # Threading for thread safety
        self._lock = threading.Lock()
        
        logger.info(f"EdgeTPU inference engine created for {model_info.name}")
    
    def initialize(self) -> bool:
        """Initialize EdgeTPU interpreter."""
        try:
            if not EDGETPU_AVAILABLE:
                logger.error("EdgeTPU libraries not available")
                return False
            
            # List available EdgeTPU devices
            edge_tpu_devices = edgetpu.list_edge_tpus()
            if not edge_tpu_devices:
                logger.error("No EdgeTPU devices found")
                return False
            
            logger.info(f"Found {len(edge_tpu_devices)} EdgeTPU device(s)")
            for i, device in enumerate(edge_tpu_devices):
                logger.info(f"EdgeTPU {i}: {device}")
            
            # Create interpreter with EdgeTPU delegate
            self.interpreter = tflite.Interpreter(
                model_path=self.model_path,
                experimental_delegates=[
                    tflite.load_delegate('libedgetpu.so.1')
                ]
            )
            
            # Allocate tensors
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            logger.info(f"EdgeTPU interpreter initialized successfully")
            logger.info(f"Model input shape: {self.input_details[0]['shape']}")
            logger.info(f"Model output tensors: {len(self.output_details)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize EdgeTPU interpreter: {e}")
            return False
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for EdgeTPU inference."""
        target_size = self.model_info.input_size
        
        # Resize image
        resized_image = cv2.resize(image, target_size)
        
        # Convert BGR to RGB if needed
        if len(resized_image.shape) == 3 and resized_image.shape[2] == 3:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # For quantized models, keep as uint8
        input_tensor = np.expand_dims(resized_image, axis=0).astype(np.uint8)
        
        return input_tensor
    
    def run_inference(self, preprocessed_image: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference on EdgeTPU."""
        with self._lock:
            start_time = time.time()
            
            try:
                # Set input tensor
                self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_image)
                
                # Run inference
                self.interpreter.invoke()
                
                # Get output tensors
                outputs = {}
                for output_detail in self.output_details:
                    tensor_name = output_detail['name']
                    tensor_data = self.interpreter.get_tensor(output_detail['index'])
                    outputs[tensor_name] = tensor_data
                
                # Update performance metrics
                inference_time = time.time() - start_time
                self.last_inference_time = inference_time
                self.total_inference_time += inference_time
                self.inference_count += 1
                
                logger.debug(f"EdgeTPU inference completed in {inference_time*1000:.2f}ms")
                
                return outputs
                
            except Exception as e:
                logger.error(f"EdgeTPU inference failed: {e}")
                raise
    
    def postprocess_detections(self, outputs: Dict[str, np.ndarray], 
                             original_shape: Tuple[int, int]) -> List[DetectionResult]:
        """Postprocess inference outputs to detection results."""
        detections = []
        
        try:
            # Extract detection arrays (format may vary by model)
            if 'detection_boxes' in outputs:
                # Standard SSD format
                boxes = outputs['detection_boxes'][0]
                scores = outputs['detection_scores'][0]
                classes = outputs['detection_classes'][0] if 'detection_classes' in outputs else None
                
                # Filter by score threshold
                valid_detections = scores > self.model_info.score_threshold
                
                for i in range(min(len(boxes), self.model_info.max_detections)):
                    if not valid_detections[i]:
                        break
                    
                    # Convert normalized coordinates to pixel coordinates
                    y1, x1, y2, x2 = boxes[i]
                    
                    x1 = int(x1 * original_shape[1])
                    y1 = int(y1 * original_shape[0])
                    x2 = int(x2 * original_shape[1])
                    y2 = int(y2 * original_shape[0])
                    
                    # Create detection result
                    detection = DetectionResult(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(scores[i]),
                        inference_time=self.last_inference_time
                    )
                    
                    detections.append(detection)
            
            else:
                # Try to infer output format
                output_arrays = list(outputs.values())
                if len(output_arrays) >= 2:
                    # Assume first is boxes, second is scores
                    boxes = output_arrays[0][0]
                    scores = output_arrays[1][0]
                    
                    # Process detections
                    for i in range(min(len(boxes), self.model_info.max_detections)):
                        if len(scores) > i and scores[i] > self.model_info.score_threshold:
                            # Convert coordinates (assuming normalized)
                            if len(boxes[i]) >= 4:
                                x1, y1, x2, y2 = boxes[i][:4]
                                
                                x1 = int(x1 * original_shape[1])
                                y1 = int(y1 * original_shape[0])
                                x2 = int(x2 * original_shape[1])
                                y2 = int(y2 * original_shape[0])
                                
                                detection = DetectionResult(
                                    bbox=(x1, y1, x2, y2),
                                    confidence=float(scores[i]),
                                    inference_time=self.last_inference_time
                                )
                                
                                detections.append(detection)
            
            # Apply Non-Maximum Suppression if multiple detections
            if len(detections) > 1:
                detections = self._apply_nms(detections)
            
            logger.debug(f"Postprocessing completed: {len(detections)} detections")
            
        except Exception as e:
            logger.error(f"Postprocessing failed: {e}")
        
        return detections
    
    def _apply_nms(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Apply Non-Maximum Suppression to remove duplicate detections."""
        if len(detections) <= 1:
            return detections
        
        try:
            # Convert to OpenCV format for NMS
            boxes = []
            scores = []
            
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                boxes.append([x1, y1, x2 - x1, y2 - y1])  # Convert to (x, y, w, h)
                scores.append(detection.confidence)
            
            boxes = np.array(boxes, dtype=np.float32)
            scores = np.array(scores, dtype=np.float32)
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), scores.tolist(), 
                self.model_info.score_threshold, 
                self.model_info.iou_threshold
            )
            
            # Filter detections by NMS results
            if len(indices) > 0:
                indices = indices.flatten()
                return [detections[i] for i in indices]
            else:
                return []
                
        except Exception as e:
            logger.error(f"NMS failed: {e}")
            return detections  # Return original if NMS fails
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get inference performance statistics."""
        avg_inference_time = (self.total_inference_time / max(1, self.inference_count))
        
        return {
            'inference_count': self.inference_count,
            'total_inference_time': self.total_inference_time,
            'average_inference_time': avg_inference_time,
            'last_inference_time': self.last_inference_time,
            'fps_theoretical': 1.0 / max(0.001, avg_inference_time),
            'model_name': self.model_info.name,
            'input_size': self.model_info.input_size
        }


class EdgeTPUDetector(BaseDetector):
    """Production EdgeTPU face detector with comprehensive optimizations."""
    
    def __init__(self, config: Dict[str, Any]):
        # Configuration
        self.model_name = config.get('model', 'mobilenet_face')
        self.auto_download = config.get('auto_download', False)  # Default False for safety
        self.force_download = config.get('force_download', False)
        self.device_path = config.get('device_path', None)
        self.enable_monitoring = config.get('enable_monitoring', True)
        self.temperature_limit = config.get('temperature_limit', 85.0)
        
        # Components
        self.model_manager = EdgeTPUModelManager(config.get('models_dir', 'models'))
        self.inference_engine = None
        self.model_info = None
        
        # State
        self.is_initialized = False
        self.initialization_error = None
        
        # Performance tracking
        self.inference_times = []
        self.temperature_readings = []
        self.throttle_events = 0
        
        # Initialize base detector
        super().__init__(config)
        
        logger.info(f"EdgeTPU detector created with model: {self.model_name}")
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if EdgeTPU hardware and libraries are available."""
        if not EDGETPU_AVAILABLE:
            return False
        
        try:
            edge_tpu_devices = edgetpu.list_edge_tpus()
            return len(edge_tpu_devices) > 0
        except Exception:
            return False
    
    def _get_detector_type(self) -> DetectorType:
        """Return EdgeTPU detector type."""
        return DetectorType.EDGETPU
    
    def _initialize_model(self) -> None:
        """Load EdgeTPU model and initialize inference engine."""
        try:
            # Check EdgeTPU availability
            if not self.is_available():
                self.initialization_error = "EdgeTPU hardware not available"
                logger.error(self.initialization_error)
                raise RuntimeError(self.initialization_error)
            
            # Get model information
            self.model_info = self.model_manager.get_model_info(self.model_name)
            if not self.model_info:
                self.initialization_error = f"Unknown model: {self.model_name}"
                logger.error(self.initialization_error)
                raise RuntimeError(self.initialization_error)
            
            # Download model if needed and auto_download is enabled
            if self.auto_download:
                if not self.model_manager.download_model(self.model_name, self.force_download):
                    self.initialization_error = f"Failed to download model: {self.model_name}"
                    logger.error(self.initialization_error)
                    raise RuntimeError(self.initialization_error)
            
            # Get model path
            model_path = self.model_manager.get_model_path(self.model_name)
            if not model_path or not model_path.exists():
                self.initialization_error = f"Model file not found: {self.model_name}"
                logger.error(self.initialization_error)
                raise RuntimeError(self.initialization_error)
            
            # Create and initialize inference engine
            self.inference_engine = EdgeTPUInferenceEngine(str(model_path), self.model_info)
            
            if not self.inference_engine.initialize():
                self.initialization_error = "Failed to initialize EdgeTPU inference engine"
                logger.error(self.initialization_error)
                raise RuntimeError(self.initialization_error)
            
            self.is_initialized = True
            logger.info(f"EdgeTPU detector initialized successfully with {self.model_name}")
            
        except Exception as e:
            self.initialization_error = f"EdgeTPU initialization failed: {e}"
            logger.error(self.initialization_error)
            raise RuntimeError(self.initialization_error) from e
    
    def _run_inference(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """Run face detection inference on single image."""
        if not self.is_initialized:
            logger.error("EdgeTPU detector not initialized")
            return []
        
        try:
            # Check temperature if monitoring enabled
            if self.enable_monitoring:
                self._check_temperature()
            
            # Preprocess image
            preprocessed_image = self.inference_engine.preprocess_image(image)
            
            # Run inference
            start_time = time.time()
            outputs = self.inference_engine.run_inference(preprocessed_image)
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Postprocess results
            detections = self.inference_engine.postprocess_detections(
                outputs, (image.shape[0], image.shape[1])
            )
            
            # Convert to FaceDetectionResult format
            results = []
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                
                # Convert to (top, right, bottom, left) format
                result = FaceDetectionResult(
                    bounding_box=(y1, x2, y2, x1),
                    confidence=detection.confidence,
                    landmarks=None,
                    quality_score=0.0
                )
                
                results.append(result)
            
            logger.debug(f"EdgeTPU face detection completed in {inference_time*1000:.2f}ms, found {len(results)} faces")
            
            return results
            
        except Exception as e:
            logger.error(f"EdgeTPU face detection failed: {e}")
            return []
    
    def _check_temperature(self) -> None:
        """Check EdgeTPU temperature and throttle if needed."""
        try:
            # Temperature monitoring would require additional pycoral APIs
            # For now, this is a placeholder
            pass
            
        except Exception as e:
            logger.debug(f"Temperature check failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information."""
        if not self.model_info:
            return {}
        
        return {
            'name': self.model_info.name,
            'file_path': self.model_info.file_path,
            'input_size': self.model_info.input_size,
            'score_threshold': self.model_info.score_threshold,
            'max_detections': self.model_info.max_detections,
            'checksum': self.model_info.checksum
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get EdgeTPU-specific performance metrics."""
        base_metrics = self.get_performance_stats()
        
        edgetpu_metrics = {
            'device_path': self.device_path or 'auto',
            'model': self.model_name,
            'backend': 'tflite',
            'quantized': True,
            'average_inference_time_ms': (
                np.mean(self.inference_times) * 1000 if self.inference_times else 0.0
            ),
            'fps': (
                1.0 / np.mean(self.inference_times) if self.inference_times else 0.0
            ),
            'throttle_events': self.throttle_events,
            'is_initialized': self.is_initialized,
            'initialization_error': self.initialization_error,
            'hardware_available': self.is_available()
        }
        
        if self.temperature_readings:
            edgetpu_metrics['avg_temperature_c'] = np.mean(self.temperature_readings)
            edgetpu_metrics['max_temperature_c'] = np.max(self.temperature_readings)
        
        return {**base_metrics, **edgetpu_metrics}
    
    def benchmark(self, test_images: List[np.ndarray], iterations: int = 10) -> Dict[str, Any]:
        """Run performance benchmark on test images."""
        if not self.is_initialized:
            return {'error': 'Detector not initialized'}
        
        logger.info(f"Starting EdgeTPU benchmark with {len(test_images)} images, {iterations} iterations")
        
        results = {
            'total_iterations': iterations * len(test_images),
            'inference_times': [],
            'detection_counts': [],
            'errors': 0
        }
        
        for iteration in range(iterations):
            for i, image in enumerate(test_images):
                try:
                    start_time = time.time()
                    detections, metrics = self.detect_faces(image)
                    end_time = time.time()
                    
                    inference_time = end_time - start_time
                    results['inference_times'].append(inference_time)
                    results['detection_counts'].append(len(detections))
                    
                    logger.debug(f"Iteration {iteration+1}, Image {i+1}: {inference_time*1000:.2f}ms, {len(detections)} faces")
                    
                except Exception as e:
                    logger.error(f"Benchmark error: {e}")
                    results['errors'] += 1
        
        # Calculate statistics
        if results['inference_times']:
            inference_times = results['inference_times']
            results['statistics'] = {
                'mean_inference_time': np.mean(inference_times),
                'median_inference_time': np.median(inference_times),
                'min_inference_time': np.min(inference_times),
                'max_inference_time': np.max(inference_times),
                'std_inference_time': np.std(inference_times),
                'mean_fps': 1.0 / np.mean(inference_times),
                'mean_detections': np.mean(results['detection_counts']),
                'error_rate': results['errors'] / results['total_iterations']
            }
        
        logger.info(f"EdgeTPU benchmark completed: {results['statistics']['mean_inference_time']*1000:.2f}ms avg, {results['statistics']['mean_fps']:.1f} FPS")
        
        return results
    
    def cleanup(self) -> None:
        """Cleanup EdgeTPU resources."""
        super().cleanup()
        
        try:
            if hasattr(self, 'inference_engine') and self.inference_engine:
                del self.inference_engine
                self.inference_engine = None
            
            logger.debug("EdgeTPU detector cleanup completed")
            
        except Exception as e:
            logger.warning(f"EdgeTPU cleanup failed: {e}")


# Hardware detection utilities
def detect_edgetpu_devices() -> List[Dict[str, Any]]:
    """Detect available EdgeTPU devices."""
    devices = []
    
    if not EDGETPU_AVAILABLE:
        logger.warning("EdgeTPU libraries not available")
        return devices
    
    try:
        edge_tpu_devices = edgetpu.list_edge_tpus()
        
        for i, device in enumerate(edge_tpu_devices):
            device_info = {
                'id': i,
                'type': device.get('type', 'unknown'),
                'path': device.get('path', 'unknown')
            }
            devices.append(device_info)
        
        logger.info(f"Found {len(devices)} EdgeTPU device(s)")
        
    except Exception as e:
        logger.error(f"Failed to detect EdgeTPU devices: {e}")
    
    return devices


def is_edgetpu_available() -> bool:
    """Check if EdgeTPU is available on the system."""
    if not EDGETPU_AVAILABLE:
        return False
    
    try:
        edge_tpu_devices = edgetpu.list_edge_tpus()
        return len(edge_tpu_devices) > 0
    except Exception:
        return False
