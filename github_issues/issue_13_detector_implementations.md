# Issue #13: Detector Implementations with Hardware Acceleration

## 📋 **Overview**

Implement concrete detector strategies for GPU and EdgeTPU hardware acceleration, completing the pluggable detector framework. This issue provides high-performance face detection backends that significantly improve processing speed and efficiency for production deployments, especially on edge devices and GPU-enabled systems.

## 🎯 **Objectives**

### **Primary Goals**
1. **GPU Detector Implementation**: CUDA-accelerated face detection with TensorFlow/PyTorch backends
2. **EdgeTPU Detector Implementation**: Coral Edge TPU optimized detection for ultra-low latency
3. **Performance Benchmarking**: Comprehensive performance comparison across all detector types
4. **Hardware Optimization**: Device-specific optimizations and tuning
5. **Production Readiness**: Robust error handling, fallback mechanisms, and monitoring

### **Success Criteria**
- GPU detector achieves 5-10x speedup over CPU on compatible hardware
- EdgeTPU detector provides sub-100ms inference time on Coral devices
- Automatic hardware detection and optimal detector selection
- Seamless fallback to CPU when specialized hardware unavailable
- Comprehensive benchmarking suite with performance metrics
- Production-ready error handling and resource management

## 🏗️ **Architecture Requirements**

### **Detector Strategy Hierarchy**
```
BaseDetector (Abstract)
├── CPUDetector (✅ Implemented)
├── GPUDetector (🔄 This Issue)
├── EdgeTPUDetector (🔄 This Issue)
└── MockDetector (✅ Implemented)
```

### **Hardware Detection Priority**
```
Auto-Detection Order: EdgeTPU → GPU → CPU → Mock
Performance Tiers: EdgeTPU (fastest) → GPU (fast) → CPU (baseline) → Mock (testing)
```

### **Integration Flow**
```
DetectorFactory.auto_detect() → Hardware Check → Optimal Detector → Fallback Chain
```

## 📁 **Implementation Specifications**

### **Files to Create/Modify**

#### **New Files**
```
src/detectors/gpu_detector.py                    # GPU-accelerated face detection
src/detectors/edgetpu_detector.py               # Coral EdgeTPU optimized detection
src/detectors/hardware_detector.py              # Hardware capability detection
src/detectors/performance_profiler.py           # Detector performance profiling
src/detectors/model_manager.py                  # Model download and caching management

tests/test_gpu_detector.py                      # GPU detector tests
tests/test_edgetpu_detector.py                  # EdgeTPU detector tests
tests/test_hardware_detection.py               # Hardware detection tests
tests/performance/detector_benchmarks.py        # Comprehensive benchmarking

config/detector_models.py                       # Model configuration and paths
docs/detectors/                                 # Detector-specific documentation
    gpu_setup.md                                # GPU setup and troubleshooting
    edgetpu_setup.md                            # EdgeTPU setup and installation
    performance_guide.md                        # Performance optimization guide
```

#### **Modified Files**
```
src/detectors/detector_factory.py               # Enhanced with new detectors
src/detectors/base_detector.py                  # Extended base functionality
config/pipeline_config.py                       # Detector-specific configurations
requirements-gpu.txt                            # GPU-specific dependencies
requirements-edgetpu.txt                        # EdgeTPU-specific dependencies
```

### **Core Component: GPU Detector**
```python
#!/usr/bin/env python3
"""
GPU-Accelerated Face Detector

High-performance face detection using GPU acceleration with TensorFlow/ONNX backends.
Optimized for NVIDIA CUDA and supports batch processing for maximum throughput.
"""

import os
import logging
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path

import numpy as np

from src.detectors.base_detector import (
    BaseDetector, 
    DetectorType, 
    ModelType, 
    FaceDetectionResult, 
    DetectionMetrics
)
from src.detectors.model_manager import ModelManager

logger = logging.getLogger(__name__)


class GPUDetector(BaseDetector):
    """
    GPU-accelerated face detector using TensorFlow/ONNX.
    
    Features:
    - CUDA acceleration for 5-10x speedup
    - Batch processing support
    - Multiple model backends (TensorFlow, ONNX)
    - Memory optimization and pooling
    - Automatic mixed precision (AMP)
    """
    
    # Supported models and their configurations
    SUPPORTED_MODELS = {
        'retinaface': {
            'model_path': 'models/retinaface_gpu.onnx',
            'input_size': (640, 640),
            'confidence_threshold': 0.7,
            'nms_threshold': 0.4,
            'backend': 'onnx'
        },
        'mtcnn': {
            'model_path': 'models/mtcnn_gpu.pb',
            'input_size': (224, 224),
            'confidence_threshold': 0.6,
            'backend': 'tensorflow'
        },
        'yolov5_face': {
            'model_path': 'models/yolov5_face_gpu.onnx',
            'input_size': (416, 416),
            'confidence_threshold': 0.5,
            'nms_threshold': 0.5,
            'backend': 'onnx'
        }
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GPU detector.
        
        Args:
            config: Detector configuration including model, device, and optimization settings
        """
        # GPU-specific configuration
        self.device = config.get('device', 'cuda:0')
        self.model_name = config.get('model', 'retinaface')
        self.batch_size = config.get('batch_size', 4)
        self.enable_amp = config.get('enable_amp', True)  # Automatic Mixed Precision
        self.memory_fraction = config.get('memory_fraction', 0.7)
        
        # Model configuration
        if self.model_name not in self.SUPPORTED_MODELS:
            logger.warning(f"Unknown GPU model '{self.model_name}', defaulting to 'retinaface'")
            self.model_name = 'retinaface'
        
        self.model_config = self.SUPPORTED_MODELS[self.model_name]
        self.backend = self.model_config['backend']
        
        # Performance tracking
        self.batch_buffer = []
        self.inference_times = []
        self.memory_usage = []
        
        # Initialize base detector
        super().__init__(config)
        
        logger.info(f"Initialized GPU detector: {self.model_name} on {self.device}")
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if GPU acceleration is available."""
        try:
            # Check CUDA availability
            if not cls._check_cuda():
                return False
            
            # Check backend availability
            if not cls._check_backends():
                return False
            
            # Check for compatible GPU
            if not cls._check_gpu_compatibility():
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"GPU detector availability check failed: {e}")
            return False
    
    @classmethod
    def _check_cuda(cls) -> bool:
        """Check CUDA availability."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                import tensorflow as tf
                return len(tf.config.list_physical_devices('GPU')) > 0
            except ImportError:
                return False
    
    @classmethod
    def _check_backends(cls) -> bool:
        """Check if required backends are available."""
        backends = []
        
        # Check TensorFlow
        try:
            import tensorflow as tf
            backends.append('tensorflow')
        except ImportError:
            pass
        
        # Check ONNX Runtime with GPU
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in providers:
                backends.append('onnx')
        except ImportError:
            pass
        
        return len(backends) > 0
    
    @classmethod
    def _check_gpu_compatibility(cls) -> bool:
        """Check GPU compatibility and memory."""
        try:
            # Check minimum GPU memory (2GB)
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                min_memory = 2 * 1024**3  # 2GB
                return gpu_memory >= min_memory
            return False
        except Exception:
            return False
    
    def _get_detector_type(self) -> DetectorType:
        """Return GPU detector type."""
        return DetectorType.GPU
    
    def _initialize_model(self) -> None:
        """Initialize GPU model and inference session."""
        try:
            # Download/cache model if needed
            model_manager = ModelManager()
            model_path = model_manager.get_model(
                self.model_name, 
                self.model_config['model_path']
            )
            
            if self.backend == 'onnx':
                self._initialize_onnx_model(model_path)
            elif self.backend == 'tensorflow':
                self._initialize_tensorflow_model(model_path)
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
            
            # Configure GPU memory
            self._configure_gpu_memory()
            
            # Warm up model
            self._warmup_model()
            
            logger.info(f"GPU detector initialized: {self.model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GPU model: {e}")
            raise RuntimeError(f"GPU detector initialization failed: {e}")
    
    def _initialize_onnx_model(self, model_path: Path) -> None:
        """Initialize ONNX Runtime model."""
        import onnxruntime as ort
        
        # Configure providers with GPU acceleration
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': int(self.device.split(':')[1]) if ':' in self.device else 0,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'  # Fallback
        ]
        
        # Create inference session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        if self.enable_amp:
            session_options.add_session_config_entry('session.use_env_allocators', '1')
        
        self.session = ort.InferenceSession(
            str(model_path),
            providers=providers,
            sess_options=session_options
        )
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        logger.debug(f"ONNX model loaded: {model_path}")
        logger.debug(f"Input shape: {self.input_shape}, Outputs: {self.output_names}")
    
    def _initialize_tensorflow_model(self, model_path: Path) -> None:
        """Initialize TensorFlow model."""
        import tensorflow as tf
        
        # Configure GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                # Set memory limit if specified
                if self.memory_fraction < 1.0:
                    memory_limit = int(tf.config.experimental.get_memory_info(gpus[0])['total'] * self.memory_fraction)
                    tf.config.experimental.set_memory_growth(gpus[0], False)
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
                    )
            except RuntimeError as e:
                logger.warning(f"GPU configuration failed: {e}")
        
        # Load model
        with tf.device(f'/GPU:0'):
            self.model = tf.saved_model.load(str(model_path))
            
        logger.debug(f"TensorFlow model loaded: {model_path}")
    
    def _configure_gpu_memory(self) -> None:
        """Configure GPU memory management."""
        if self.backend == 'onnx':
            # ONNX memory optimization
            pass  # Handled in session creation
        elif self.backend == 'tensorflow':
            # TensorFlow memory optimization
            import tensorflow as tf
            tf.config.experimental.enable_memory_growth = True
    
    def _warmup_model(self) -> None:
        """Warm up model with dummy inference."""
        try:
            # Create dummy input
            input_size = self.model_config['input_size']
            dummy_input = np.random.randint(0, 255, (1, input_size[1], input_size[0], 3), dtype=np.uint8)
            
            # Run warmup inference
            start_time = time.time()
            _ = self._run_inference_batch([dummy_input])
            warmup_time = time.time() - start_time
            
            logger.debug(f"Model warmup completed in {warmup_time:.3f}s")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def _run_inference(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """
        Run face detection inference on single image.
        
        Args:
            image: Input image in RGB format
            
        Returns:
            List of face detection results
        """
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Run batch inference with single image
            results = self._run_inference_batch([processed_image])
            
            return results[0] if results else []
            
        except Exception as e:
            logger.error(f"GPU inference failed: {e}")
            return []
    
    def _run_inference_batch(self, images: List[np.ndarray]) -> List[List[FaceDetectionResult]]:
        """
        Run batch inference on multiple images.
        
        Args:
            images: List of preprocessed images
            
        Returns:
            List of detection results for each image
        """
        try:
            if not images:
                return []
            
            # Prepare batch
            batch = np.array(images)
            
            # Run inference based on backend
            if self.backend == 'onnx':
                outputs = self.session.run(self.output_names, {self.input_name: batch})
            elif self.backend == 'tensorflow':
                import tensorflow as tf
                with tf.device(f'/GPU:0'):
                    outputs = self.model(batch)
                    if isinstance(outputs, dict):
                        outputs = [outputs[key].numpy() for key in sorted(outputs.keys())]
                    else:
                        outputs = [outputs.numpy()]
            
            # Post-process results
            batch_results = []
            for i in range(len(images)):
                # Extract detections for this image
                image_outputs = [output[i] if len(output.shape) > 1 else output for output in outputs]
                detections = self._postprocess_detections(image_outputs, images[i].shape)
                batch_results.append(detections)
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            return [[] for _ in images]
    
    def _postprocess_detections(self, outputs: List[np.ndarray], original_shape: Tuple[int, int, int]) -> List[FaceDetectionResult]:
        """
        Post-process raw model outputs to face detection results.
        
        Args:
            outputs: Raw model outputs
            original_shape: Original image shape (H, W, C)
            
        Returns:
            List of face detection results
        """
        try:
            if self.model_name == 'retinaface':
                return self._postprocess_retinaface(outputs, original_shape)
            elif self.model_name == 'mtcnn':
                return self._postprocess_mtcnn(outputs, original_shape)
            elif self.model_name == 'yolov5_face':
                return self._postprocess_yolov5(outputs, original_shape)
            else:
                logger.error(f"Unknown model for postprocessing: {self.model_name}")
                return []
                
        except Exception as e:
            logger.error(f"Postprocessing failed: {e}")
            return []
    
    def _postprocess_retinaface(self, outputs: List[np.ndarray], original_shape: Tuple[int, int, int]) -> List[FaceDetectionResult]:
        """Post-process RetinaFace outputs."""
        try:
            # RetinaFace outputs: [boxes, scores, landmarks]
            boxes = outputs[0]  # [N, 4] - (x1, y1, x2, y2)
            scores = outputs[1]  # [N] - confidence scores
            landmarks = outputs[2] if len(outputs) > 2 else None  # [N, 10] - 5 landmarks * 2
            
            confidence_threshold = self.model_config['confidence_threshold']
            nms_threshold = self.model_config['nms_threshold']
            
            # Filter by confidence
            valid_indices = scores > confidence_threshold
            if not np.any(valid_indices):
                return []
            
            filtered_boxes = boxes[valid_indices]
            filtered_scores = scores[valid_indices]
            filtered_landmarks = landmarks[valid_indices] if landmarks is not None else None
            
            # Apply NMS
            nms_indices = self._apply_nms(filtered_boxes, filtered_scores, nms_threshold)
            
            # Convert to detection results
            detections = []
            input_size = self.model_config['input_size']
            scale_x = original_shape[1] / input_size[0]
            scale_y = original_shape[0] / input_size[1]
            
            for idx in nms_indices:
                box = filtered_boxes[idx]
                score = filtered_scores[idx]
                
                # Scale coordinates back to original image size
                x1 = int(box[0] * scale_x)
                y1 = int(box[1] * scale_y)
                x2 = int(box[2] * scale_x)
                y2 = int(box[3] * scale_y)
                
                # Extract landmarks if available
                face_landmarks = None
                if filtered_landmarks is not None:
                    landmark_points = filtered_landmarks[idx].reshape(-1, 2)
                    face_landmarks = {
                        'left_eye': (int(landmark_points[0][0] * scale_x), int(landmark_points[0][1] * scale_y)),
                        'right_eye': (int(landmark_points[1][0] * scale_x), int(landmark_points[1][1] * scale_y)),
                        'nose': (int(landmark_points[2][0] * scale_x), int(landmark_points[2][1] * scale_y)),
                        'left_mouth': (int(landmark_points[3][0] * scale_x), int(landmark_points[3][1] * scale_y)),
                        'right_mouth': (int(landmark_points[4][0] * scale_x), int(landmark_points[4][1] * scale_y))
                    }
                
                detection = FaceDetectionResult(
                    bounding_box=(y1, x2, y2, x1),  # (top, right, bottom, left)
                    confidence=float(score),
                    landmarks=face_landmarks,
                    quality_score=float(score)  # Use confidence as quality score
                )
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"RetinaFace postprocessing failed: {e}")
            return []
    
    def _postprocess_mtcnn(self, outputs: List[np.ndarray], original_shape: Tuple[int, int, int]) -> List[FaceDetectionResult]:
        """Post-process MTCNN outputs."""
        # Implementation for MTCNN postprocessing
        # Similar structure to RetinaFace but adapted for MTCNN output format
        pass
    
    def _postprocess_yolov5(self, outputs: List[np.ndarray], original_shape: Tuple[int, int, int]) -> List[FaceDetectionResult]:
        """Post-process YOLOv5 Face outputs."""
        # Implementation for YOLOv5 Face postprocessing
        # Adapted for YOLO output format with face-specific modifications
        pass
    
    def _apply_nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """Apply Non-Maximum Suppression."""
        try:
            # Calculate areas
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)
            
            # Sort by scores
            order = np.argsort(scores)[::-1]
            
            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                
                # Calculate IoU with remaining boxes
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
                
                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)
                intersection = w * h
                
                iou = intersection / (areas[i] + areas[order[1:]] - intersection)
                
                # Keep boxes with IoU below threshold
                indices = np.where(iou <= threshold)[0]
                order = order[indices + 1]
            
            return keep
            
        except Exception as e:
            logger.error(f"NMS failed: {e}")
            return list(range(len(boxes)))
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for GPU inference."""
        try:
            import cv2
            
            input_size = self.model_config['input_size']
            
            # Resize image
            resized = cv2.resize(image, input_size)
            
            # Normalize (model-specific)
            if self.model_name in ['retinaface', 'yolov5_face']:
                # Normalize to [0, 1]
                normalized = resized.astype(np.float32) / 255.0
            elif self.model_name == 'mtcnn':
                # Normalize to [-1, 1]
                normalized = (resized.astype(np.float32) - 127.5) / 127.5
            else:
                normalized = resized.astype(np.float32)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get GPU-specific performance metrics."""
        base_metrics = super().get_performance_metrics()
        
        gpu_metrics = {
            'device': self.device,
            'model': self.model_name,
            'backend': self.backend,
            'batch_size': self.batch_size,
            'average_inference_time': np.mean(self.inference_times) if self.inference_times else 0.0,
            'gpu_memory_usage': self._get_gpu_memory_usage(),
            'batch_efficiency': len(self.inference_times) / max(1, len(self.inference_times) / self.batch_size)
        }
        
        return {**base_metrics, **gpu_metrics}
    
    def _get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        try:
            if self.backend == 'onnx':
                # ONNX Runtime doesn't provide direct memory info
                return {'allocated': 0.0, 'cached': 0.0}
            elif self.backend == 'tensorflow':
                import tensorflow as tf
                if tf.config.list_physical_devices('GPU'):
                    gpu_info = tf.config.experimental.get_memory_info('GPU:0')
                    return {
                        'allocated': gpu_info['current'] / 1024**2,  # MB
                        'peak': gpu_info['peak'] / 1024**2  # MB
                    }
            
            return {'allocated': 0.0, 'cached': 0.0}
            
        except Exception as e:
            logger.debug(f"Failed to get GPU memory usage: {e}")
            return {'allocated': 0.0, 'cached': 0.0}
    
    def cleanup(self) -> None:
        """Cleanup GPU resources."""
        super().cleanup()
        
        try:
            if hasattr(self, 'session'):
                # ONNX cleanup
                del self.session
            
            if hasattr(self, 'model'):
                # TensorFlow cleanup
                del self.model
                import tensorflow as tf
                tf.keras.backend.clear_session()
            
            # Clear GPU cache
            if self.backend == 'tensorflow':
                import tensorflow as tf
                tf.config.experimental.reset_memory_stats('GPU:0')
            
            logger.debug("GPU detector cleanup completed")
            
        except Exception as e:
            logger.warning(f"GPU cleanup failed: {e}")
```

### **Core Component: EdgeTPU Detector**
```python
#!/usr/bin/env python3
"""
Coral EdgeTPU Face Detector

Ultra-low latency face detection optimized for Google Coral Edge TPU.
Provides sub-100ms inference time with minimal power consumption.
"""

import os
import logging
import time
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import numpy as np

from src.detectors.base_detector import (
    BaseDetector, 
    DetectorType, 
    ModelType, 
    FaceDetectionResult, 
    DetectionMetrics
)
from src.detectors.model_manager import ModelManager

logger = logging.getLogger(__name__)


class EdgeTPUDetector(BaseDetector):
    """
    Coral EdgeTPU optimized face detector.
    
    Features:
    - Sub-100ms inference time
    - Minimal power consumption
    - Optimized TensorFlow Lite models
    - Hardware-specific quantization
    - Temperature monitoring and throttling
    """
    
    # EdgeTPU optimized models
    EDGETPU_MODELS = {
        'mobilenet_face': {
            'model_path': 'models/mobilenet_face_edgetpu.tflite',
            'input_size': (224, 224),
            'confidence_threshold': 0.6,
            'quantized': True
        },
        'efficientdet_face': {
            'model_path': 'models/efficientdet_face_edgetpu.tflite',
            'input_size': (320, 320),
            'confidence_threshold': 0.7,
            'quantized': True
        },
        'blazeface': {
            'model_path': 'models/blazeface_edgetpu.tflite',
            'input_size': (128, 128),
            'confidence_threshold': 0.5,
            'quantized': True
        }
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize EdgeTPU detector.
        
        Args:
            config: Detector configuration including model and device settings
        """
        # EdgeTPU-specific configuration
        self.model_name = config.get('model', 'mobilenet_face')
        self.device_path = config.get('device_path', None)  # Specific TPU device
        self.enable_monitoring = config.get('enable_monitoring', True)
        self.temperature_limit = config.get('temperature_limit', 85.0)  # Celsius
        
        # Model configuration
        if self.model_name not in self.EDGETPU_MODELS:
            logger.warning(f"Unknown EdgeTPU model '{self.model_name}', defaulting to 'mobilenet_face'")
            self.model_name = 'mobilenet_face'
        
        self.model_config = self.EDGETPU_MODELS[self.model_name]
        
        # Performance tracking
        self.inference_times = []
        self.temperature_readings = []
        self.throttle_events = 0
        
        # Initialize base detector
        super().__init__(config)
        
        logger.info(f"Initialized EdgeTPU detector: {self.model_name}")
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if Coral EdgeTPU is available."""
        try:
            # Check for pycoral library
            import pycoral
            from pycoral.utils.edgetpu import list_edge_tpus
            
            # Check for available EdgeTPU devices
            devices = list_edge_tpus()
            return len(devices) > 0
            
        except ImportError:
            logger.debug("pycoral library not available")
            return False
        except Exception as e:
            logger.debug(f"EdgeTPU availability check failed: {e}")
            return False
    
    def _get_detector_type(self) -> DetectorType:
        """Return EdgeTPU detector type."""
        return DetectorType.EDGETPU
    
    def _initialize_model(self) -> None:
        """Initialize EdgeTPU model and interpreter."""
        try:
            from pycoral.utils.edgetpu import make_interpreter
            from pycoral.utils.dataset import read_label_file
            
            # Download/cache model if needed
            model_manager = ModelManager()
            model_path = model_manager.get_model(
                self.model_name, 
                self.model_config['model_path']
            )
            
            # Create EdgeTPU interpreter
            self.interpreter = make_interpreter(
                str(model_path),
                device=self.device_path
            )
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Verify input shape
            expected_shape = self.input_details[0]['shape']
            logger.debug(f"EdgeTPU model input shape: {expected_shape}")
            
            # Initialize temperature monitoring
            if self.enable_monitoring:
                self._initialize_monitoring()
            
            # Warm up the model
            self._warmup_model()
            
            logger.info(f"EdgeTPU detector initialized: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize EdgeTPU model: {e}")
            raise RuntimeError(f"EdgeTPU detector initialization failed: {e}")
    
    def _initialize_monitoring(self) -> None:
        """Initialize EdgeTPU device monitoring."""
        try:
            from pycoral.utils.edgetpu import get_edge_tpu_temperature
            
            # Test temperature reading
            temp = get_edge_tpu_temperature()
            logger.debug(f"Initial EdgeTPU temperature: {temp}°C")
            
        except Exception as e:
            logger.warning(f"EdgeTPU monitoring initialization failed: {e}")
            self.enable_monitoring = False
    
    def _warmup_model(self) -> None:
        """Warm up EdgeTPU model."""
        try:
            # Create dummy input
            input_shape = self.input_details[0]['shape']
            dummy_input = np.random.randint(0, 255, input_shape, dtype=np.uint8)
            
            # Run warmup inference
            start_time = time.time()
            self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
            self.interpreter.invoke()
            warmup_time = time.time() - start_time
            
            logger.debug(f"EdgeTPU warmup completed in {warmup_time*1000:.1f}ms")
            
        except Exception as e:
            logger.warning(f"EdgeTPU warmup failed: {e}")
    
    def _run_inference(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """
        Run face detection inference on EdgeTPU.
        
        Args:
            image: Input image in RGB format
            
        Returns:
            List of face detection results
        """
        try:
            # Check temperature before inference
            if self.enable_monitoring and not self._check_temperature():
                logger.warning("EdgeTPU temperature too high, throttling")
                self.throttle_events += 1
                return []
            
            # Preprocess image
            input_tensor = self._preprocess_image_edgetpu(image)
            
            # Run inference
            start_time = time.time()
            
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Get outputs
            outputs = []
            for output_detail in self.output_details:
                output_data = self.interpreter.get_tensor(output_detail['index'])
                outputs.append(output_data)
            
            # Post-process results
            detections = self._postprocess_edgetpu_outputs(outputs, image.shape)
            
            logger.debug(f"EdgeTPU inference: {inference_time*1000:.1f}ms, {len(detections)} faces")
            
            return detections
            
        except Exception as e:
            logger.error(f"EdgeTPU inference failed: {e}")
            return []
    
    def _check_temperature(self) -> bool:
        """Check EdgeTPU temperature and prevent overheating."""
        try:
            from pycoral.utils.edgetpu import get_edge_tpu_temperature
            
            temp = get_edge_tpu_temperature()
            self.temperature_readings.append(temp)
            
            # Keep only recent readings
            if len(self.temperature_readings) > 100:
                self.temperature_readings = self.temperature_readings[-100:]
            
            return temp < self.temperature_limit
            
        except Exception as e:
            logger.debug(f"Temperature check failed: {e}")
            return True  # Allow inference if monitoring fails
    
    def _preprocess_image_edgetpu(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for EdgeTPU inference."""
        try:
            import cv2
            
            input_size = self.model_config['input_size']
            
            # Resize image
            resized = cv2.resize(image, input_size)
            
            # Convert to uint8 format (EdgeTPU models are typically quantized)
            if self.model_config['quantized']:
                processed = resized.astype(np.uint8)
            else:
                # Normalize to [0, 1] and convert to float32
                processed = (resized.astype(np.float32) / 255.0)
            
            # Add batch dimension
            return np.expand_dims(processed, axis=0)
            
        except Exception as e:
            logger.error(f"EdgeTPU preprocessing failed: {e}")
            return image
    
    def _postprocess_edgetpu_outputs(self, outputs: List[np.ndarray], original_shape: Tuple[int, int, int]) -> List[FaceDetectionResult]:
        """
        Post-process EdgeTPU model outputs.
        
        Args:
            outputs: Raw model outputs
            original_shape: Original image shape (H, W, C)
            
        Returns:
            List of face detection results
        """
        try:
            if self.model_name == 'mobilenet_face':
                return self._postprocess_mobilenet(outputs, original_shape)
            elif self.model_name == 'efficientdet_face':
                return self._postprocess_efficientdet(outputs, original_shape)
            elif self.model_name == 'blazeface':
                return self._postprocess_blazeface(outputs, original_shape)
            else:
                logger.error(f"Unknown EdgeTPU model for postprocessing: {self.model_name}")
                return []
                
        except Exception as e:
            logger.error(f"EdgeTPU postprocessing failed: {e}")
            return []
    
    def _postprocess_mobilenet(self, outputs: List[np.ndarray], original_shape: Tuple[int, int, int]) -> List[FaceDetectionResult]:
        """Post-process MobileNet face detection outputs."""
        try:
            # MobileNet SSD outputs: [locations, classifications, num_detections]
            locations = outputs[0][0]  # [N, 4] - normalized coordinates
            classifications = outputs[1][0]  # [N, num_classes] - class probabilities
            num_detections = int(outputs[2][0])  # Number of valid detections
            
            detections = []
            confidence_threshold = self.model_config['confidence_threshold']
            
            for i in range(min(num_detections, len(locations))):
                # Get face confidence (assuming class 1 is face)
                confidence = classifications[i][1] if len(classifications[i]) > 1 else classifications[i][0]
                
                if confidence < confidence_threshold:
                    continue
                
                # Extract normalized coordinates
                y1, x1, y2, x2 = locations[i]
                
                # Convert to absolute coordinates
                height, width = original_shape[:2]
                abs_x1 = int(x1 * width)
                abs_y1 = int(y1 * height)
                abs_x2 = int(x2 * width)
                abs_y2 = int(y2 * height)
                
                detection = FaceDetectionResult(
                    bounding_box=(abs_y1, abs_x2, abs_y2, abs_x1),  # (top, right, bottom, left)
                    confidence=float(confidence),
                    landmarks=None,  # MobileNet doesn't provide landmarks
                    quality_score=float(confidence)
                )
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"MobileNet postprocessing failed: {e}")
            return []
    
    def _postprocess_efficientdet(self, outputs: List[np.ndarray], original_shape: Tuple[int, int, int]) -> List[FaceDetectionResult]:
        """Post-process EfficientDet face detection outputs."""
        # Implementation for EfficientDet postprocessing
        # Similar to MobileNet but adapted for EfficientDet output format
        pass
    
    def _postprocess_blazeface(self, outputs: List[np.ndarray], original_shape: Tuple[int, int, int]) -> List[FaceDetectionResult]:
        """Post-process BlazeFace detection outputs."""
        # Implementation for BlazeFace postprocessing
        # Includes landmark detection capabilities
        pass
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get EdgeTPU-specific performance metrics."""
        base_metrics = super().get_performance_metrics()
        
        edgetpu_metrics = {
            'model': self.model_name,
            'device_path': self.device_path,
            'average_inference_time_ms': np.mean(self.inference_times) * 1000 if self.inference_times else 0.0,
            'min_inference_time_ms': np.min(self.inference_times) * 1000 if self.inference_times else 0.0,
            'max_inference_time_ms': np.max(self.inference_times) * 1000 if self.inference_times else 0.0,
            'current_temperature': self._get_current_temperature(),
            'average_temperature': np.mean(self.temperature_readings) if self.temperature_readings else 0.0,
            'throttle_events': self.throttle_events,
            'inferences_per_second': len(self.inference_times) / sum(self.inference_times) if self.inference_times else 0.0
        }
        
        return {**base_metrics, **edgetpu_metrics}
    
    def _get_current_temperature(self) -> float:
        """Get current EdgeTPU temperature."""
        try:
            if self.enable_monitoring:
                from pycoral.utils.edgetpu import get_edge_tpu_temperature
                return get_edge_tpu_temperature()
            return 0.0
        except Exception:
            return 0.0
    
    def cleanup(self) -> None:
        """Cleanup EdgeTPU resources."""
        super().cleanup()
        
        try:
            if hasattr(self, 'interpreter'):
                del self.interpreter
            
            logger.debug("EdgeTPU detector cleanup completed")
            
        except Exception as e:
            logger.warning(f"EdgeTPU cleanup failed: {e}")
```

### **Hardware Detection System**
```python
#!/usr/bin/env python3
"""
Hardware Detection and Capability Assessment

Comprehensive hardware detection for optimal detector selection.
"""

import logging
import platform
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class HardwareType(Enum):
    """Types of hardware accelerators."""
    CPU = "cpu"
    NVIDIA_GPU = "nvidia_gpu"
    AMD_GPU = "amd_gpu"
    INTEL_GPU = "intel_gpu"
    CORAL_EDGETPU = "coral_edgetpu"
    APPLE_NEURAL_ENGINE = "apple_neural_engine"


@dataclass
class HardwareCapability:
    """Hardware capability information."""
    hardware_type: HardwareType
    available: bool
    device_name: str
    memory_gb: float
    compute_capability: Optional[str] = None
    driver_version: Optional[str] = None
    performance_score: float = 1.0
    limitations: List[str] = None


class HardwareDetector:
    """Detect and assess hardware capabilities for face detection."""
    
    def __init__(self):
        """Initialize hardware detector."""
        self.capabilities: Dict[HardwareType, HardwareCapability] = {}
        self._detect_all_hardware()
    
    def get_optimal_detector_type(self) -> str:
        """Get the optimal detector type for current hardware."""
        # Priority order: EdgeTPU > GPU > CPU
        priority_order = [
            HardwareType.CORAL_EDGETPU,
            HardwareType.NVIDIA_GPU,
            HardwareType.AMD_GPU,
            HardwareType.INTEL_GPU,
            HardwareType.APPLE_NEURAL_ENGINE,
            HardwareType.CPU
        ]
        
        for hardware_type in priority_order:
            capability = self.capabilities.get(hardware_type)
            if capability and capability.available and capability.performance_score > 0.5:
                return self._hardware_type_to_detector(hardware_type)
        
        # Fallback to CPU
        return "cpu"
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get comprehensive hardware information."""
        return {
            'platform': platform.platform(),
            'architecture': platform.architecture(),
            'processor': platform.processor(),
            'capabilities': {
                hw_type.value: {
                    'available': cap.available,
                    'device_name': cap.device_name,
                    'memory_gb': cap.memory_gb,
                    'performance_score': cap.performance_score,
                    'limitations': cap.limitations or []
                }
                for hw_type, cap in self.capabilities.items()
            }
        }
    
    def _detect_all_hardware(self) -> None:
        """Detect all available hardware."""
        # Detect CPU
        self.capabilities[HardwareType.CPU] = self._detect_cpu()
        
        # Detect GPUs
        self.capabilities[HardwareType.NVIDIA_GPU] = self._detect_nvidia_gpu()
        self.capabilities[HardwareType.AMD_GPU] = self._detect_amd_gpu()
        self.capabilities[HardwareType.INTEL_GPU] = self._detect_intel_gpu()
        
        # Detect specialized hardware
        self.capabilities[HardwareType.CORAL_EDGETPU] = self._detect_coral_edgetpu()
        self.capabilities[HardwareType.APPLE_NEURAL_ENGINE] = self._detect_apple_neural_engine()
    
    def _detect_cpu(self) -> HardwareCapability:
        """Detect CPU capabilities."""
        try:
            import psutil
            
            # Get CPU info
            cpu_count = psutil.cpu_count(logical=False)
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # Calculate performance score based on cores and memory
            performance_score = min(1.0, (cpu_count * memory_gb) / 32.0)
            
            return HardwareCapability(
                hardware_type=HardwareType.CPU,
                available=True,
                device_name=platform.processor(),
                memory_gb=memory_gb,
                performance_score=performance_score
            )
            
        except Exception as e:
            logger.error(f"CPU detection failed: {e}")
            return HardwareCapability(
                hardware_type=HardwareType.CPU,
                available=True,
                device_name="Unknown CPU",
                memory_gb=4.0,
                performance_score=0.5
            )
    
    def _detect_nvidia_gpu(self) -> HardwareCapability:
        """Detect NVIDIA GPU capabilities."""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return HardwareCapability(
                    hardware_type=HardwareType.NVIDIA_GPU,
                    available=False,
                    device_name="No NVIDIA GPU",
                    memory_gb=0.0
                )
            
            # Get GPU properties
            device_count = torch.cuda.device_count()
            device_props = torch.cuda.get_device_properties(0)
            
            memory_gb = device_props.total_memory / (1024**3)
            compute_capability = f"{device_props.major}.{device_props.minor}"
            
            # Calculate performance score
            performance_score = min(1.0, memory_gb / 8.0)  # Normalize to 8GB
            
            limitations = []
            if memory_gb < 4.0:
                limitations.append("Low GPU memory may limit batch size")
            if float(compute_capability) < 6.0:
                limitations.append("Older compute capability may affect performance")
            
            return HardwareCapability(
                hardware_type=HardwareType.NVIDIA_GPU,
                available=True,
                device_name=device_props.name,
                memory_gb=memory_gb,
                compute_capability=compute_capability,
                performance_score=performance_score,
                limitations=limitations
            )
            
        except ImportError:
            return HardwareCapability(
                hardware_type=HardwareType.NVIDIA_GPU,
                available=False,
                device_name="PyTorch not available",
                memory_gb=0.0
            )
        except Exception as e:
            logger.error(f"NVIDIA GPU detection failed: {e}")
            return HardwareCapability(
                hardware_type=HardwareType.NVIDIA_GPU,
                available=False,
                device_name="Detection failed",
                memory_gb=0.0
            )
    
    def _detect_amd_gpu(self) -> HardwareCapability:
        """Detect AMD GPU capabilities."""
        # Implementation for AMD GPU detection (ROCm)
        return HardwareCapability(
            hardware_type=HardwareType.AMD_GPU,
            available=False,
            device_name="AMD GPU detection not implemented",
            memory_gb=0.0
        )
    
    def _detect_intel_gpu(self) -> HardwareCapability:
        """Detect Intel GPU capabilities."""
        # Implementation for Intel GPU detection
        return HardwareCapability(
            hardware_type=HardwareType.INTEL_GPU,
            available=False,
            device_name="Intel GPU detection not implemented",
            memory_gb=0.0
        )
    
    def _detect_coral_edgetpu(self) -> HardwareCapability:
        """Detect Coral EdgeTPU capabilities."""
        try:
            from pycoral.utils.edgetpu import list_edge_tpus, get_edge_tpu_name
            
            devices = list_edge_tpus()
            if not devices:
                return HardwareCapability(
                    hardware_type=HardwareType.CORAL_EDGETPU,
                    available=False,
                    device_name="No EdgeTPU devices found",
                    memory_gb=0.0
                )
            
            # Get first device info
            device_name = get_edge_tpu_name(devices[0]['path'])
            
            return HardwareCapability(
                hardware_type=HardwareType.CORAL_EDGETPU,
                available=True,
                device_name=device_name,
                memory_gb=0.0,  # EdgeTPU doesn't have traditional memory
                performance_score=1.0  # Highest performance for edge inference
            )
            
        except ImportError:
            return HardwareCapability(
                hardware_type=HardwareType.CORAL_EDGETPU,
                available=False,
                device_name="pycoral not available",
                memory_gb=0.0
            )
        except Exception as e:
            logger.error(f"EdgeTPU detection failed: {e}")
            return HardwareCapability(
                hardware_type=HardwareType.CORAL_EDGETPU,
                available=False,
                device_name="Detection failed",
                memory_gb=0.0
            )
    
    def _detect_apple_neural_engine(self) -> HardwareCapability:
        """Detect Apple Neural Engine capabilities."""
        try:
            # Check if running on Apple Silicon
            if platform.machine() not in ['arm64', 'aarch64']:
                return HardwareCapability(
                    hardware_type=HardwareType.APPLE_NEURAL_ENGINE,
                    available=False,
                    device_name="Not Apple Silicon",
                    memory_gb=0.0
                )
            
            # Try to import Core ML
            import coremltools
            
            return HardwareCapability(
                hardware_type=HardwareType.APPLE_NEURAL_ENGINE,
                available=True,
                device_name="Apple Neural Engine",
                memory_gb=0.0,  # Shared with system memory
                performance_score=0.9  # High performance for Apple devices
            )
            
        except ImportError:
            return HardwareCapability(
                hardware_type=HardwareType.APPLE_NEURAL_ENGINE,
                available=False,
                device_name="Core ML not available",
                memory_gb=0.0
            )
        except Exception as e:
            logger.error(f"Apple Neural Engine detection failed: {e}")
            return HardwareCapability(
                hardware_type=HardwareType.APPLE_NEURAL_ENGINE,
                available=False,
                device_name="Detection failed",
                memory_gb=0.0
            )
    
    def _hardware_type_to_detector(self, hardware_type: HardwareType) -> str:
        """Map hardware type to detector type."""
        mapping = {
            HardwareType.CPU: "cpu",
            HardwareType.NVIDIA_GPU: "gpu",
            HardwareType.AMD_GPU: "gpu",
            HardwareType.INTEL_GPU: "gpu",
            HardwareType.CORAL_EDGETPU: "edgetpu",
            HardwareType.APPLE_NEURAL_ENGINE: "coreml"  # Future implementation
        }
        return mapping.get(hardware_type, "cpu")
```

## 📊 **Performance Benchmarking**

### **Comprehensive Benchmarking Suite**
```python
#!/usr/bin/env python3
"""
Detector Performance Benchmarking Suite

Comprehensive benchmarking and comparison of all detector implementations.
"""

import time
import logging
import statistics
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import numpy as np

from src.detectors.detector_factory import DetectorFactory
from src.detectors.hardware_detector import HardwareDetector

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark result for a single detector."""
    detector_type: str
    model_name: str
    average_inference_time: float
    min_inference_time: float
    max_inference_time: float
    throughput_fps: float
    memory_usage_mb: float
    accuracy_score: float
    hardware_utilization: float


class DetectorBenchmark:
    """Comprehensive detector benchmarking system."""
    
    def __init__(self):
        """Initialize benchmarking system."""
        self.hardware_detector = HardwareDetector()
        self.test_images = self._load_test_images()
        self.ground_truth = self._load_ground_truth()
    
    def run_full_benchmark(self) -> Dict[str, BenchmarkResult]:
        """Run comprehensive benchmark on all available detectors."""
        results = {}
        
        # Get available detectors
        available_detectors = DetectorFactory.get_available_detectors()
        
        for detector_type in available_detectors:
            logger.info(f"Benchmarking {detector_type} detector...")
            
            try:
                result = self._benchmark_detector(detector_type)
                results[detector_type] = result
                
                logger.info(f"{detector_type}: {result.average_inference_time*1000:.1f}ms avg, "
                          f"{result.throughput_fps:.1f} FPS")
                
            except Exception as e:
                logger.error(f"Benchmark failed for {detector_type}: {e}")
        
        return results
    
    def _benchmark_detector(self, detector_type: str) -> BenchmarkResult:
        """Benchmark a single detector."""
        # Create detector
        config = self._get_detector_config(detector_type)
        detector = DetectorFactory.create(detector_type, config)
        
        try:
            # Warmup
            self._warmup_detector(detector)
            
            # Performance benchmark
            inference_times = []
            memory_usage = []
            
            for image in self.test_images:
                start_time = time.time()
                
                # Run detection
                faces, metrics = detector.detect_faces(image)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                memory_usage.append(metrics.memory_usage)
            
            # Calculate statistics
            avg_time = statistics.mean(inference_times)
            min_time = min(inference_times)
            max_time = max(inference_times)
            throughput = len(self.test_images) / sum(inference_times)
            avg_memory = statistics.mean(memory_usage) if memory_usage else 0.0
            
            # Accuracy benchmark
            accuracy = self._calculate_accuracy(detector)
            
            # Hardware utilization
            hw_util = self._measure_hardware_utilization(detector)
            
            return BenchmarkResult(
                detector_type=detector_type,
                model_name=getattr(detector, 'model_name', 'unknown'),
                average_inference_time=avg_time,
                min_inference_time=min_time,
                max_inference_time=max_time,
                throughput_fps=throughput,
                memory_usage_mb=avg_memory,
                accuracy_score=accuracy,
                hardware_utilization=hw_util
            )
            
        finally:
            detector.cleanup()
    
    def _load_test_images(self) -> List[np.ndarray]:
        """Load standardized test images."""
        # Implementation to load test dataset
        # Should include variety of scenarios: different lighting, angles, etc.
        pass
    
    def _load_ground_truth(self) -> Dict[str, List]:
        """Load ground truth annotations for accuracy testing."""
        # Implementation to load ground truth face annotations
        pass
    
    def _calculate_accuracy(self, detector) -> float:
        """Calculate detection accuracy against ground truth."""
        # Implementation for accuracy calculation
        # Using metrics like mAP, precision, recall
        pass


class PerformanceProfiler:
    """Real-time performance profiling for detectors."""
    
    def __init__(self, detector):
        """Initialize profiler for a detector."""
        self.detector = detector
        self.inference_times = []
        self.memory_snapshots = []
        self.cpu_usage = []
        self.gpu_usage = []
    
    def start_profiling(self):
        """Start performance profiling."""
        # Implementation for continuous performance monitoring
        pass
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        # Implementation for detailed performance analysis
        pass
```

## 🧪 **Testing Requirements**

### **Hardware-Specific Testing**
- GPU detector testing with different CUDA versions
- EdgeTPU testing on actual Coral devices
- Performance regression testing
- Memory leak detection
- Temperature monitoring validation

### **Compatibility Testing**
- Cross-platform compatibility (Linux, Windows, macOS)
- Different hardware configurations
- Driver version compatibility
- Model format validation

### **Performance Testing**
- Latency benchmarks across all detectors
- Throughput testing with concurrent requests
- Memory usage profiling
- Power consumption measurement (EdgeTPU)

## 📋 **Acceptance Criteria**

### **Functional Requirements**
- [ ] GPU detector achieves 5-10x speedup over CPU
- [ ] EdgeTPU detector provides sub-100ms inference
- [ ] Automatic hardware detection and optimal selection
- [ ] Seamless fallback mechanisms
- [ ] Comprehensive error handling and recovery

### **Performance Requirements**
- [ ] GPU: >30 FPS on RTX 3070 or equivalent
- [ ] EdgeTPU: <100ms inference time
- [ ] Memory efficiency: <4GB GPU memory usage
- [ ] Temperature management for EdgeTPU

### **Quality Requirements**
- [ ] 95% test coverage for new detectors
- [ ] Hardware compatibility documentation
- [ ] Performance benchmarking suite
- [ ] Setup and troubleshooting guides

---

**This issue completes the detector strategy framework with high-performance hardware-accelerated implementations, providing significant performance improvements for production deployments while maintaining the flexibility and fallback mechanisms of the existing architecture.**