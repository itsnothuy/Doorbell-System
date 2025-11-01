#!/usr/bin/env python3
"""
Pipeline Configuration - Frigate-Inspired Configuration Management

This module provides configuration management for the entire pipeline,
with sensible defaults and platform-specific optimizations.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.platform_detector import platform_detector

logger = logging.getLogger(__name__)


@dataclass
@dataclass
class FrameCaptureConfig:
    """Configuration for frame capture stage."""
    enabled: bool = True
    fps: int = 10
    resolution: tuple = (640, 480)
    ring_buffer_size: int = 30
    buffer_size: int = 30  # Alias for ring_buffer_size
    quality: int = 85
    format: str = "jpeg"
    retry_attempts: int = 3
    retry_delay: float = 0.5
    
    # Frame capture specific
    capture_fps: int = 30
    burst_count: int = 5
    burst_interval: float = 0.2
    
    # Platform-specific settings
    use_threading: bool = True
    buffer_timeout: float = 1.0


@dataclass
class MotionDetectionConfig:
    """Configuration for motion detection stage."""
    enabled: bool = False  # Optional optimization step
    sensitivity: float = 0.1
    min_area: int = 500
    max_area: int = 50000
    blur_kernel: int = 21
    threshold: int = 25
    
    # Advanced settings
    background_subtraction: bool = True
    morphology_enabled: bool = True
    contour_filtering: bool = True
    
    # New comprehensive settings (aligned with motion_config.py)
    motion_threshold: float = 25.0
    min_contour_area: int = 500
    gaussian_blur_kernel: tuple = (21, 21)
    bg_subtractor_type: str = "MOG2"  # MOG2, KNN
    bg_learning_rate: float = 0.01
    bg_history: int = 500
    bg_var_threshold: int = 50
    motion_history_size: int = 10
    motion_smoothing_factor: float = 0.3
    min_motion_duration: float = 0.5
    max_static_duration: float = 30.0
    roi_enabled: bool = False
    roi_coordinates: Optional[tuple] = None
    frame_resize_factor: float = 0.5
    skip_frame_count: int = 0
    worker_id: str = "motion_detector"
    queue_size: int = 100
    timeout: float = 10.0


@dataclass
@dataclass
class FaceDetectionConfig:
    """Configuration for face detection stage."""
    enabled: bool = True
    worker_count: int = 2
    detector_type: str = "cpu"  # cpu, gpu, edgetpu, mock
    model: str = "hog"  # hog or cnn for CPU detector
    model_type: str = "hog"  # Deprecated: use 'model' instead
    
    # Detection parameters
    number_of_times_to_upsample: int = 1
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_face_size: tuple = (30, 30)
    max_face_size: tuple = (1000, 1000)
    confidence_threshold: float = 0.5
    
    # Performance settings
    max_queue_size: int = 100
    batch_size: int = 1
    job_timeout: float = 30.0  # Job expiration timeout
    timeout: float = 10.0
    
    # Hardware acceleration
    gpu_memory_fraction: float = 0.3
    use_tensorrt: bool = False
    
    # Priority settings
    doorbell_priority: int = 1  # Higher priority for doorbell events
    continuous_priority: int = 2  # Lower priority for continuous monitoring


@dataclass
class FaceRecognitionConfig:
    """Configuration for face recognition stage."""
    enabled: bool = True
    worker_count: int = 2
    model_type: str = "dlib"  # dlib, facenet, arcface
    encoding_model: str = "small"  # small, large
    
    # Recognition parameters
    tolerance: float = 0.6
    blacklist_tolerance: float = 0.5  # Stricter for blacklist
    min_confidence: float = 0.4
    face_jitter: int = 1
    num_encodings: int = 1
    number_of_times_to_upsample: int = 1
    batch_size: int = 5
    
    # Database settings
    known_faces_path: str = "data/known_faces"
    blacklist_faces_path: str = "data/blacklist_faces"
    database_path: str = "data/faces.db"
    max_faces_per_person: int = 10
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    
    # Cache settings - use field with default_factory for complex types
    cache: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'cache_size': 1000,
        'ttl_seconds': 3600,
        'warm_up_enabled': True,
        'warm_up_limit': 100
    })
    cache_encodings: bool = True
    cache_file: str = "data/face_encodings_cache.pkl"
    
    # Performance settings
    max_queue_size: int = 50
    timeout: float = 15.0
    max_concurrent_recognitions: int = 10
    encoding_timeout_seconds: float = 5.0
    
    # Advanced settings
    similarity_metric: str = "euclidean"  # euclidean, cosine
    enable_anti_spoofing: bool = False


@dataclass
class EventProcessingConfig:
    """Configuration for event processing and enrichment."""
    enabled: bool = True
    max_queue_size: int = 1000
    batch_processing: bool = False
    batch_size: int = 10
    batch_timeout: float = 5.0
    
    # Enrichment settings
    enable_telegram: bool = True
    enable_web_events: bool = True
    enable_database: bool = True
    
    # Event lifecycle
    event_retention_days: int = 30
    cleanup_interval_hours: int = 24
    
    # Performance settings
    worker_threads: int = 4
    max_concurrent_enrichments: int = 10


@dataclass
class StorageConfig:
    """Configuration for data storage."""
    database_path: str = "data/events.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    
    # Image storage
    capture_path: str = "data/captures"
    keep_images_days: int = 7
    compress_images: bool = True
    
    # Logging
    log_path: str = "data/logs"
    log_level: str = "INFO"
    log_rotation_size: str = "10MB"
    log_retention_days: int = 30


@dataclass
class NotificationConfig:
    """Configuration for notifications."""
    # Internal notification system
    enabled: bool = True
    debounce_period: float = 5.0
    aggregation_window: float = 30.0
    max_per_hour: int = 100
    
    # Legacy telegram settings
    telegram_enabled: bool = True
    telegram_timeout: float = 30.0
    
    # Message formatting
    include_images: bool = True
    max_image_size: int = 2048  # pixels
    image_quality: int = 85
    
    # Rate limiting (legacy, kept for backward compatibility)
    rate_limit_window: int = 60  # seconds
    max_notifications_per_window: int = 10
    
    # Priority handling
    priority_bypass_rate_limit: bool = True
    emergency_contacts: List[str] = field(default_factory=list)
    
    # Internal notification system configuration
    alerts: Dict[str, Any] = field(default_factory=lambda: {
        'blacklist_detection': {'enabled': True, 'priority': 'critical'},
        'known_person_detection': {'enabled': True, 'priority': 'info'},
        'unknown_person_detection': {'enabled': True, 'priority': 'warning'},
        'system_alerts': {'enabled': True, 'priority': 'warning'}
    })
    
    rate_limiting: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'max_per_minute': 10,
        'max_per_hour': 100,
        'burst_allowance': 3,
        'cooldown_period': 60
    })
    
    delivery: Dict[str, Any] = field(default_factory=lambda: {
        'web_interface': {'enabled': True, 'endpoint': '/api/notifications', 'real_time': True},
        'file_log': {'enabled': True, 'path': 'data/logs/notifications.log', 'format': 'json'},
        'console': {'enabled': False}
    })
    
    storage: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'db_path': 'data/notifications.db',
        'retention_days': 30
    })


@dataclass
class HardwareConfig:
    """Configuration for hardware interfaces."""
    # GPIO settings
    doorbell_pin: int = 18
    led_pin: int = 16
    button_debounce_time: float = 5.0
    
    # Camera settings
    camera_type: str = "auto"  # auto, picamera, opencv, mock
    camera_index: int = 0
    camera_warmup_time: float = 2.0
    
    # Performance optimization
    gpio_polling_rate: float = 0.1
    camera_buffer_size: int = 1


@dataclass
class MonitoringConfig:
    """Configuration for system monitoring and health checks."""
    enabled: bool = True
    health_check_interval: float = 30.0
    metrics_collection_interval: float = 60.0
    
    # Performance monitoring
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    queue_threshold: int = 100
    
    # Alerting
    alert_on_high_cpu: bool = True
    alert_on_high_memory: bool = True
    alert_on_errors: bool = True
    error_threshold: int = 10  # errors per minute


class PipelineConfig:
    """Main pipeline configuration manager."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize pipeline configuration."""
        # Load default configuration
        self._load_defaults()
        
        # Apply platform-specific optimizations
        self._apply_platform_optimizations()
        
        # Override with provided configuration
        if config_dict:
            self._apply_config_dict(config_dict)
        
        # Load from environment variables
        self._load_from_environment()
        
        # Validate configuration
        self._validate_config()
    
    def _load_defaults(self) -> None:
        """Load default configuration values."""
        self.frame_capture = FrameCaptureConfig()
        self.motion_detection = MotionDetectionConfig()
        self.face_detection = FaceDetectionConfig()
        self.face_recognition = FaceRecognitionConfig()
        self.event_processing = EventProcessingConfig()
        self.storage = StorageConfig()
        self.notifications = NotificationConfig()
        self.hardware = HardwareConfig()
        self.monitoring = MonitoringConfig()
    
    def _apply_platform_optimizations(self) -> None:
        """Apply platform-specific optimizations."""
        # Raspberry Pi optimizations
        if platform_detector.is_raspberry_pi:
            # Reduce worker counts for limited CPU
            self.face_detection.worker_count = 1
            self.face_recognition.worker_count = 1
            
            # Use lighter models
            self.face_detection.model_type = "hog"
            self.face_recognition.encoding_model = "small"
            
            # Reduce queue sizes
            self.face_detection.max_queue_size = 50
            self.face_recognition.max_queue_size = 25
            
            # Enable motion detection for performance
            self.motion_detection.enabled = True
        
        # macOS/Development optimizations
        elif platform_detector.is_macos:
            # Use more workers for development
            self.face_detection.worker_count = 4
            self.face_recognition.worker_count = 3
            
            # Higher quality for testing
            self.frame_capture.quality = 95
            self.notifications.image_quality = 95
            
            # Enable web interface
            self.event_processing.enable_web_events = True
        
        # Docker optimizations
        if os.getenv('DOCKER_CONTAINER') or os.getenv('KUBERNETES_SERVICE_HOST'):
            # Adjust paths for container
            self.storage.database_path = "/app/data/events.db"
            self.storage.capture_path = "/app/data/captures"
            self.storage.log_path = "/app/data/logs"
            self.face_recognition.known_faces_path = "/app/data/known_faces"
            self.face_recognition.blacklist_faces_path = "/app/data/blacklist_faces"
    
    def _apply_config_dict(self, config_dict: Dict[str, Any]) -> None:
        """Apply configuration from dictionary."""
        # Check if config_dict is actually a dictionary
        if not isinstance(config_dict, dict):
            logger.warning(f"Expected dict for config_dict, got {type(config_dict)}. Skipping.")
            return
            
        for section_name, section_config in config_dict.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_config.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        # Face recognition settings
        if os.getenv('FACE_RECOGNITION_TOLERANCE'):
            self.face_recognition.tolerance = float(os.getenv('FACE_RECOGNITION_TOLERANCE'))
        
        # Hardware settings
        if os.getenv('DOORBELL_PIN'):
            self.hardware.doorbell_pin = int(os.getenv('DOORBELL_PIN'))
        
        # Storage paths
        if os.getenv('DATA_PATH'):
            data_path = os.getenv('DATA_PATH')
            self.storage.database_path = f"{data_path}/events.db"
            self.storage.capture_path = f"{data_path}/captures"
            self.storage.log_path = f"{data_path}/logs"
        
        # Performance tuning
        if os.getenv('WORKER_COUNT'):
            worker_count = int(os.getenv('WORKER_COUNT'))
            self.face_detection.worker_count = worker_count
            self.face_recognition.worker_count = max(1, worker_count // 2)
        
        # Debug mode
        if os.getenv('DEBUG') == 'true':
            self.storage.log_level = "DEBUG"
            self.monitoring.health_check_interval = 10.0
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Ensure worker counts are reasonable
        if self.face_detection.worker_count < 1:
            self.face_detection.worker_count = 1
        if self.face_recognition.worker_count < 1:
            self.face_recognition.worker_count = 1
        
        # Ensure timeouts are positive
        if self.face_detection.timeout <= 0:
            self.face_detection.timeout = 10.0
        if self.face_recognition.timeout <= 0:
            self.face_recognition.timeout = 15.0
        
        # Ensure paths exist
        Path(self.storage.capture_path).mkdir(parents=True, exist_ok=True)
        Path(self.storage.log_path).mkdir(parents=True, exist_ok=True)
        Path(self.face_recognition.known_faces_path).mkdir(parents=True, exist_ok=True)
        Path(self.face_recognition.blacklist_faces_path).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'frame_capture': self.frame_capture.__dict__,
            'motion_detection': self.motion_detection.__dict__,
            'face_detection': self.face_detection.__dict__,
            'face_recognition': self.face_recognition.__dict__,
            'event_processing': self.event_processing.__dict__,
            'storage': self.storage.__dict__,
            'notifications': self.notifications.__dict__,
            'hardware': self.hardware.__dict__,
            'monitoring': self.monitoring.__dict__
        }
    
    def get_performance_profile(self) -> str:
        """Get current performance profile."""
        total_workers = (self.face_detection.worker_count + 
                        self.face_recognition.worker_count)
        
        if total_workers <= 2:
            return "low_power"
        elif total_workers <= 4:
            return "balanced"
        elif total_workers <= 8:
            return "performance"
        else:
            return "high_performance"
    
    def optimize_for_hardware(self, cpu_cores: int, memory_gb: float) -> None:
        """Optimize configuration for specific hardware specs."""
        # Adjust worker counts based on CPU cores
        if cpu_cores <= 2:
            self.face_detection.worker_count = 1
            self.face_recognition.worker_count = 1
        elif cpu_cores <= 4:
            self.face_detection.worker_count = 2
            self.face_recognition.worker_count = 1
        else:
            self.face_detection.worker_count = min(4, cpu_cores // 2)
            self.face_recognition.worker_count = min(3, cpu_cores // 3)
        
        # Adjust queue sizes based on memory
        if memory_gb < 2.0:
            self.face_detection.max_queue_size = 25
            self.face_recognition.max_queue_size = 15
        elif memory_gb < 4.0:
            self.face_detection.max_queue_size = 50
            self.face_recognition.max_queue_size = 30
        # else use defaults
        
        # Enable motion detection for low-end hardware
        if cpu_cores <= 2 or memory_gb < 2.0:
            self.motion_detection.enabled = True


def load_config_from_file(config_path: str) -> PipelineConfig:
    """Load configuration from YAML or JSON file."""
    import json
    from pathlib import Path
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        if config_path.endswith('.json'):
            config_dict = json.load(f)
        elif config_path.endswith(('.yml', '.yaml')):
            import yaml
            config_dict = yaml.safe_load(f)
        else:
            raise ValueError("Configuration file must be JSON or YAML")
    
    return PipelineConfig(config_dict)


def create_minimal_config() -> PipelineConfig:
    """Create minimal configuration for testing."""
    config = PipelineConfig()
    
    # Minimal settings for fast startup
    config.face_detection.worker_count = 1
    config.face_recognition.worker_count = 1
    config.motion_detection.enabled = False
    config.monitoring.enabled = False
    
    return config


def create_production_config() -> PipelineConfig:
    """Create production-ready configuration."""
    config = PipelineConfig()
    
    # Production settings
    config.motion_detection.enabled = True
    config.monitoring.enabled = True
    config.storage.backup_enabled = True
    config.notifications.rate_limit_window = 300  # 5 minutes
    
    return config