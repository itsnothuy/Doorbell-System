#!/usr/bin/env python3
"""
Motion Detection Configuration

Configuration management for the motion detection worker with platform-specific
optimizations and sensible defaults.
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.platform_detector import platform_detector


@dataclass
class MotionConfig:
    """Configuration for motion detection worker."""
    
    # Base worker configuration
    enabled: bool = True
    
    # Motion detection parameters
    motion_threshold: float = 25.0
    min_contour_area: int = 500
    gaussian_blur_kernel: Tuple[int, int] = (21, 21)
    
    # Background subtraction
    bg_subtractor_type: str = "MOG2"  # MOG2, KNN
    bg_learning_rate: float = 0.01
    bg_history: int = 500
    bg_var_threshold: int = 50
    
    # Motion analysis
    motion_history_size: int = 10
    motion_smoothing_factor: float = 0.3
    min_motion_duration: float = 0.5  # seconds
    max_static_duration: float = 30.0  # seconds
    
    # Region of interest
    roi_enabled: bool = False
    roi_coordinates: Optional[Tuple[int, int, int, int]] = None
    
    # Performance optimization
    frame_resize_factor: float = 0.5
    skip_frame_count: int = 0  # Process every Nth frame (0 = process all)
    
    # Worker configuration
    worker_id: str = "motion_detector"
    queue_size: int = 100
    timeout: float = 10.0
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MotionConfig':
        """
        Create configuration from dictionary with validation.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            MotionConfig: Validated configuration instance
        """
        # Create config with defaults
        config = cls()
        
        # Update with provided values
        for key, value in config_dict.items():
            if hasattr(config, key):
                # Type validation for critical parameters
                if key == 'motion_threshold' and not isinstance(value, (int, float)):
                    raise ValueError(f"motion_threshold must be numeric, got {type(value)}")
                if key == 'min_contour_area' and not isinstance(value, int):
                    raise ValueError(f"min_contour_area must be int, got {type(value)}")
                if key == 'bg_subtractor_type' and value not in ['MOG2', 'KNN']:
                    raise ValueError(f"bg_subtractor_type must be MOG2 or KNN, got {value}")
                
                setattr(config, key, value)
        
        # Validate configuration
        config._validate()
        
        return config
    
    def _validate(self) -> None:
        """Validate configuration values."""
        # Threshold validation
        if self.motion_threshold < 0 or self.motion_threshold > 100:
            raise ValueError(f"motion_threshold must be between 0 and 100, got {self.motion_threshold}")
        
        # Contour area validation
        if self.min_contour_area < 0:
            raise ValueError(f"min_contour_area must be positive, got {self.min_contour_area}")
        
        # Gaussian blur kernel must be odd
        if self.gaussian_blur_kernel[0] % 2 == 0 or self.gaussian_blur_kernel[1] % 2 == 0:
            raise ValueError(f"gaussian_blur_kernel values must be odd, got {self.gaussian_blur_kernel}")
        
        # Learning rate validation
        if self.bg_learning_rate < 0 or self.bg_learning_rate > 1:
            raise ValueError(f"bg_learning_rate must be between 0 and 1, got {self.bg_learning_rate}")
        
        # Frame resize factor validation
        if self.frame_resize_factor <= 0 or self.frame_resize_factor > 1.0:
            raise ValueError(f"frame_resize_factor must be between 0 and 1, got {self.frame_resize_factor}")
        
        # ROI validation
        if self.roi_enabled and self.roi_coordinates is not None:
            if len(self.roi_coordinates) != 4:
                raise ValueError(f"roi_coordinates must have 4 values (x, y, w, h)")
            if any(v < 0 for v in self.roi_coordinates):
                raise ValueError(f"roi_coordinates values must be non-negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'enabled': self.enabled,
            'motion_threshold': self.motion_threshold,
            'min_contour_area': self.min_contour_area,
            'gaussian_blur_kernel': self.gaussian_blur_kernel,
            'bg_subtractor_type': self.bg_subtractor_type,
            'bg_learning_rate': self.bg_learning_rate,
            'bg_history': self.bg_history,
            'bg_var_threshold': self.bg_var_threshold,
            'motion_history_size': self.motion_history_size,
            'motion_smoothing_factor': self.motion_smoothing_factor,
            'min_motion_duration': self.min_motion_duration,
            'max_static_duration': self.max_static_duration,
            'roi_enabled': self.roi_enabled,
            'roi_coordinates': self.roi_coordinates,
            'frame_resize_factor': self.frame_resize_factor,
            'skip_frame_count': self.skip_frame_count,
            'worker_id': self.worker_id,
            'queue_size': self.queue_size,
            'timeout': self.timeout
        }
    
    def apply_platform_optimizations(self) -> None:
        """Apply platform-specific optimizations."""
        platform_info = platform_detector.get_platform_info()
        
        # Raspberry Pi optimizations
        if platform_info.get('is_raspberry_pi', False):
            # More aggressive frame resizing for limited CPU
            self.frame_resize_factor = 0.5
            
            # Use MOG2 (faster than KNN on CPU)
            self.bg_subtractor_type = "MOG2"
            
            # Reduce history for memory efficiency
            self.bg_history = 300
            
            # Process fewer frames
            self.skip_frame_count = 1  # Process every other frame
            
            # Lower thresholds for better performance
            self.min_contour_area = 400
        
        # Docker optimizations
        elif platform_info.get('in_docker', False):
            # Moderate settings for containerized environment
            self.frame_resize_factor = 0.6
            self.skip_frame_count = 0
        
        # Development/macOS optimizations
        elif platform_info.get('platform') == 'darwin':
            # Higher quality for development
            self.frame_resize_factor = 0.8
            self.skip_frame_count = 0
            
            # Use KNN for better quality
            self.bg_subtractor_type = "MOG2"  # MOG2 is more stable
    
    def optimize_for_low_power(self) -> None:
        """Optimize configuration for low-power devices."""
        self.frame_resize_factor = 0.4
        self.skip_frame_count = 2  # Process every 3rd frame
        self.bg_history = 200
        self.min_contour_area = 600
        self.motion_history_size = 5
    
    def optimize_for_performance(self) -> None:
        """Optimize configuration for high-performance systems."""
        self.frame_resize_factor = 0.8
        self.skip_frame_count = 0
        self.bg_history = 1000
        self.min_contour_area = 300
        self.motion_history_size = 20


def create_default_config() -> MotionConfig:
    """
    Create default motion detection configuration.
    
    Returns:
        MotionConfig: Default configuration instance
    """
    config = MotionConfig()
    config.apply_platform_optimizations()
    return config


def load_config_from_env() -> MotionConfig:
    """
    Load motion detection configuration from environment variables.
    
    Returns:
        MotionConfig: Configuration loaded from environment
    """
    config = create_default_config()
    
    # Override with environment variables if present
    if os.getenv('MOTION_DETECTION_ENABLED'):
        config.enabled = os.getenv('MOTION_DETECTION_ENABLED').lower() == 'true'
    
    if os.getenv('MOTION_THRESHOLD'):
        config.motion_threshold = float(os.getenv('MOTION_THRESHOLD'))
    
    if os.getenv('MIN_CONTOUR_AREA'):
        config.min_contour_area = int(os.getenv('MIN_CONTOUR_AREA'))
    
    if os.getenv('FRAME_RESIZE_FACTOR'):
        config.frame_resize_factor = float(os.getenv('FRAME_RESIZE_FACTOR'))
    
    if os.getenv('BG_SUBTRACTOR_TYPE'):
        config.bg_subtractor_type = os.getenv('BG_SUBTRACTOR_TYPE')
    
    return config


# Default configuration instance
DEFAULT_MOTION_CONFIG = create_default_config()
