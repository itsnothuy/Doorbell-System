#!/usr/bin/env python3
"""
Tests for Pipeline Configuration System

Tests the basic configuration loading, validation, and serialization.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.pipeline_config import (
    PipelineConfig,
    FrameCaptureConfig,
    MotionDetectionConfig,
    FaceDetectionConfig,
    FaceRecognitionConfig,
    EventProcessingConfig,
    StorageConfig,
    NotificationConfig,
    HardwareConfig,
    MonitoringConfig,
    load_config_from_file,
    create_minimal_config,
    create_production_config
)


class TestDataclasses:
    """Test configuration dataclasses."""
    
    def test_frame_capture_config_defaults(self):
        """Test FrameCaptureConfig default values."""
        config = FrameCaptureConfig()
        
        assert config.enabled is True
        assert config.fps == 10
        assert config.resolution == (640, 480)
        assert config.quality == 85
    
    def test_motion_detection_config_defaults(self):
        """Test MotionDetectionConfig default values."""
        config = MotionDetectionConfig()
        
        assert config.enabled is False
        assert config.sensitivity == 0.1
        assert config.min_area == 500
    
    def test_face_detection_config_defaults(self):
        """Test FaceDetectionConfig default values."""
        config = FaceDetectionConfig()
        
        assert config.enabled is True
        assert config.worker_count == 2
        assert config.detector_type == "cpu"
        assert config.model == "hog"
    
    def test_face_recognition_config_defaults(self):
        """Test FaceRecognitionConfig default values."""
        config = FaceRecognitionConfig()
        
        assert config.enabled is True
        assert config.worker_count == 2
        assert config.tolerance == 0.6
        assert config.blacklist_tolerance == 0.5
    
    def test_hardware_config_defaults(self):
        """Test HardwareConfig default values."""
        config = HardwareConfig()
        
        assert config.doorbell_pin == 18
        assert config.camera_type == "auto"


class TestPipelineConfig:
    """Test PipelineConfig class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        config = PipelineConfig()
        
        assert config is not None
        assert isinstance(config.frame_capture, FrameCaptureConfig)
        assert isinstance(config.motion_detection, MotionDetectionConfig)
        assert isinstance(config.face_detection, FaceDetectionConfig)
        assert isinstance(config.face_recognition, FaceRecognitionConfig)
    
    def test_initialization_with_dict(self):
        """Test initialization with configuration dictionary."""
        config_dict = {
            'face_detection': {
                'worker_count': 4,
                'detector_type': 'gpu'
            },
            'face_recognition': {
                'tolerance': 0.5
            }
        }
        
        config = PipelineConfig(config_dict)
        
        assert config.face_detection.worker_count == 4
        assert config.face_detection.detector_type == 'gpu'
        assert config.face_recognition.tolerance == 0.5
    
    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = PipelineConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'frame_capture' in config_dict
        assert 'face_detection' in config_dict
        assert 'face_recognition' in config_dict
        
        # Check nested structure
        assert isinstance(config_dict['frame_capture'], dict)
        assert 'fps' in config_dict['frame_capture']
    
    def test_get_performance_profile_low_power(self):
        """Test performance profile detection - low power."""
        config = PipelineConfig()
        config.face_detection.worker_count = 1
        config.face_recognition.worker_count = 1
        
        profile = config.get_performance_profile()
        
        assert profile == "low_power"
    
    def test_get_performance_profile_balanced(self):
        """Test performance profile detection - balanced."""
        config = PipelineConfig()
        config.face_detection.worker_count = 2
        config.face_recognition.worker_count = 2
        
        profile = config.get_performance_profile()
        
        assert profile == "balanced"
    
    def test_get_performance_profile_high_performance(self):
        """Test performance profile detection - high performance."""
        config = PipelineConfig()
        config.face_detection.worker_count = 6
        config.face_recognition.worker_count = 4
        
        profile = config.get_performance_profile()
        
        assert profile == "high_performance"
    
    def test_optimize_for_hardware_low_end(self):
        """Test hardware optimization for low-end system."""
        config = PipelineConfig()
        config.optimize_for_hardware(cpu_cores=2, memory_gb=1.5)
        
        assert config.face_detection.worker_count == 1
        assert config.face_recognition.worker_count == 1
        assert config.motion_detection.enabled is True
        assert config.face_detection.max_queue_size == 25
    
    def test_optimize_for_hardware_mid_range(self):
        """Test hardware optimization for mid-range system."""
        config = PipelineConfig()
        config.optimize_for_hardware(cpu_cores=4, memory_gb=3.0)
        
        assert config.face_detection.worker_count == 2
        assert config.face_recognition.worker_count == 1
        assert config.face_detection.max_queue_size == 50
    
    def test_optimize_for_hardware_high_end(self):
        """Test hardware optimization for high-end system."""
        config = PipelineConfig()
        config.optimize_for_hardware(cpu_cores=8, memory_gb=8.0)
        
        assert config.face_detection.worker_count == 4
        assert config.face_recognition.worker_count >= 2
    
    def test_load_from_environment_worker_count(self):
        """Test loading worker count from environment."""
        os.environ['WORKER_COUNT'] = '5'
        
        config = PipelineConfig()
        
        assert config.face_detection.worker_count == 5
        
        # Cleanup
        del os.environ['WORKER_COUNT']
    
    def test_load_from_environment_debug_mode(self):
        """Test loading debug mode from environment."""
        os.environ['DEBUG'] = 'true'
        
        config = PipelineConfig()
        
        assert config.storage.log_level == 'DEBUG'
        
        # Cleanup
        del os.environ['DEBUG']
    
    def test_validation_ensures_positive_worker_count(self):
        """Test that validation ensures positive worker counts."""
        config_dict = {
            'face_detection': {'worker_count': 0}
        }
        
        config = PipelineConfig(config_dict)
        
        # Should be corrected to minimum of 1
        assert config.face_detection.worker_count >= 1
    
    def test_validation_ensures_positive_timeouts(self):
        """Test that validation ensures positive timeouts."""
        config_dict = {
            'face_detection': {'timeout': -5.0}
        }
        
        config = PipelineConfig(config_dict)
        
        # Should be corrected to positive value
        assert config.face_detection.timeout > 0
    
    def test_validation_creates_paths(self):
        """Test that validation creates necessary paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dict = {
                'storage': {
                    'capture_path': f'{temp_dir}/captures',
                    'log_path': f'{temp_dir}/logs'
                },
                'face_recognition': {
                    'known_faces_path': f'{temp_dir}/known_faces'
                }
            }
            
            config = PipelineConfig(config_dict)
            
            # Paths should be created
            assert Path(config.storage.capture_path).exists()
            assert Path(config.storage.log_path).exists()
            assert Path(config.face_recognition.known_faces_path).exists()


class TestConfigurationFiles:
    """Test configuration file loading."""
    
    def test_load_config_from_json_file(self):
        """Test loading configuration from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('''{
                "face_detection": {
                    "worker_count": 3,
                    "detector_type": "cpu"
                }
            }''')
            json_path = f.name
        
        try:
            config = load_config_from_file(json_path)
            
            assert isinstance(config, PipelineConfig)
            assert config.face_detection.worker_count == 3
            assert config.face_detection.detector_type == "cpu"
        finally:
            Path(json_path).unlink()
    
    def test_load_config_from_nonexistent_file(self):
        """Test loading from non-existent file raises error."""
        try:
            load_config_from_file('/nonexistent/path/config.json')
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError:
            pass


class TestConfigFactories:
    """Test configuration factory functions."""
    
    def test_create_minimal_config(self):
        """Test creating minimal configuration."""
        config = create_minimal_config()
        
        assert isinstance(config, PipelineConfig)
        assert config.face_detection.worker_count == 1
        assert config.face_recognition.worker_count == 1
        assert config.motion_detection.enabled is False
        assert config.monitoring.enabled is False
    
    def test_create_production_config(self):
        """Test creating production configuration."""
        config = create_production_config()
        
        assert isinstance(config, PipelineConfig)
        assert config.motion_detection.enabled is True
        assert config.monitoring.enabled is True
        assert config.storage.backup_enabled is True


class TestConfigurationIntegration:
    """Integration tests for configuration system."""
    
    def test_full_configuration_workflow(self):
        """Test complete configuration workflow."""
        # Create configuration
        config = PipelineConfig()
        
        # Modify settings
        config.face_detection.worker_count = 3
        config.face_recognition.tolerance = 0.7
        
        # Convert to dict
        config_dict = config.to_dict()
        
        # Create new configuration from dict
        new_config = PipelineConfig(config_dict)
        
        # Verify settings preserved
        assert new_config.face_detection.worker_count == 3
        assert new_config.face_recognition.tolerance == 0.7
    
    def test_configuration_serialization_roundtrip(self):
        """Test configuration can be serialized and deserialized."""
        import json
        
        # Create config
        config = PipelineConfig()
        config.face_detection.worker_count = 5
        
        # Serialize to JSON
        config_dict = config.to_dict()
        json_str = json.dumps(config_dict)
        
        # Deserialize
        restored_dict = json.loads(json_str)
        restored_config = PipelineConfig(restored_dict)
        
        # Verify
        assert restored_config.face_detection.worker_count == 5


if __name__ == '__main__':
    # Simple test runner for manual execution
    test_classes = [
        TestDataclasses,
        TestPipelineConfig,
        TestConfigurationFiles,
        TestConfigFactories,
        TestConfigurationIntegration
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        print(f"\n{'=' * 60}")
        print(f"Running {test_class.__name__}")
        print('=' * 60)
        
        instance = test_class()
        
        # Get all test methods
        test_methods = [
            method for method in dir(instance)
            if method.startswith('test_') and callable(getattr(instance, method))
        ]
        
        for method_name in test_methods:
            try:
                # Setup
                if hasattr(instance, 'setup_method'):
                    instance.setup_method()
                
                # Run test
                method = getattr(instance, method_name)
                method()
                
                print(f"✓ {method_name}")
                passed += 1
                
                # Teardown
                if hasattr(instance, 'teardown_method'):
                    instance.teardown_method()
                    
            except Exception as e:
                print(f"✗ {method_name}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print('=' * 60)
