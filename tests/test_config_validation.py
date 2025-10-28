#!/usr/bin/env python3
"""
Tests for Configuration Validation System

Tests the validation functionality including schema validation,
dependency checking, and error reporting.
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.validation import (
    ConfigValidator,
    ValidationError,
    ValidationResult,
    create_validator
)


class TestValidationError:
    """Test suite for ValidationError dataclass."""
    
    def test_validation_error_creation(self):
        """Test creating a validation error."""
        error = ValidationError(
            path="test.path",
            message="Test message",
            severity="error",
            suggestion="Test suggestion"
        )
        
        assert error.path == "test.path"
        assert error.message == "Test message"
        assert error.severity == "error"
        assert error.suggestion == "Test suggestion"
    
    def test_validation_error_string_representation(self):
        """Test string representation of validation error."""
        error = ValidationError(
            path="test.path",
            message="Test message",
            severity="warning",
            suggestion="Fix this"
        )
        
        error_str = str(error)
        assert "WARNING" in error_str
        assert "test.path" in error_str
        assert "Test message" in error_str
        assert "Fix this" in error_str
    
    def test_validation_error_without_suggestion(self):
        """Test validation error without suggestion."""
        error = ValidationError(
            path="test.path",
            message="Test message",
            severity="error"
        )
        
        assert error.suggestion is None
        error_str = str(error)
        assert "Suggestion" not in error_str


class TestValidationResult:
    """Test suite for ValidationResult dataclass."""
    
    def test_validation_result_valid(self):
        """Test valid validation result."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[]
        )
        
        assert result.is_valid is True
        assert result.has_errors is False
        assert result.has_warnings is False
    
    def test_validation_result_with_errors(self):
        """Test validation result with errors."""
        error = ValidationError("test", "error", "error")
        result = ValidationResult(
            is_valid=False,
            errors=[error],
            warnings=[]
        )
        
        assert result.is_valid is False
        assert result.has_errors is True
        assert result.has_warnings is False
    
    def test_validation_result_with_warnings(self):
        """Test validation result with warnings."""
        warning = ValidationError("test", "warning", "warning")
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[warning]
        )
        
        assert result.is_valid is True
        assert result.has_errors is False
        assert result.has_warnings is True
    
    def test_validation_result_string_representation(self):
        """Test string representation of validation result."""
        error = ValidationError("test.error", "Error message", "error")
        warning = ValidationError("test.warning", "Warning message", "warning")
        
        result = ValidationResult(
            is_valid=False,
            errors=[error],
            warnings=[warning]
        )
        
        result_str = str(result)
        assert "✗" in result_str
        assert "Error message" in result_str
        assert "Warning message" in result_str


class TestConfigValidator:
    """Test suite for ConfigValidator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.validator = ConfigValidator()
    
    def test_initialization(self):
        """Test validator initialization."""
        assert self.validator is not None
        assert isinstance(self.validator.dependency_rules, dict)
    
    def test_validate_empty_config(self):
        """Test validating empty configuration."""
        config: Dict[str, Any] = {}
        result = self.validator.validate_full_config(config)
        
        # Empty config should be valid (no required fields)
        assert isinstance(result, ValidationResult)
    
    def test_validate_basic_config(self):
        """Test validating basic valid configuration."""
        config = {
            'face_detection': {
                'enabled': True,
                'worker_count': 2,
                'detector_type': 'cpu'
            },
            'face_recognition': {
                'enabled': True,
                'worker_count': 2
            }
        }
        
        result = self.validator.validate_full_config(config)
        
        assert isinstance(result, ValidationResult)
        # Should be valid or have only warnings
        assert result.is_valid or not result.has_errors
    
    def test_validate_gpu_detector_without_cuda(self):
        """Test validation fails for GPU detector without CUDA."""
        config = {
            'face_detection': {
                'enabled': True,
                'detector_type': 'gpu'
            },
            'hardware': {
                'gpu_available': False
            }
        }
        
        result = self.validator.validate_full_config(config)
        
        # Should have error about GPU requirement
        assert result.has_errors
        gpu_errors = [e for e in result.errors if 'GPU' in e.message or 'CUDA' in e.message]
        assert len(gpu_errors) > 0
    
    def test_validate_edgetpu_detector_without_hardware(self):
        """Test validation fails for EdgeTPU detector without hardware."""
        config = {
            'face_detection': {
                'enabled': True,
                'detector_type': 'edgetpu'
            },
            'hardware': {
                'edgetpu_available': False
            }
        }
        
        result = self.validator.validate_full_config(config)
        
        # Should have error about EdgeTPU requirement
        assert result.has_errors
        edgetpu_errors = [e for e in result.errors if 'EdgeTPU' in e.message]
        assert len(edgetpu_errors) > 0
    
    def test_validate_high_worker_count_warning(self):
        """Test warning for high worker count."""
        config = {
            'face_detection': {
                'enabled': True,
                'worker_count': 10,  # High count
                'detector_type': 'cpu'
            }
        }
        
        result = self.validator.validate_full_config(config)
        
        # Should have warning about high worker count
        assert result.has_warnings
        worker_warnings = [w for w in result.warnings if 'worker' in w.message.lower()]
        assert len(worker_warnings) > 0
    
    def test_validate_high_fps_warning(self):
        """Test warning for high FPS setting."""
        config = {
            'frame_capture': {
                'fps': 60  # Very high FPS
            }
        }
        
        result = self.validator.validate_full_config(config)
        
        # Should have warning about high FPS
        assert result.has_warnings
        fps_warnings = [w for w in result.warnings if 'fps' in w.message.lower()]
        assert len(fps_warnings) > 0
    
    def test_validate_large_queue_size_warning(self):
        """Test warning for large queue size."""
        config = {
            'face_detection': {
                'enabled': True,
                'max_queue_size': 5000  # Very large
            }
        }
        
        result = self.validator.validate_full_config(config)
        
        # Should have warning about large queue
        assert result.has_warnings
        queue_warnings = [w for w in result.warnings if 'queue' in w.message.lower()]
        assert len(queue_warnings) > 0
    
    def test_validate_short_timeout_warning(self):
        """Test warning for short timeout."""
        config = {
            'face_recognition': {
                'enabled': True,
                'timeout': 2.0  # Very short
            }
        }
        
        result = self.validator.validate_full_config(config)
        
        # Should have warning about short timeout
        assert result.has_warnings
        timeout_warnings = [w for w in result.warnings if 'timeout' in w.message.lower()]
        assert len(timeout_warnings) > 0
    
    def test_validate_motion_detection_disabled_info(self):
        """Test info message when motion detection is disabled."""
        config = {
            'motion_detection': {
                'enabled': False
            },
            'face_detection': {
                'enabled': True
            }
        }
        
        result = self.validator.validate_full_config(config)
        
        # Should have info about motion detection
        motion_warnings = [
            w for w in result.warnings 
            if 'motion' in w.message.lower()
        ]
        assert len(motion_warnings) > 0
    
    def test_validate_large_cache_size_warning(self):
        """Test warning for large cache size."""
        config = {
            'face_recognition': {
                'enabled': True,
                'cache': {
                    'cache_size': 20000  # Very large
                }
            }
        }
        
        result = self.validator.validate_full_config(config)
        
        # Should have warning about cache size
        assert result.has_warnings
        cache_warnings = [w for w in result.warnings if 'cache' in w.message.lower()]
        assert len(cache_warnings) > 0
    
    def test_validate_high_total_workers_warning(self):
        """Test warning for high total worker count."""
        config = {
            'face_detection': {
                'enabled': True,
                'worker_count': 10
            },
            'face_recognition': {
                'enabled': True,
                'worker_count': 10
            }
        }
        
        result = self.validator.validate_full_config(config)
        
        # Should have warning about total workers
        assert result.has_warnings
        worker_warnings = [w for w in result.warnings if 'worker' in w.message.lower()]
        assert len(worker_warnings) > 0
    
    def test_validate_section(self):
        """Test validating a specific section."""
        section_config = {
            'enabled': True,
            'worker_count': 2
        }
        
        result = self.validator.validate_section('face_detection', section_config)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
    
    def test_validate_section_invalid_type(self):
        """Test validating section with invalid type."""
        section_config = "not a dict"  # Invalid type
        
        result = self.validator.validate_section('face_detection', section_config)
        
        assert not result.is_valid
        assert result.has_errors
    
    def test_dependency_rules_loaded(self):
        """Test that dependency rules are loaded."""
        assert len(self.validator.dependency_rules) > 0
        
        # Check for expected rules
        assert 'gpu_detector_requires_cuda' in self.validator.dependency_rules
        assert 'edgetpu_detector_requires_hardware' in self.validator.dependency_rules
    
    def test_schema_suggestion_for_type_error(self):
        """Test schema suggestion generation for type errors."""
        # Create a mock validation error
        mock_error = Mock()
        mock_error.validator = 'type'
        mock_error.validator_value = 'string'
        
        suggestion = self.validator._get_schema_suggestion(mock_error)
        
        assert suggestion is not None
        assert 'string' in suggestion
    
    def test_schema_suggestion_for_minimum_error(self):
        """Test schema suggestion generation for minimum errors."""
        mock_error = Mock()
        mock_error.validator = 'minimum'
        mock_error.validator_value = 1
        
        suggestion = self.validator._get_schema_suggestion(mock_error)
        
        assert suggestion is not None
        assert '1' in suggestion


class TestCreateValidator:
    """Test suite for create_validator factory function."""
    
    def test_create_validator_default(self):
        """Test creating validator with defaults."""
        validator = create_validator()
        
        assert isinstance(validator, ConfigValidator)
    
    def test_create_validator_custom_schema_path(self):
        """Test creating validator with custom schema path."""
        custom_path = Path("/tmp/schemas")
        validator = create_validator(schema_path=custom_path)
        
        assert isinstance(validator, ConfigValidator)
        assert validator.schema_path == custom_path


class TestValidationIntegration:
    """Integration tests for validation system."""
    
    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        validator = create_validator()
        
        # Create a realistic configuration
        config = {
            'frame_capture': {
                'enabled': True,
                'fps': 30,
                'resolution': (640, 480)
            },
            'motion_detection': {
                'enabled': True
            },
            'face_detection': {
                'enabled': True,
                'worker_count': 2,
                'detector_type': 'cpu',
                'max_queue_size': 100
            },
            'face_recognition': {
                'enabled': True,
                'worker_count': 2,
                'tolerance': 0.6,
                'timeout': 15.0
            },
            'storage': {
                'log_level': 'INFO'
            }
        }
        
        result = validator.validate_full_config(config)
        
        # Should succeed (may have warnings but no errors)
        assert not result.has_errors
        
        # Print result for inspection
        print(f"\nValidation result:\n{result}")
    
    def test_validation_with_multiple_errors(self):
        """Test validation with multiple configuration errors."""
        validator = create_validator()
        
        config = {
            'face_detection': {
                'enabled': True,
                'detector_type': 'gpu',  # Requires GPU
                'worker_count': 20  # Too many workers
            },
            'hardware': {
                'gpu_available': False  # GPU not available
            },
            'face_recognition': {
                'enabled': True,
                'timeout': 1.0  # Too short
            }
        }
        
        result = validator.validate_full_config(config)
        
        # Should have multiple issues
        assert result.has_errors or result.has_warnings
        total_issues = len(result.errors) + len(result.warnings)
        assert total_issues > 1


if __name__ == '__main__':
    # Simple test runner for manual execution
    test_classes = [
        TestValidationError,
        TestValidationResult,
        TestConfigValidator,
        TestCreateValidator,
        TestValidationIntegration
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
