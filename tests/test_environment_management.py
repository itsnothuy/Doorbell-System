#!/usr/bin/env python3
"""
Tests for Environment Management System

Tests environment-specific configuration management, environment detection,
and environment variable integration.
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.environment import (
    Environment,
    EnvironmentManager,
    load_config_for_environment,
    get_current_environment
)
from config.pipeline_config import PipelineConfig


class TestEnvironment:
    """Test suite for Environment enum."""
    
    def test_environment_values(self):
        """Test environment enum values."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.TESTING.value == "testing"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"
    
    def test_environment_from_string_valid(self):
        """Test creating environment from valid string."""
        env = Environment.from_string("development")
        assert env == Environment.DEVELOPMENT
        
        env = Environment.from_string("PRODUCTION")
        assert env == Environment.PRODUCTION
    
    def test_environment_from_string_invalid(self):
        """Test creating environment from invalid string."""
        try:
            Environment.from_string("invalid")
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "Invalid environment" in str(e)


class TestEnvironmentManager:
    """Test suite for EnvironmentManager."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Save original environment
        self.original_env = os.environ.copy()
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_initialization_default(self):
        """Test manager initialization with default environment."""
        # Clear environment variable
        if 'DOORBELL_ENV' in os.environ:
            del os.environ['DOORBELL_ENV']
        if 'ENV' in os.environ:
            del os.environ['ENV']
        
        manager = EnvironmentManager()
        
        assert manager is not None
        assert manager.current_env == Environment.DEVELOPMENT
    
    def test_initialization_from_doorbell_env(self):
        """Test initialization from DOORBELL_ENV variable."""
        os.environ['DOORBELL_ENV'] = 'production'
        
        manager = EnvironmentManager()
        
        assert manager.current_env == Environment.PRODUCTION
    
    def test_initialization_from_env(self):
        """Test initialization from ENV variable."""
        if 'DOORBELL_ENV' in os.environ:
            del os.environ['DOORBELL_ENV']
        os.environ['ENV'] = 'staging'
        
        manager = EnvironmentManager()
        
        assert manager.current_env == Environment.STAGING
    
    def test_initialization_from_ci(self):
        """Test initialization from CI variable."""
        if 'DOORBELL_ENV' in os.environ:
            del os.environ['DOORBELL_ENV']
        if 'ENV' in os.environ:
            del os.environ['ENV']
        os.environ['CI'] = 'true'
        
        manager = EnvironmentManager()
        
        assert manager.current_env == Environment.TESTING
    
    def test_initialization_from_railway(self):
        """Test initialization from Railway variable."""
        if 'DOORBELL_ENV' in os.environ:
            del os.environ['DOORBELL_ENV']
        if 'CI' in os.environ:
            del os.environ['CI']
        os.environ['RAILWAY_ENVIRONMENT'] = 'production'
        
        manager = EnvironmentManager()
        
        assert manager.current_env == Environment.PRODUCTION
    
    def test_load_environment_config(self):
        """Test loading environment-specific configuration."""
        os.environ['DOORBELL_ENV'] = 'development'
        
        manager = EnvironmentManager()
        config = manager.load_environment_config()
        
        assert isinstance(config, PipelineConfig)
        assert config is not None
    
    def test_get_environment(self):
        """Test getting current environment."""
        os.environ['DOORBELL_ENV'] = 'production'
        
        manager = EnvironmentManager()
        env = manager.get_environment()
        
        assert env == Environment.PRODUCTION
    
    def test_is_production(self):
        """Test checking if environment is production."""
        os.environ['DOORBELL_ENV'] = 'production'
        manager = EnvironmentManager()
        assert manager.is_production() is True
        
        os.environ['DOORBELL_ENV'] = 'development'
        manager = EnvironmentManager()
        assert manager.is_production() is False
    
    def test_is_development(self):
        """Test checking if environment is development."""
        os.environ['DOORBELL_ENV'] = 'development'
        manager = EnvironmentManager()
        assert manager.is_development() is True
        
        os.environ['DOORBELL_ENV'] = 'production'
        manager = EnvironmentManager()
        assert manager.is_development() is False
    
    def test_get_environment_info(self):
        """Test getting environment information."""
        os.environ['DOORBELL_ENV'] = 'production'
        
        manager = EnvironmentManager()
        info = manager.get_environment_info()
        
        assert 'environment' in info
        assert info['environment'] == 'production'
        assert 'is_production' in info
        assert info['is_production'] is True
        assert 'env_vars' in info
    
    def test_apply_environment_defaults_development(self):
        """Test applying development environment defaults."""
        os.environ['DOORBELL_ENV'] = 'development'
        
        manager = EnvironmentManager()
        config = manager.load_environment_config()
        
        # Development should have DEBUG logging
        assert config.storage.log_level == 'DEBUG'
        assert config.monitoring.enabled is True
    
    def test_apply_environment_defaults_testing(self):
        """Test applying testing environment defaults."""
        os.environ['DOORBELL_ENV'] = 'testing'
        
        manager = EnvironmentManager()
        config = manager.load_environment_config()
        
        # Testing should have minimal resources
        assert config.face_detection.worker_count == 1
        assert config.face_recognition.worker_count == 1
        assert config.motion_detection.enabled is False
        assert config.notifications.enabled is False
    
    def test_apply_environment_defaults_production(self):
        """Test applying production environment defaults."""
        os.environ['DOORBELL_ENV'] = 'production'
        
        manager = EnvironmentManager()
        config = manager.load_environment_config()
        
        # Production should have monitoring and backups
        assert config.monitoring.enabled is True
        assert config.storage.backup_enabled is True
        assert config.motion_detection.enabled is True
        
        # Worker counts should be reasonable
        assert config.face_detection.worker_count <= 4
        assert config.face_recognition.worker_count <= 3
    
    def test_apply_env_variables(self):
        """Test applying environment variable overrides."""
        os.environ['DOORBELL_ENV'] = 'development'
        os.environ['DOORBELL_WORKER_COUNT'] = '8'
        os.environ['DOORBELL_FACE_TOLERANCE'] = '0.5'
        
        manager = EnvironmentManager()
        config = manager.load_environment_config()
        
        # Environment variables should be applied
        assert config.face_detection.worker_count == 8
        assert config.face_recognition.tolerance == 0.5
    
    def test_apply_env_variables_boolean(self):
        """Test applying boolean environment variables."""
        os.environ['DOORBELL_ENV'] = 'development'
        # Note: Boolean conversion happens for specific known boolean fields
        
        manager = EnvironmentManager()
        config = manager.load_environment_config()
        
        # Should be able to load without errors
        assert config is not None
    
    def test_convert_env_value_int(self):
        """Test converting environment value to int."""
        manager = EnvironmentManager()
        
        value = manager._convert_env_value('42', int)
        assert value == 42
        assert isinstance(value, int)
    
    def test_convert_env_value_float(self):
        """Test converting environment value to float."""
        manager = EnvironmentManager()
        
        value = manager._convert_env_value('3.14', float)
        assert value == 3.14
        assert isinstance(value, float)
    
    def test_convert_env_value_bool_true(self):
        """Test converting environment value to bool (true)."""
        manager = EnvironmentManager()
        
        for true_value in ['true', 'True', '1', 'yes', 'YES', 'on']:
            value = manager._convert_env_value(true_value, bool)
            assert value is True
    
    def test_convert_env_value_bool_false(self):
        """Test converting environment value to bool (false)."""
        manager = EnvironmentManager()
        
        for false_value in ['false', 'False', '0', 'no', 'NO', 'off']:
            value = manager._convert_env_value(false_value, bool)
            assert value is False
    
    def test_convert_env_value_string(self):
        """Test converting environment value to string."""
        manager = EnvironmentManager()
        
        value = manager._convert_env_value('test_value', str)
        assert value == 'test_value'
        assert isinstance(value, str)
    
    def test_validate_environment_config_production_monitoring(self):
        """Test production environment requires monitoring."""
        os.environ['DOORBELL_ENV'] = 'production'
        
        manager = EnvironmentManager()
        config = manager.load_environment_config()
        
        # Production must have monitoring enabled
        assert config.monitoring.enabled is True
    
    def test_environment_detection_priority(self):
        """Test environment detection priority order."""
        # DOORBELL_ENV should take precedence
        os.environ['DOORBELL_ENV'] = 'production'
        os.environ['CI'] = 'true'
        
        manager = EnvironmentManager()
        
        # Should be production, not testing
        assert manager.current_env == Environment.PRODUCTION


class TestLoadConfigForEnvironment:
    """Test suite for load_config_for_environment function."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.original_env = os.environ.copy()
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_load_config_default_environment(self):
        """Test loading config for default environment."""
        config = load_config_for_environment()
        
        assert isinstance(config, PipelineConfig)
    
    def test_load_config_specific_environment(self):
        """Test loading config for specific environment."""
        config = load_config_for_environment(Environment.PRODUCTION)
        
        assert isinstance(config, PipelineConfig)
        # Should have production settings
        assert config.monitoring.enabled is True
    
    def test_load_config_restores_original_env(self):
        """Test that function restores original environment."""
        os.environ['DOORBELL_ENV'] = 'development'
        original_value = os.environ['DOORBELL_ENV']
        
        # Load config for different environment
        load_config_for_environment(Environment.PRODUCTION)
        
        # Original environment should be restored
        assert os.environ['DOORBELL_ENV'] == original_value


class TestGetCurrentEnvironment:
    """Test suite for get_current_environment function."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.original_env = os.environ.copy()
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_get_current_environment_default(self):
        """Test getting current environment (default)."""
        if 'DOORBELL_ENV' in os.environ:
            del os.environ['DOORBELL_ENV']
        if 'ENV' in os.environ:
            del os.environ['ENV']
        
        env = get_current_environment()
        
        assert env == Environment.DEVELOPMENT
    
    def test_get_current_environment_from_var(self):
        """Test getting current environment from variable."""
        os.environ['DOORBELL_ENV'] = 'production'
        
        env = get_current_environment()
        
        assert env == Environment.PRODUCTION


class TestEnvironmentIntegration:
    """Integration tests for environment management."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.original_env = os.environ.copy()
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_full_environment_workflow_development(self):
        """Test complete workflow for development environment."""
        os.environ['DOORBELL_ENV'] = 'development'
        
        # Create manager
        manager = EnvironmentManager()
        assert manager.is_development()
        
        # Load config
        config = manager.load_environment_config()
        
        # Verify development settings
        assert config.storage.log_level == 'DEBUG'
        assert config.monitoring.enabled is True
    
    def test_full_environment_workflow_production(self):
        """Test complete workflow for production environment."""
        os.environ['DOORBELL_ENV'] = 'production'
        
        # Create manager
        manager = EnvironmentManager()
        assert manager.is_production()
        
        # Load config
        config = manager.load_environment_config()
        
        # Verify production settings
        assert config.storage.log_level == 'INFO'
        assert config.monitoring.enabled is True
        assert config.storage.backup_enabled is True
        assert config.motion_detection.enabled is True
    
    def test_environment_with_overrides(self):
        """Test environment with multiple variable overrides."""
        os.environ['DOORBELL_ENV'] = 'production'
        os.environ['DOORBELL_WORKER_COUNT'] = '4'
        os.environ['DOORBELL_LOG_LEVEL'] = 'DEBUG'
        os.environ['DOORBELL_FACE_TOLERANCE'] = '0.7'
        
        manager = EnvironmentManager()
        config = manager.load_environment_config()
        
        # Check overrides are applied
        assert config.face_detection.worker_count == 4
        assert config.storage.log_level == 'DEBUG'
        assert config.face_recognition.tolerance == 0.7
        
        # Production defaults should still apply
        assert config.motion_detection.enabled is True


if __name__ == '__main__':
    # Simple test runner for manual execution
    test_classes = [
        TestEnvironment,
        TestEnvironmentManager,
        TestLoadConfigForEnvironment,
        TestGetCurrentEnvironment,
        TestEnvironmentIntegration
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
