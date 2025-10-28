#!/usr/bin/env python3
"""
Integration Tests for Configuration Management System

Tests the integration of hot-reloading, validation, and environment management.
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.hot_reload import ConfigurationReloader, create_reloader
from config.validation import ConfigValidator, ValidationResult
from config.environment import EnvironmentManager, Environment, load_config_for_environment
from config.pipeline_config import PipelineConfig


class TestHotReloadWithValidation:
    """Test hot-reloading with validation integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.py"
        self.config_path.touch()
        
        self.validator = ConfigValidator()
        self.mock_message_bus = Mock()
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_reload_with_validation_success(self):
        """Test reload with successful validation."""
        reloader = ConfigurationReloader(
            config_paths=[self.config_path],
            message_bus=self.mock_message_bus,
            validator=self.validator
        )
        
        # Force reload
        success = reloader.reload_configuration(force=True)
        
        # Should succeed with valid config
        assert success is True
        assert reloader.failed_reloads == 0
    
    def test_reload_with_validation_failure(self):
        """Test reload with validation failure."""
        # Create mock validator that fails
        mock_validator = Mock()
        mock_result = Mock()
        mock_result.is_valid = False
        mock_result.errors = ["Test error"]
        mock_validator.validate_full_config.return_value = mock_result
        
        reloader = ConfigurationReloader(
            config_paths=[self.config_path],
            message_bus=self.mock_message_bus,
            validator=mock_validator
        )
        
        # Attempt reload
        success = reloader.reload_configuration(force=True)
        
        # Should fail
        assert success is False
        assert reloader.failed_reloads > 0


class TestEnvironmentWithValidation:
    """Test environment management with validation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.original_env = os.environ.copy()
        self.validator = ConfigValidator()
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_production_config_validation(self):
        """Test production configuration passes validation."""
        os.environ['DOORBELL_ENV'] = 'production'
        
        manager = EnvironmentManager()
        config = manager.load_environment_config()
        
        # Validate production config
        config_dict = config.to_dict()
        result = self.validator.validate_full_config(config_dict)
        
        # Production config should be valid
        assert not result.has_errors
    
    def test_testing_config_validation(self):
        """Test testing configuration passes validation."""
        os.environ['DOORBELL_ENV'] = 'testing'
        
        manager = EnvironmentManager()
        config = manager.load_environment_config()
        
        # Validate testing config
        config_dict = config.to_dict()
        result = self.validator.validate_full_config(config_dict)
        
        # Testing config should be valid
        assert not result.has_errors
    
    def test_development_config_validation(self):
        """Test development configuration passes validation."""
        os.environ['DOORBELL_ENV'] = 'development'
        
        manager = EnvironmentManager()
        config = manager.load_environment_config()
        
        # Validate development config
        config_dict = config.to_dict()
        result = self.validator.validate_full_config(config_dict)
        
        # Development config should be valid
        assert not result.has_errors


class TestFullConfigurationWorkflow:
    """Test complete configuration management workflow."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "config.py"
        self.config_path.touch()
        self.original_env = os.environ.copy()
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_complete_workflow(self):
        """Test complete configuration workflow."""
        # 1. Set environment
        os.environ['DOORBELL_ENV'] = 'production'
        
        # 2. Load environment-specific config
        env_manager = EnvironmentManager()
        config = env_manager.load_environment_config()
        
        assert config.monitoring.enabled is True
        assert config.storage.backup_enabled is True
        
        # 3. Validate configuration
        validator = ConfigValidator()
        config_dict = config.to_dict()
        result = validator.validate_full_config(config_dict)
        
        assert not result.has_errors
        
        # 4. Setup hot-reloading
        mock_bus = Mock()
        reloader = ConfigurationReloader(
            config_paths=[self.config_path],
            message_bus=mock_bus,
            validator=validator
        )
        
        # 5. Register component callback
        callback_called = []
        
        def component_callback(diff):
            callback_called.append(diff)
        
        reloader.register_reload_callback("test_component", component_callback)
        
        # 6. Trigger reload
        success = reloader.reload_configuration(force=True)
        
        assert success is True
        assert len(callback_called) > 0


class TestConfigurationPersistence:
    """Test configuration persistence and recovery."""
    
    def test_config_to_dict_and_back(self):
        """Test configuration can be converted to dict and restored."""
        # Create config with specific settings
        config1 = PipelineConfig()
        config1.face_detection.worker_count = 5
        config1.face_recognition.tolerance = 0.8
        config1.motion_detection.enabled = True
        
        # Convert to dict
        config_dict = config1.to_dict()
        
        # Create new config from dict
        config2 = PipelineConfig(config_dict)
        
        # Verify all settings match
        assert config2.face_detection.worker_count == 5
        assert config2.face_recognition.tolerance == 0.8
        assert config2.motion_detection.enabled is True
    
    def test_config_serialization_with_validation(self):
        """Test configuration serialization maintains validity."""
        validator = ConfigValidator()
        
        # Create and validate initial config
        config1 = PipelineConfig()
        result1 = validator.validate_full_config(config1.to_dict())
        
        # Serialize and deserialize
        config_dict = config1.to_dict()
        config2 = PipelineConfig(config_dict)
        result2 = validator.validate_full_config(config2.to_dict())
        
        # Both should be valid
        assert not result1.has_errors
        assert not result2.has_errors


class TestErrorRecovery:
    """Test error recovery and rollback mechanisms."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "config.py"
        self.config_path.touch()
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_rollback_on_invalid_config(self):
        """Test configuration rollback on validation failure."""
        # Create a validator that fails validation
        mock_validator = Mock()
        mock_result = Mock()
        mock_result.is_valid = False
        mock_result.errors = ["Critical validation error"]
        mock_result.has_warnings = False
        mock_validator.validate_full_config = Mock(return_value=mock_result)
        
        reloader = ConfigurationReloader(
            config_paths=[self.config_path],
            validator=mock_validator
        )
        
        # Get initial version
        initial_version = reloader.config_version
        
        # Attempt reload (will fail validation)
        success = reloader.reload_configuration(force=True)
        
        # Should fail and not increment version
        assert success is False
        assert reloader.config_version == initial_version
        assert reloader.failed_reloads > 0


class TestPerformanceAndScaling:
    """Test configuration system performance and scaling."""
    
    def test_reload_performance(self):
        """Test configuration reload performance."""
        temp_dir = tempfile.mkdtemp()
        config_path = Path(temp_dir) / "config.py"
        config_path.touch()
        
        try:
            reloader = ConfigurationReloader(
                config_paths=[config_path],
                validator=ConfigValidator()
            )
            
            # Time multiple reloads
            start_time = time.time()
            
            for _ in range(10):
                reloader.reload_configuration(force=True)
            
            elapsed = time.time() - start_time
            avg_reload_time = elapsed / 10
            
            # Average reload should be under 1 second
            assert avg_reload_time < 1.0
            
            # Check statistics
            stats = reloader.get_statistics()
            assert stats['reload_count'] == 10
            assert stats['success_rate'] == 1.0
            
        finally:
            import shutil
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
    
    def test_validation_performance(self):
        """Test validation performance."""
        validator = ConfigValidator()
        config = PipelineConfig()
        config_dict = config.to_dict()
        
        # Time multiple validations
        start_time = time.time()
        
        for _ in range(100):
            validator.validate_full_config(config_dict)
        
        elapsed = time.time() - start_time
        avg_validation_time = elapsed / 100
        
        # Average validation should be under 0.1 seconds
        assert avg_validation_time < 0.1


class TestConcurrency:
    """Test concurrent access to configuration."""
    
    def test_thread_safe_reload(self):
        """Test thread-safe configuration reload."""
        import threading
        
        temp_dir = tempfile.mkdtemp()
        config_path = Path(temp_dir) / "config.py"
        config_path.touch()
        
        try:
            reloader = ConfigurationReloader(
                config_paths=[config_path],
                validator=ConfigValidator()
            )
            
            errors = []
            
            def reload_worker():
                try:
                    for _ in range(5):
                        reloader.reload_configuration(force=True)
                except Exception as e:
                    errors.append(e)
            
            # Start multiple threads
            threads = [threading.Thread(target=reload_worker) for _ in range(3)]
            for thread in threads:
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # No errors should occur
            assert len(errors) == 0
            
            # All reloads should be tracked
            stats = reloader.get_statistics()
            assert stats['reload_count'] >= 15  # 3 threads * 5 reloads
            
        finally:
            import shutil
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)


if __name__ == '__main__':
    # Simple test runner for manual execution
    test_classes = [
        TestHotReloadWithValidation,
        TestEnvironmentWithValidation,
        TestFullConfigurationWorkflow,
        TestConfigurationPersistence,
        TestErrorRecovery,
        TestPerformanceAndScaling,
        TestConcurrency
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
