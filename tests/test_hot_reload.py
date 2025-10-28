#!/usr/bin/env python3
"""
Tests for Configuration Hot-Reloading System

Tests the hot-reloading functionality including file watching,
configuration updates, and component notifications.
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.hot_reload import ConfigurationReloader, ConfigFileHandler, create_reloader
from config.pipeline_config import PipelineConfig
from config.validation import ConfigValidator


class TestConfigurationReloader:
    """Test suite for ConfigurationReloader."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.py"
        self.config_path.touch()
        
        # Create mock message bus
        self.mock_message_bus = Mock()
        self.mock_message_bus.publish = Mock()
        
        # Create mock validator
        self.mock_validator = Mock(spec=ConfigValidator)
        self.mock_validator.validate_full_config = Mock()
        
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test reloader initialization."""
        reloader = ConfigurationReloader(
            config_paths=[self.config_path],
            message_bus=self.mock_message_bus,
            validator=self.mock_validator
        )
        
        assert reloader.config_paths == [self.config_path]
        assert reloader.message_bus == self.mock_message_bus
        assert reloader.validator == self.mock_validator
        assert reloader.config_version == 1
        assert reloader.current_config is not None
        assert reloader.reload_count == 0
        assert reloader.failed_reloads == 0
    
    def test_initial_configuration_loaded(self):
        """Test that initial configuration is loaded on creation."""
        reloader = ConfigurationReloader(
            config_paths=[self.config_path],
            message_bus=self.mock_message_bus
        )
        
        assert reloader.current_config is not None
        assert isinstance(reloader.current_config, PipelineConfig)
        assert reloader.config_version == 1
        assert reloader.last_reload_time is not None
    
    def test_register_reload_callback(self):
        """Test registering reload callbacks."""
        reloader = ConfigurationReloader(
            config_paths=[self.config_path],
            message_bus=self.mock_message_bus
        )
        
        callback = Mock()
        reloader.register_reload_callback("test_component", callback)
        
        assert "test_component" in reloader.reload_callbacks
        assert reloader.reload_callbacks["test_component"] == callback
    
    def test_unregister_reload_callback(self):
        """Test unregistering reload callbacks."""
        reloader = ConfigurationReloader(
            config_paths=[self.config_path],
            message_bus=self.mock_message_bus
        )
        
        callback = Mock()
        reloader.register_reload_callback("test_component", callback)
        reloader.unregister_reload_callback("test_component")
        
        assert "test_component" not in reloader.reload_callbacks
    
    def test_reload_configuration_success(self):
        """Test successful configuration reload."""
        # Setup mock validator to return success
        mock_result = Mock()
        mock_result.is_valid = True
        mock_result.has_warnings = False
        mock_result.errors = []
        mock_result.warnings = []
        
        self.mock_validator.validate_full_config.return_value = mock_result
        
        reloader = ConfigurationReloader(
            config_paths=[self.config_path],
            message_bus=self.mock_message_bus,
            validator=self.mock_validator
        )
        
        initial_version = reloader.config_version
        
        # Force reload
        success = reloader.reload_configuration(force=True)
        
        assert success is True
        assert reloader.config_version == initial_version + 1
        assert reloader.reload_count == 1
        assert reloader.failed_reloads == 0
    
    def test_reload_configuration_validation_failure(self):
        """Test configuration reload with validation failure."""
        # Setup mock validator to return failure
        mock_result = Mock()
        mock_result.is_valid = False
        mock_result.errors = ["Test error"]
        
        self.mock_validator.validate_full_config.return_value = mock_result
        
        reloader = ConfigurationReloader(
            config_paths=[self.config_path],
            message_bus=self.mock_message_bus,
            validator=self.mock_validator
        )
        
        initial_version = reloader.config_version
        
        # Attempt reload
        success = reloader.reload_configuration(force=True)
        
        assert success is False
        assert reloader.config_version == initial_version  # Version unchanged
        assert reloader.failed_reloads == 1
    
    def test_reload_configuration_with_warnings(self):
        """Test configuration reload with validation warnings."""
        # Setup mock validator with warnings
        mock_warning = Mock()
        mock_warning.message = "Test warning"
        
        mock_result = Mock()
        mock_result.is_valid = True
        mock_result.has_warnings = True
        mock_result.errors = []
        mock_result.warnings = [mock_warning]
        
        self.mock_validator.validate_full_config.return_value = mock_result
        
        reloader = ConfigurationReloader(
            config_paths=[self.config_path],
            message_bus=self.mock_message_bus,
            validator=self.mock_validator
        )
        
        # Should succeed despite warnings
        success = reloader.reload_configuration(force=True)
        
        assert success is True
        assert reloader.reload_count == 1
    
    def test_reload_notification_to_components(self):
        """Test that components are notified of configuration changes."""
        mock_result = Mock()
        mock_result.is_valid = True
        mock_result.has_warnings = False
        mock_result.errors = []
        mock_result.warnings = []
        
        self.mock_validator.validate_full_config.return_value = mock_result
        
        reloader = ConfigurationReloader(
            config_paths=[self.config_path],
            message_bus=self.mock_message_bus,
            validator=self.mock_validator
        )
        
        # Register callbacks
        callback1 = Mock()
        callback2 = Mock()
        reloader.register_reload_callback("component1", callback1)
        reloader.register_reload_callback("component2", callback2)
        
        # Force reload
        reloader.reload_configuration(force=True)
        
        # Both callbacks should be called
        assert callback1.called
        assert callback2.called
    
    def test_reload_broadcasts_event(self):
        """Test that reload broadcasts event via message bus."""
        mock_result = Mock()
        mock_result.is_valid = True
        mock_result.has_warnings = False
        mock_result.errors = []
        mock_result.warnings = []
        
        self.mock_validator.validate_full_config.return_value = mock_result
        
        reloader = ConfigurationReloader(
            config_paths=[self.config_path],
            message_bus=self.mock_message_bus,
            validator=self.mock_validator
        )
        
        # Force reload
        reloader.reload_configuration(force=True)
        
        # Message bus should be called
        assert self.mock_message_bus.publish.called
        call_args = self.mock_message_bus.publish.call_args
        assert call_args[0][0] == 'config_events'
    
    def test_configs_equal_identical(self):
        """Test configuration equality check with identical configs."""
        reloader = ConfigurationReloader(
            config_paths=[self.config_path],
            message_bus=self.mock_message_bus
        )
        
        config1 = {'section': {'key': 'value'}}
        config2 = {'section': {'key': 'value'}}
        
        assert reloader._configs_equal(config1, config2) is True
    
    def test_configs_equal_different(self):
        """Test configuration equality check with different configs."""
        reloader = ConfigurationReloader(
            config_paths=[self.config_path],
            message_bus=self.mock_message_bus
        )
        
        config1 = {'section': {'key': 'value1'}}
        config2 = {'section': {'key': 'value2'}}
        
        assert reloader._configs_equal(config1, config2) is False
    
    def test_create_config_diff(self):
        """Test configuration diff creation."""
        reloader = ConfigurationReloader(
            config_paths=[self.config_path],
            message_bus=self.mock_message_bus
        )
        
        old_config = {
            'section1': {'key1': 'value1'},
            'section2': {'key2': 'value2'},
            'section3': {'key3': 'value3'}
        }
        
        new_config = {
            'section1': {'key1': 'value1'},  # Unchanged
            'section2': {'key2': 'new_value'},  # Modified
            'section4': {'key4': 'value4'}  # Added
            # section3 removed
        }
        
        diff = reloader._create_config_diff(old_config, new_config)
        
        assert 'section1' in diff['unchanged']
        assert 'section2' in diff['modified']
        assert 'section3' in diff['removed']
        assert 'section4' in diff['added']
    
    def test_get_statistics(self):
        """Test getting reload statistics."""
        reloader = ConfigurationReloader(
            config_paths=[self.config_path],
            message_bus=self.mock_message_bus
        )
        
        stats = reloader.get_statistics()
        
        assert 'config_version' in stats
        assert 'reload_count' in stats
        assert 'failed_reloads' in stats
        assert 'last_reload_time' in stats
        assert 'watching' in stats
        assert 'registered_callbacks' in stats
        assert 'success_rate' in stats
    
    def test_context_manager(self):
        """Test reloader as context manager."""
        with patch.object(ConfigurationReloader, 'start_watching') as mock_start, \
             patch.object(ConfigurationReloader, 'stop_watching') as mock_stop:
            
            with create_reloader(self.config_path, self.mock_message_bus) as reloader:
                assert isinstance(reloader, ConfigurationReloader)
            
            # Stop should be called on exit
            assert mock_stop.called
    
    def test_get_config(self):
        """Test getting current configuration."""
        reloader = ConfigurationReloader(
            config_paths=[self.config_path],
            message_bus=self.mock_message_bus
        )
        
        config = reloader.get_config()
        
        assert config is not None
        assert isinstance(config, PipelineConfig)
        assert config == reloader.current_config


class TestConfigFileHandler:
    """Test suite for ConfigFileHandler."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_reloader = Mock(spec=ConfigurationReloader)
        self.mock_reloader.config_paths = []
        self.mock_reloader.reload_configuration = Mock()
        
        self.handler = ConfigFileHandler(self.mock_reloader)
    
    def test_initialization(self):
        """Test handler initialization."""
        assert self.handler.reloader == self.mock_reloader
        assert self.handler.debounce_delay == 1.0
        assert self.handler.debounce_timer is None
    
    def test_on_modified_directory_ignored(self):
        """Test that directory modifications are ignored."""
        mock_event = Mock()
        mock_event.is_directory = True
        
        self.handler.on_modified(mock_event)
        
        # Reload should not be called for directories
        assert not self.mock_reloader.reload_configuration.called
    
    def test_debounced_reload_creates_timer(self):
        """Test that debounced reload creates a timer."""
        self.handler._debounced_reload()
        
        assert self.handler.debounce_timer is not None
    
    def test_debounced_reload_cancels_previous_timer(self):
        """Test that new reload cancels previous timer."""
        # Create first timer
        self.handler._debounced_reload()
        first_timer = self.handler.debounce_timer
        
        # Create second timer
        self.handler._debounced_reload()
        second_timer = self.handler.debounce_timer
        
        assert first_timer != second_timer


class TestCreateReloader:
    """Test suite for create_reloader factory function."""
    
    def test_create_reloader_default_path(self):
        """Test creating reloader with default path."""
        reloader = create_reloader()
        
        assert isinstance(reloader, ConfigurationReloader)
        assert len(reloader.config_paths) == 1
    
    def test_create_reloader_custom_path(self):
        """Test creating reloader with custom path."""
        custom_path = Path("/tmp/custom_config.py")
        reloader = create_reloader(config_path=custom_path)
        
        assert isinstance(reloader, ConfigurationReloader)
        assert custom_path in reloader.config_paths
    
    def test_create_reloader_with_message_bus(self):
        """Test creating reloader with message bus."""
        mock_bus = Mock()
        reloader = create_reloader(message_bus=mock_bus)
        
        assert reloader.message_bus == mock_bus


if __name__ == '__main__':
    # Simple test runner for manual execution
    import inspect
    
    test_classes = [
        TestConfigurationReloader,
        TestConfigFileHandler,
        TestCreateReloader
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
                failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print('=' * 60)
