#!/usr/bin/env python3
"""
Legacy Compatibility Tests

Tests to ensure backward compatibility with existing code and interfaces.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.integration.legacy_adapter import LegacyAdapter
from src.integration.orchestrator_manager import OrchestratorManager


class TestLegacyAPICompatibility:
    """Test that legacy API remains compatible."""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        orchestrator = MagicMock()
        orchestrator.running = False
        orchestrator.message_bus = MagicMock()
        orchestrator.event_database = MagicMock()
        orchestrator.config = MagicMock()
        orchestrator.config.storage = MagicMock()
        orchestrator.config.face_recognition = MagicMock()
        orchestrator.get_pipeline_metrics.return_value = {
            'uptime': 0,
            'pipeline_status': 'stopped',
            'events_per_minute': 0
        }
        return orchestrator
    
    @pytest.fixture
    def legacy_system(self, mock_orchestrator):
        """Create legacy adapter that mimics old system."""
        return LegacyAdapter(mock_orchestrator)
    
    def test_start_stop_interface(self, legacy_system):
        """Test start/stop methods exist and work."""
        # Should have start method
        assert hasattr(legacy_system, 'start')
        assert callable(legacy_system.start)
        
        # Should have stop method
        assert hasattr(legacy_system, 'stop')
        assert callable(legacy_system.stop)
        
        # Test calling them
        legacy_system.start()
        assert legacy_system.running
        
        legacy_system.stop()
        assert not legacy_system.running
    
    def test_doorbell_callback_interface(self, legacy_system):
        """Test doorbell callback interface compatibility."""
        # Should have on_doorbell_pressed method
        assert hasattr(legacy_system, 'on_doorbell_pressed')
        assert callable(legacy_system.on_doorbell_pressed)
        
        # Test calling with channel parameter (GPIO style)
        legacy_system.on_doorbell_pressed(channel=18)
        
        # Should publish to message bus
        legacy_system.orchestrator.message_bus.publish.assert_called()
    
    def test_settings_attribute(self, legacy_system):
        """Test settings attribute exists with expected fields."""
        assert hasattr(legacy_system, 'settings')
        settings = legacy_system.settings
        
        # Check expected attributes
        expected_attrs = [
            'DEBOUNCE_TIME',
            'CAPTURES_DIR',
            'CROPPED_FACES_DIR',
            'KNOWN_FACES_DIR',
            'BLACKLIST_FACES_DIR',
            'LOGS_DIR'
        ]
        
        for attr in expected_attrs:
            assert hasattr(settings, attr), f"Missing settings attribute: {attr}"
    
    def test_face_manager_attribute(self, legacy_system):
        """Test face_manager attribute exists with expected methods."""
        assert hasattr(legacy_system, 'face_manager')
        face_manager = legacy_system.face_manager
        
        # Check expected methods
        expected_methods = [
            'load_known_faces',
            'load_blacklist_faces',
            'get_known_faces_count'
        ]
        
        for method in expected_methods:
            assert hasattr(face_manager, method), f"Missing face_manager method: {method}"
            assert callable(getattr(face_manager, method))
    
    def test_camera_attribute(self, legacy_system):
        """Test camera attribute exists with expected methods."""
        assert hasattr(legacy_system, 'camera')
        camera = legacy_system.camera
        
        # Check expected methods
        expected_methods = [
            'initialize',
            'capture_image',
            'cleanup'
        ]
        
        for method in expected_methods:
            assert hasattr(camera, method), f"Missing camera method: {method}"
            assert callable(getattr(camera, method))
    
    def test_gpio_attribute(self, legacy_system):
        """Test gpio attribute exists with expected methods."""
        assert hasattr(legacy_system, 'gpio')
        gpio = legacy_system.gpio
        
        # Check expected methods
        expected_methods = [
            'setup_doorbell_button',
            'cleanup'
        ]
        
        for method in expected_methods:
            assert hasattr(gpio, method), f"Missing gpio method: {method}"
            assert callable(getattr(gpio, method))


class TestWebInterfaceCompatibility:
    """Test compatibility with web interface expectations."""
    
    @pytest.fixture
    def system_with_adapter(self):
        """Create system with legacy adapter."""
        with patch('src.integration.orchestrator_manager.PipelineOrchestrator'):
            manager = OrchestratorManager({'pipeline_config': {}})
            legacy = manager.get_legacy_interface()
            yield legacy
    
    def test_web_interface_can_get_status(self, system_with_adapter):
        """Test web interface can get system status."""
        status = system_with_adapter.get_system_status()
        
        assert isinstance(status, dict)
        assert 'status' in status
        assert status['status'] in ['running', 'stopped', 'error']
    
    def test_web_interface_can_get_captures(self, system_with_adapter):
        """Test web interface can get recent captures."""
        system_with_adapter.orchestrator.event_database.get_recent_events.return_value = []
        
        captures = system_with_adapter.get_recent_captures(limit=10)
        
        assert isinstance(captures, list)
    
    def test_web_interface_can_trigger_doorbell(self, system_with_adapter):
        """Test web interface can trigger doorbell for testing."""
        # Through GPIO proxy
        system_with_adapter.gpio.simulate_doorbell_press()
        
        # Should publish event
        system_with_adapter.orchestrator.message_bus.publish.assert_called()


class TestDataStructureCompatibility:
    """Test that data structures remain compatible."""
    
    @pytest.fixture
    def legacy_adapter(self):
        """Create legacy adapter."""
        orchestrator = MagicMock()
        orchestrator.config = MagicMock()
        orchestrator.config.storage = MagicMock()
        orchestrator.config.face_recognition = MagicMock()
        orchestrator.event_database = MagicMock()
        return LegacyAdapter(orchestrator)
    
    def test_settings_paths_are_pathlib(self, legacy_adapter):
        """Test that settings paths are Path objects (legacy compatibility)."""
        settings = legacy_adapter.settings
        
        assert isinstance(settings.CAPTURES_DIR, Path)
        assert isinstance(settings.CROPPED_FACES_DIR, Path)
        assert isinstance(settings.KNOWN_FACES_DIR, Path)
        assert isinstance(settings.BLACKLIST_FACES_DIR, Path)
        assert isinstance(settings.LOGS_DIR, Path)
    
    def test_debounce_time_is_numeric(self, legacy_adapter):
        """Test debounce time is numeric."""
        settings = legacy_adapter.settings
        
        assert isinstance(settings.DEBOUNCE_TIME, (int, float))
        assert settings.DEBOUNCE_TIME > 0


class TestMigrationCompatibility:
    """Test migration from legacy to new system."""
    
    def test_can_import_new_modules(self):
        """Test new modules can be imported."""
        # These should not raise ImportError
        from src.integration.orchestrator_manager import OrchestratorManager
        from src.integration.legacy_adapter import LegacyAdapter
        from src.integration.migration_utils import MigrationUtils
        from config.orchestrator_config import OrchestratorConfig
        
        # Verify they are classes
        assert isinstance(OrchestratorManager, type)
        assert isinstance(LegacyAdapter, type)
        assert isinstance(MigrationUtils, type)
        assert isinstance(OrchestratorConfig, type)
    
    def test_can_create_orchestrator_manager(self):
        """Test orchestrator manager can be created."""
        with patch('src.integration.orchestrator_manager.PipelineOrchestrator'):
            manager = OrchestratorManager()
            assert manager is not None
            assert hasattr(manager, 'start')
            assert hasattr(manager, 'stop')
    
    def test_migration_utils_available(self):
        """Test migration utilities are available."""
        from src.integration.migration_utils import MigrationUtils
        
        # Check methods exist
        assert hasattr(MigrationUtils, 'validate_migration_compatibility')
        assert hasattr(MigrationUtils, 'migrate_legacy_data')
        assert hasattr(MigrationUtils, 'create_backup')
        assert hasattr(MigrationUtils, 'verify_pipeline_health')
