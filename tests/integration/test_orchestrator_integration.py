#!/usr/bin/env python3
"""
Orchestrator Integration Tests

Comprehensive testing of the pipeline orchestrator integration.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.integration.orchestrator_manager import OrchestratorManager, SystemState, SystemHealth
from src.integration.legacy_adapter import LegacyAdapter


class TestOrchestratorManager:
    """Test orchestrator manager functionality."""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        with patch('src.integration.orchestrator_manager.PipelineOrchestrator') as mock:
            orchestrator = MagicMock()
            orchestrator.running = False
            orchestrator.message_bus = MagicMock()
            orchestrator.event_database = MagicMock()
            orchestrator.config = MagicMock()
            orchestrator.get_pipeline_metrics.return_value = {
                'pipeline_status': 'stopped',
                'uptime': 0,
                'stages': {},
                'queue_status': {},
                'worker_status': {},
                'message_bus_stats': {},
                'events_per_minute': 0
            }
            mock.return_value = orchestrator
            yield orchestrator
    
    @pytest.fixture
    def orchestrator_manager(self, mock_orchestrator):
        """Create test orchestrator manager."""
        config = {
            'health_check_interval': 1.0,
            'auto_recovery_enabled': False,  # Disable for testing
            'pipeline_config': {
                'frame_capture': {'enabled': True},
                'face_detection': {'enabled': True, 'worker_count': 1},
                'event_processing': {'enabled': True}
            }
        }
        
        with patch('src.integration.orchestrator_manager.OrchestratorConfig'):
            manager = OrchestratorManager(config)
            manager.orchestrator = mock_orchestrator
            yield manager
            if manager.running:
                manager.stop()
    
    def test_manager_initialization(self, orchestrator_manager):
        """Test manager initializes correctly."""
        assert orchestrator_manager.state == SystemState.INITIALIZING
        assert not orchestrator_manager.running
        assert orchestrator_manager.error_count == 0
        assert orchestrator_manager.warning_count == 0
    
    def test_manager_start_stop(self, orchestrator_manager):
        """Test manager startup and shutdown."""
        # Test startup
        orchestrator_manager.start()
        assert orchestrator_manager.state == SystemState.RUNNING
        assert orchestrator_manager.running is True
        
        # Verify orchestrator was started
        orchestrator_manager.orchestrator.start.assert_called_once()
        
        # Test health status
        health = orchestrator_manager.get_health_status()
        assert isinstance(health, SystemHealth)
        assert health.state == SystemState.RUNNING
        assert health.uptime >= 0
        
        # Test shutdown
        orchestrator_manager.stop()
        assert orchestrator_manager.state == SystemState.STOPPED
        assert orchestrator_manager.running is False
        
        # Verify orchestrator was shut down
        orchestrator_manager.orchestrator.shutdown.assert_called_once()
    
    def test_health_status_reporting(self, orchestrator_manager):
        """Test health status reporting."""
        orchestrator_manager.start()
        
        # Get health status
        health = orchestrator_manager.get_health_status()
        
        assert isinstance(health, SystemHealth)
        assert health.state == SystemState.RUNNING
        assert health.uptime >= 0
        assert health.error_count == 0
        assert health.performance_score > 0
        assert isinstance(health.pipeline_status, dict)
    
    def test_doorbell_trigger(self, orchestrator_manager):
        """Test doorbell event triggering."""
        orchestrator_manager.start()
        
        # Trigger doorbell
        result = orchestrator_manager.trigger_doorbell()
        
        assert result['status'] == 'success'
        assert 'event_id' in result
        
        # Verify event was published
        orchestrator_manager.orchestrator.message_bus.publish.assert_called()
    
    def test_event_callback_registration(self, orchestrator_manager):
        """Test event callback registration."""
        callback_called = []
        
        def test_callback(data):
            callback_called.append(data)
        
        # Register callback
        orchestrator_manager.register_event_callback('test_event', test_callback)
        
        # Trigger callback
        orchestrator_manager._notify_callbacks('test_event', {'test': 'data'})
        
        # Verify callback was called
        assert len(callback_called) == 1
        assert callback_called[0] == {'test': 'data'}
    
    def test_performance_score_calculation(self, orchestrator_manager):
        """Test performance score calculation."""
        # Test with good metrics
        good_metrics = {
            'pipeline_status': 'running',
            'queue_status': {},
            'worker_status': {}
        }
        score = orchestrator_manager._calculate_performance_score(good_metrics)
        assert score == 1.0
        
        # Test with error
        error_metrics = {
            'error': 'Some error',
            'queue_status': {},
            'worker_status': {}
        }
        score = orchestrator_manager._calculate_performance_score(error_metrics)
        assert score < 1.0
        
        # Test with queue backlog
        backlog_metrics = {
            'queue_status': {'backlog_warning': True},
            'worker_status': {}
        }
        score = orchestrator_manager._calculate_performance_score(backlog_metrics)
        assert score < 1.0
    
    def test_get_legacy_interface(self, orchestrator_manager):
        """Test getting legacy adapter interface."""
        legacy = orchestrator_manager.get_legacy_interface()
        
        assert isinstance(legacy, LegacyAdapter)
        assert legacy.orchestrator == orchestrator_manager.orchestrator


class TestLegacyAdapter:
    """Test legacy adapter compatibility."""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        orchestrator = MagicMock()
        orchestrator.running = False
        orchestrator.message_bus = MagicMock()
        orchestrator.event_database = MagicMock()
        orchestrator.config = MagicMock()
        orchestrator.config.storage = MagicMock()
        orchestrator.config.face_recognition = MagicMock()
        orchestrator.event_database.get_recent_events.return_value = []
        orchestrator.get_pipeline_metrics.return_value = {
            'uptime': 100,
            'pipeline_status': 'running',
            'events_per_minute': 5
        }
        return orchestrator
    
    @pytest.fixture
    def legacy_adapter(self, mock_orchestrator):
        """Create test legacy adapter."""
        return LegacyAdapter(mock_orchestrator)
    
    def test_adapter_initialization(self, legacy_adapter):
        """Test adapter initializes with all expected attributes."""
        assert hasattr(legacy_adapter, 'settings')
        assert hasattr(legacy_adapter, 'face_manager')
        assert hasattr(legacy_adapter, 'camera')
        assert hasattr(legacy_adapter, 'gpio')
        assert not legacy_adapter.running
    
    def test_adapter_start_stop(self, legacy_adapter):
        """Test adapter start and stop."""
        # Test start
        legacy_adapter.start()
        assert legacy_adapter.running
        legacy_adapter.orchestrator.start.assert_called_once()
        
        # Test stop
        legacy_adapter.stop()
        assert not legacy_adapter.running
        legacy_adapter.orchestrator.shutdown.assert_called_once()
    
    def test_doorbell_press_handling(self, legacy_adapter):
        """Test doorbell press handling."""
        # Trigger doorbell press
        legacy_adapter.on_doorbell_pressed(channel=18)
        
        # Verify event was published
        legacy_adapter.orchestrator.message_bus.publish.assert_called()
        
        # Verify debounce
        legacy_adapter.on_doorbell_pressed(channel=18)
        # Should still be called only once (debounced)
        assert legacy_adapter.orchestrator.message_bus.publish.call_count == 1
    
    def test_get_recent_captures(self, legacy_adapter):
        """Test getting recent captures."""
        # Mock return data
        mock_events = [
            {'timestamp': '2024-01-01T12:00:00', 'type': 'face_recognition'},
            {'timestamp': '2024-01-01T12:05:00', 'type': 'face_recognition'}
        ]
        legacy_adapter.orchestrator.event_database.get_recent_events.return_value = mock_events
        
        # Get recent captures
        captures = legacy_adapter.get_recent_captures(limit=10)
        
        assert len(captures) == 2
        legacy_adapter.orchestrator.event_database.get_recent_events.assert_called_with(
            event_type='face_recognition',
            limit=10
        )
    
    def test_get_system_status(self, legacy_adapter):
        """Test system status reporting."""
        legacy_adapter.running = True
        
        status = legacy_adapter.get_system_status()
        
        assert status['status'] == 'running'
        assert 'uptime' in status
        assert 'pipeline_health' in status
        assert 'events_processed' in status
    
    def test_settings_proxy(self, legacy_adapter):
        """Test settings proxy attributes."""
        settings = legacy_adapter.settings
        
        assert hasattr(settings, 'DEBOUNCE_TIME')
        assert hasattr(settings, 'CAPTURES_DIR')
        assert hasattr(settings, 'KNOWN_FACES_DIR')
        assert hasattr(settings, 'BLACKLIST_FACES_DIR')
        assert hasattr(settings, 'LOGS_DIR')
    
    def test_face_manager_proxy(self, legacy_adapter):
        """Test face manager proxy methods."""
        face_manager = legacy_adapter.face_manager
        
        # These should not raise errors
        face_manager.load_known_faces()
        face_manager.load_blacklist_faces()
        
        # Get count
        count = face_manager.get_known_faces_count()
        assert isinstance(count, int)
    
    def test_camera_proxy(self, legacy_adapter):
        """Test camera proxy methods."""
        camera = legacy_adapter.camera
        
        # These should not raise errors
        camera.initialize()
        camera.cleanup()
    
    def test_gpio_proxy(self, legacy_adapter):
        """Test GPIO proxy methods."""
        gpio = legacy_adapter.gpio
        
        # Setup doorbell button
        callback = Mock()
        gpio.setup_doorbell_button(callback)
        
        # Simulate doorbell press
        gpio.simulate_doorbell_press()
        legacy_adapter.orchestrator.message_bus.publish.assert_called()
        
        # Cleanup
        gpio.cleanup()


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def full_system(self):
        """Create a full system setup (mocked hardware)."""
        with patch('src.hardware.camera_handler.CameraHandler'), \
             patch('src.hardware.gpio_handler.GPIOHandler'), \
             patch('src.storage.event_database.EventDatabase'), \
             patch('src.communication.message_bus.MessageBus'):
            
            config = {
                'health_check_interval': 1.0,
                'auto_recovery_enabled': False,
                'pipeline_config': {
                    'frame_capture': {'enabled': False},  # Disable for testing
                    'face_detection': {'enabled': False, 'worker_count': 1},
                    'event_processing': {'enabled': False}
                }
            }
            
            manager = OrchestratorManager(config)
            yield manager
            if manager.running:
                manager.stop()
    
    def test_full_system_lifecycle(self, full_system):
        """Test complete system lifecycle."""
        # Start system
        full_system.start()
        assert full_system.state == SystemState.RUNNING
        
        # Get legacy interface
        legacy = full_system.get_legacy_interface()
        assert legacy is not None
        
        # Trigger doorbell
        result = full_system.trigger_doorbell()
        assert result['status'] == 'success'
        
        # Check health
        health = full_system.get_health_status()
        assert health.state == SystemState.RUNNING
        
        # Stop system
        full_system.stop()
        assert full_system.state == SystemState.STOPPED
