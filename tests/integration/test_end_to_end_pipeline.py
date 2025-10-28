#!/usr/bin/env python3
"""
End-to-End Pipeline Tests

Complete pipeline flow testing from doorbell press to event processing.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.integration.orchestrator_manager import OrchestratorManager, SystemState
from src.communication.events import EventType


class TestEndToEndPipeline:
    """Test complete pipeline flow."""
    
    @pytest.fixture
    def mocked_components(self):
        """Mock all hardware and external components."""
        mocks = {}
        
        with patch('src.hardware.camera_handler.CameraHandler') as camera_mock, \
             patch('src.hardware.gpio_handler.GPIOHandler') as gpio_mock, \
             patch('src.storage.event_database.EventDatabase') as db_mock, \
             patch('src.communication.message_bus.MessageBus') as bus_mock:
            
            # Setup camera mock
            mocks['camera'] = MagicMock()
            camera_mock.create.return_value = mocks['camera']
            
            # Setup GPIO mock
            mocks['gpio'] = MagicMock()
            gpio_mock.return_value = mocks['gpio']
            
            # Setup database mock
            mocks['database'] = MagicMock()
            db_mock.return_value = mocks['database']
            
            # Setup message bus mock
            mocks['message_bus'] = MagicMock()
            bus_mock.return_value = mocks['message_bus']
            
            yield mocks
    
    @pytest.fixture
    def pipeline_system(self, mocked_components):
        """Create full pipeline system."""
        config = {
            'health_check_interval': 1.0,
            'auto_recovery_enabled': False,
            'pipeline_config': {
                'frame_capture': {'enabled': False},  # Disable for testing
                'motion_detection': {'enabled': False},
                'face_detection': {'enabled': False, 'worker_count': 1},
                'face_recognition': {'enabled': False, 'worker_count': 1},
                'event_processing': {'enabled': False}
            }
        }
        
        with patch('src.pipeline.orchestrator.PipelineOrchestrator'):
            manager = OrchestratorManager(config)
            yield manager
            if manager.running:
                manager.stop()
    
    def test_system_startup_sequence(self, pipeline_system):
        """Test complete system startup sequence."""
        # System should be in INITIALIZING state
        assert pipeline_system.state == SystemState.INITIALIZING
        
        # Start the system
        pipeline_system.start()
        
        # System should transition to RUNNING
        assert pipeline_system.state == SystemState.RUNNING
        assert pipeline_system.running is True
        
        # Orchestrator should be started
        assert pipeline_system.orchestrator.start.called
    
    def test_doorbell_event_flow(self, pipeline_system):
        """Test doorbell press triggers correct event flow."""
        pipeline_system.start()
        
        # Trigger doorbell
        result = pipeline_system.trigger_doorbell()
        
        # Should succeed
        assert result['status'] == 'success'
        assert 'event_id' in result
        
        # Event should be published to message bus
        pipeline_system.orchestrator.message_bus.publish.assert_called()
        
        # Verify event data
        call_args = pipeline_system.orchestrator.message_bus.publish.call_args
        assert call_args[0][0] == 'doorbell_events'  # Topic
        event = call_args[0][1]  # Event object
        assert event.event_type == EventType.DOORBELL_PRESSED
    
    def test_legacy_interface_integration(self, pipeline_system):
        """Test legacy interface works with pipeline."""
        pipeline_system.start()
        
        # Get legacy interface
        legacy = pipeline_system.get_legacy_interface()
        
        # Trigger doorbell through legacy interface
        legacy.on_doorbell_pressed(channel=18)
        
        # Should publish to message bus
        pipeline_system.orchestrator.message_bus.publish.assert_called()
    
    def test_health_monitoring_integration(self, pipeline_system):
        """Test health monitoring during operation."""
        pipeline_system.start()
        
        # Get initial health
        health = pipeline_system.get_health_status()
        
        assert health.state == SystemState.RUNNING
        assert health.uptime >= 0
        assert health.error_count == 0
        
        # Simulate some operations
        pipeline_system.trigger_doorbell()
        
        # Get health again
        health2 = pipeline_system.get_health_status()
        assert health2.uptime >= health.uptime
    
    def test_graceful_shutdown_sequence(self, pipeline_system):
        """Test graceful shutdown sequence."""
        pipeline_system.start()
        assert pipeline_system.state == SystemState.RUNNING
        
        # Stop the system
        pipeline_system.stop()
        
        # System should transition to STOPPED
        assert pipeline_system.state == SystemState.STOPPED
        assert pipeline_system.running is False
        
        # Orchestrator should be shut down
        pipeline_system.orchestrator.shutdown.assert_called()
    
    def test_callback_system_integration(self, pipeline_system):
        """Test event callback system."""
        pipeline_system.start()
        
        # Register callbacks
        doorbell_events = []
        health_events = []
        
        def doorbell_callback(data):
            doorbell_events.append(data)
        
        def health_callback(data):
            health_events.append(data)
        
        pipeline_system.register_event_callback('doorbell_trigger', doorbell_callback)
        pipeline_system.register_event_callback('health_update', health_callback)
        
        # Trigger events
        pipeline_system._notify_callbacks('doorbell_trigger', {'test': 'data'})
        pipeline_system._notify_callbacks('health_update', {'status': 'ok'})
        
        # Verify callbacks were called
        assert len(doorbell_events) == 1
        assert len(health_events) == 1


class TestPipelineErrorHandling:
    """Test error handling in pipeline."""
    
    @pytest.fixture
    def error_prone_system(self):
        """Create system that may encounter errors."""
        with patch('src.integration.orchestrator_manager.PipelineOrchestrator') as mock:
            orchestrator = MagicMock()
            orchestrator.running = False
            orchestrator.message_bus = MagicMock()
            
            # Make some calls fail
            orchestrator.get_pipeline_metrics.side_effect = [
                Exception("Metrics error"),
                {'pipeline_status': 'running', 'uptime': 0}
            ]
            
            mock.return_value = orchestrator
            
            config = {
                'auto_recovery_enabled': False,
                'pipeline_config': {}
            }
            
            manager = OrchestratorManager(config)
            manager.orchestrator = orchestrator
            yield manager
    
    def test_metrics_error_handling(self, error_prone_system):
        """Test handling of metrics collection errors."""
        error_prone_system.start()
        
        # First call should fail but not crash
        health = error_prone_system.get_health_status()
        
        # Should have error in pipeline status
        assert 'error' in health.pipeline_status
        
        # Second call should succeed
        health2 = error_prone_system.get_health_status()
        assert 'pipeline_status' in health2.pipeline_status


class TestPipelinePerformance:
    """Test pipeline performance characteristics."""
    
    @pytest.fixture
    def performance_system(self):
        """Create system for performance testing."""
        with patch('src.integration.orchestrator_manager.PipelineOrchestrator'):
            config = {
                'health_check_interval': 0.1,  # Fast for testing
                'pipeline_config': {}
            }
            manager = OrchestratorManager(config)
            yield manager
            if manager.running:
                manager.stop()
    
    def test_startup_time(self, performance_system):
        """Test system startup time is reasonable."""
        start = time.time()
        performance_system.start()
        elapsed = time.time() - start
        
        # Startup should be relatively quick (< 2 seconds for mocked system)
        assert elapsed < 2.0
        assert performance_system.state == SystemState.RUNNING
    
    def test_multiple_doorbell_triggers(self, performance_system):
        """Test handling multiple doorbell triggers."""
        performance_system.start()
        
        # Trigger doorbell multiple times rapidly
        results = []
        for _ in range(10):
            result = performance_system.trigger_doorbell()
            results.append(result)
        
        # All should succeed
        assert all(r['status'] == 'success' for r in results)
        
        # Should have 10 unique event IDs
        event_ids = [r['event_id'] for r in results]
        assert len(set(event_ids)) == 10
    
    def test_health_check_performance(self, performance_system):
        """Test health check performance."""
        performance_system.start()
        
        # Multiple health checks should be fast
        start = time.time()
        for _ in range(100):
            health = performance_system.get_health_status()
        elapsed = time.time() - start
        
        # 100 health checks should take < 1 second
        assert elapsed < 1.0


class TestSystemRecovery:
    """Test system recovery mechanisms."""
    
    @pytest.fixture
    def recovery_system(self):
        """Create system with recovery enabled."""
        with patch('src.integration.orchestrator_manager.PipelineOrchestrator'):
            config = {
                'auto_recovery_enabled': True,
                'max_restart_attempts': 3,
                'pipeline_config': {}
            }
            manager = OrchestratorManager(config)
            yield manager
            if manager.running:
                manager.stop()
    
    def test_error_detection(self, recovery_system):
        """Test system detects errors."""
        recovery_system.start()
        
        # Simulate error
        recovery_system.error_count += 1
        recovery_system.last_error = "Test error"
        recovery_system.state = SystemState.ERROR
        
        health = recovery_system.get_health_status()
        assert health.error_count > 0
        assert health.last_error is not None
    
    def test_performance_degradation_detection(self, recovery_system):
        """Test detection of performance degradation."""
        recovery_system.start()
        
        # Create metrics indicating poor performance
        poor_metrics = {
            'error': 'Some issue',
            'queue_status': {'backlog_warning': True},
            'worker_status': {'high_cpu': True}
        }
        
        score = recovery_system._calculate_performance_score(poor_metrics)
        
        # Score should be degraded
        assert score < 0.7
