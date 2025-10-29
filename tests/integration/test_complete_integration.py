#!/usr/bin/env python3
"""
Complete Integration Tests

End-to-end tests for the complete pipeline architecture integration.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.integration.orchestrator_manager import OrchestratorManager, SystemState
from src.integration.legacy_adapter import LegacyAdapter


class TestCompleteIntegration:
    """Test complete system integration."""
    
    @pytest.fixture
    def orchestrator_manager(self):
        """Create orchestrator manager for testing."""
        return OrchestratorManager()
    
    def test_orchestrator_manager_creation(self, orchestrator_manager):
        """Test orchestrator manager can be created."""
        assert orchestrator_manager is not None
        assert orchestrator_manager.state == SystemState.INITIALIZING
    
    def test_orchestrator_manager_start_stop(self, orchestrator_manager):
        """Test orchestrator manager start/stop lifecycle."""
        # Start should work
        orchestrator_manager.start()
        assert orchestrator_manager.running
        
        # Wait briefly for initialization
        time.sleep(1.0)
        
        # Should be in running state
        assert orchestrator_manager.state in [SystemState.STARTING, SystemState.RUNNING]
        
        # Stop should work
        orchestrator_manager.stop()
        assert not orchestrator_manager.running
        assert orchestrator_manager.state == SystemState.STOPPED
    
    def test_orchestrator_health_status(self, orchestrator_manager):
        """Test health status reporting."""
        orchestrator_manager.start()
        
        try:
            # Get health status
            health = orchestrator_manager.get_health_status()
            
            # Validate health status
            assert health is not None
            assert hasattr(health, 'state')
            assert hasattr(health, 'uptime')
            assert hasattr(health, 'performance_score')
            assert health.performance_score >= 0.0
            assert health.performance_score <= 1.0
            
        finally:
            orchestrator_manager.stop()
    
    def test_legacy_interface_creation(self, orchestrator_manager):
        """Test legacy interface can be created."""
        legacy_interface = orchestrator_manager.get_legacy_interface()
        
        assert legacy_interface is not None
        assert isinstance(legacy_interface, LegacyAdapter)
    
    def test_doorbell_trigger(self, orchestrator_manager):
        """Test doorbell trigger functionality."""
        orchestrator_manager.start()
        
        try:
            # Trigger doorbell
            result = orchestrator_manager.trigger_doorbell()
            
            # Should return success
            assert result['status'] == 'success'
            assert 'event_id' in result
            
        finally:
            orchestrator_manager.stop()
    
    def test_web_interface_compatibility(self, orchestrator_manager):
        """Test web interface can be created with orchestrator."""
        from src.web_interface import create_web_app
        
        legacy_interface = orchestrator_manager.get_legacy_interface()
        app = create_web_app(legacy_interface)
        
        assert app is not None
        
        # Test that basic routes exist
        with app.test_client() as client:
            # Should be able to access status endpoint
            response = client.get('/api/status')
            assert response is not None
    
    def test_event_callbacks(self, orchestrator_manager):
        """Test event callback registration."""
        callback_called = {'count': 0}
        
        def test_callback(data):
            callback_called['count'] += 1
        
        # Register callback
        orchestrator_manager.register_event_callback('test_event', test_callback)
        
        # Callback should be registered
        assert 'test_event' in orchestrator_manager.event_callbacks
        assert test_callback in orchestrator_manager.event_callbacks['test_event']


class TestSystemResilience:
    """Test system resilience and error handling."""
    
    @pytest.fixture
    def orchestrator_manager(self):
        """Create orchestrator manager for testing."""
        return OrchestratorManager()
    
    def test_double_start_protection(self, orchestrator_manager):
        """Test that double start is handled gracefully."""
        orchestrator_manager.start()
        
        try:
            # Second start should not crash
            orchestrator_manager.start()
            
            # Should still be running
            assert orchestrator_manager.running
            
        finally:
            orchestrator_manager.stop()
    
    def test_stop_before_start(self, orchestrator_manager):
        """Test stopping before starting doesn't crash."""
        # Should not crash
        orchestrator_manager.stop()
        
        # Should not be running
        assert not orchestrator_manager.running
    
    def test_health_status_when_stopped(self, orchestrator_manager):
        """Test health status can be retrieved when stopped."""
        # Should not crash
        health = orchestrator_manager.get_health_status()
        
        assert health is not None
        assert health.state == SystemState.INITIALIZING


class TestBackwardCompatibility:
    """Test backward compatibility with legacy system."""
    
    @pytest.fixture
    def orchestrator_manager(self):
        """Create orchestrator manager for testing."""
        return OrchestratorManager()
    
    @pytest.fixture
    def legacy_interface(self, orchestrator_manager):
        """Create legacy interface."""
        return orchestrator_manager.get_legacy_interface()
    
    def test_legacy_attributes_exist(self, legacy_interface):
        """Test that legacy attributes are present."""
        # Should have legacy attributes
        assert hasattr(legacy_interface, 'settings')
        assert hasattr(legacy_interface, 'face_manager')
        assert hasattr(legacy_interface, 'camera')
        assert hasattr(legacy_interface, 'gpio')
    
    def test_legacy_methods_exist(self, legacy_interface):
        """Test that legacy methods are present."""
        # Should have legacy methods
        assert hasattr(legacy_interface, 'start')
        assert hasattr(legacy_interface, 'stop')
        assert hasattr(legacy_interface, 'on_doorbell_pressed')
        assert hasattr(legacy_interface, 'get_recent_captures')
        assert hasattr(legacy_interface, 'get_system_status')
    
    def test_legacy_start_stop(self, legacy_interface):
        """Test legacy start/stop interface."""
        # Start should work
        legacy_interface.start()
        assert legacy_interface.running
        
        # Stop should work
        legacy_interface.stop()
        assert not legacy_interface.running
    
    def test_legacy_doorbell_press(self, legacy_interface):
        """Test legacy doorbell press interface."""
        legacy_interface.start()
        
        try:
            # Should not crash
            legacy_interface.on_doorbell_pressed()
            
            # Wait briefly
            time.sleep(0.1)
            
        finally:
            legacy_interface.stop()
    
    def test_legacy_system_status(self, legacy_interface):
        """Test legacy system status interface."""
        legacy_interface.start()
        
        try:
            # Should return status dict
            status = legacy_interface.get_system_status()
            
            assert status is not None
            assert isinstance(status, dict)
            assert 'status' in status
            
        finally:
            legacy_interface.stop()
    
    def test_legacy_settings_attributes(self, legacy_interface):
        """Test legacy settings attributes."""
        settings = legacy_interface.settings
        
        # Should have expected attributes
        assert hasattr(settings, 'DEBOUNCE_TIME')
        assert hasattr(settings, 'CAPTURES_DIR')
        assert hasattr(settings, 'KNOWN_FACES_DIR')
        assert hasattr(settings, 'BLACKLIST_FACES_DIR')
        assert hasattr(settings, 'LOGS_DIR')


class TestWebInterfaceIntegration:
    """Test web interface integration with pipeline."""
    
    @pytest.fixture
    def orchestrator_manager(self):
        """Create orchestrator manager for testing."""
        return OrchestratorManager()
    
    @pytest.fixture
    def web_app(self, orchestrator_manager):
        """Create web app with pipeline backend."""
        from src.web_interface import create_web_app
        
        legacy_interface = orchestrator_manager.get_legacy_interface()
        return create_web_app(legacy_interface)
    
    def test_status_endpoint(self, web_app):
        """Test status API endpoint."""
        with web_app.test_client() as client:
            response = client.get('/api/status')
            assert response is not None
    
    def test_faces_endpoint(self, web_app):
        """Test faces API endpoint."""
        with web_app.test_client() as client:
            response = client.get('/api/faces')
            assert response is not None
    
    def test_recent_captures_endpoint(self, web_app):
        """Test recent captures endpoint."""
        with web_app.test_client() as client:
            response = client.get('/api/recent_captures')
            assert response is not None


@pytest.mark.slow
class TestPerformanceIntegration:
    """Test performance of integrated system."""
    
    @pytest.fixture
    def orchestrator_manager(self):
        """Create orchestrator manager for testing."""
        return OrchestratorManager()
    
    def test_startup_time(self, orchestrator_manager):
        """Test system startup time is reasonable."""
        start_time = time.time()
        orchestrator_manager.start()
        
        try:
            startup_duration = time.time() - start_time
            
            # Should start within reasonable time
            assert startup_duration < 10.0, f"Startup took {startup_duration:.2f}s"
            
        finally:
            orchestrator_manager.stop()
    
    def test_shutdown_time(self, orchestrator_manager):
        """Test system shutdown time is reasonable."""
        orchestrator_manager.start()
        time.sleep(1.0)
        
        start_time = time.time()
        orchestrator_manager.stop()
        shutdown_duration = time.time() - start_time
        
        # Should stop within reasonable time
        assert shutdown_duration < 5.0, f"Shutdown took {shutdown_duration:.2f}s"
    
    def test_health_check_performance(self, orchestrator_manager):
        """Test health check is fast."""
        orchestrator_manager.start()
        
        try:
            start_time = time.time()
            health = orchestrator_manager.get_health_status()
            check_duration = time.time() - start_time
            
            # Health check should be fast
            assert check_duration < 1.0, f"Health check took {check_duration:.2f}s"
            
        finally:
            orchestrator_manager.stop()
