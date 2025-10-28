# Issue #12: Pipeline Orchestrator Integration and Main System Controller

## 📋 **Overview**

Replace the legacy `DoorbellSecuritySystem` architecture with the new Frigate-inspired `PipelineOrchestrator` as the main system controller. This issue implements the final integration that transforms the system from a traditional monolithic approach to a modern pipeline-based architecture with multi-process workers, event-driven communication, and comprehensive monitoring.

## 🎯 **Objectives**

### **Primary Goals**
1. **Replace Legacy Architecture**: Migrate from `DoorbellSecuritySystem` to `PipelineOrchestrator` as main controller
2. **Complete Pipeline Integration**: Ensure all pipeline stages work seamlessly together
3. **Production Readiness**: Implement robust error handling, monitoring, and recovery mechanisms
4. **Backward Compatibility**: Maintain existing API endpoints and web interface functionality
5. **Performance Optimization**: Achieve superior performance through pipeline architecture

### **Success Criteria**
- Complete migration from legacy to pipeline architecture
- All existing functionality preserved and enhanced
- Improved performance metrics (25% faster processing, 40% better resource utilization)
- Zero-downtime deployment capability
- Comprehensive monitoring and health checking
- Full test coverage for orchestrator and integration points

## 🏗️ **Architecture Transformation**

### **Current State (Legacy)**
```
app.py → DoorbellSecuritySystem → [FaceManager, CameraHandler, etc.] → Web Interface
```

### **Target State (Pipeline)**
```
app.py → PipelineOrchestrator → [Frame Capture → Motion Detection → Face Detection → Recognition → Event Processing] → Web Interface + Notifications
```

### **Integration Flow**
1. **Startup**: Initialize orchestrator instead of legacy system
2. **Event Flow**: GPIO events trigger pipeline instead of direct processing
3. **Web Interface**: Connect to pipeline events instead of legacy system events
4. **Monitoring**: Use orchestrator metrics instead of system-level monitoring

## 📁 **Implementation Specifications**

### **Files to Create/Modify**

#### **New Files**
```
src/integration/                               # Integration layer
    __init__.py
    legacy_adapter.py                          # Legacy API compatibility layer
    migration_utils.py                         # Migration utilities and helpers
    orchestrator_manager.py                   # High-level orchestrator management
    
src/main.py                                   # New main entry point using orchestrator
config/orchestrator_config.py                # Orchestrator-specific configuration
tests/integration/                            # Integration tests
    test_orchestrator_integration.py          # Orchestrator integration tests
    test_legacy_compatibility.py              # Legacy compatibility tests
    test_end_to_end_pipeline.py              # Complete pipeline tests
```

#### **Modified Files**
```
app.py                                        # Update to use orchestrator
src/web_interface.py                          # Connect to pipeline events
src/doorbell_security.py                     # Deprecated with compatibility layer
config/settings.py                           # Enhanced configuration management
```

### **Core Component: Orchestrator Manager**
```python
#!/usr/bin/env python3
"""
Orchestrator Manager - High-Level Pipeline Management

Provides a high-level interface to the PipelineOrchestrator with additional
management features for production deployment.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum

from src.pipeline.orchestrator import PipelineOrchestrator
from src.communication.message_bus import MessageBus
from src.communication.events import PipelineEvent, EventType
from config.orchestrator_config import OrchestratorConfig
from src.integration.legacy_adapter import LegacyAdapter

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """Overall system state."""
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class SystemHealth:
    """System health status."""
    state: SystemState
    uptime: float
    pipeline_status: Dict[str, Any]
    error_count: int
    warning_count: int
    last_error: Optional[str] = None
    performance_score: float = 1.0


class OrchestratorManager:
    """
    High-level management interface for the pipeline orchestrator.
    
    Provides additional features beyond the basic orchestrator:
    - Health monitoring and alerting
    - Automatic recovery and restart
    - Performance optimization
    - Legacy compatibility
    - Production deployment support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the orchestrator manager."""
        self.config = OrchestratorConfig(config or {})
        self.orchestrator = PipelineOrchestrator(self.config.pipeline_config)
        self.legacy_adapter = LegacyAdapter(self.orchestrator)
        
        # System state
        self.state = SystemState.INITIALIZING
        self.start_time: Optional[float] = None
        self.error_count = 0
        self.warning_count = 0
        self.last_error: Optional[str] = None
        
        # Monitoring and recovery
        self.health_check_interval = self.config.health_check_interval
        self.auto_recovery_enabled = self.config.auto_recovery_enabled
        self.max_restart_attempts = self.config.max_restart_attempts
        self.restart_attempts = 0
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {}
        
        # Background threads
        self.monitor_thread: Optional[threading.Thread] = None
        self.running = False
        
        logger.info("Orchestrator manager initialized")
    
    def start(self) -> None:
        """Start the orchestrator manager and pipeline."""
        if self.running:
            logger.warning("Orchestrator manager already running")
            return
        
        try:
            logger.info("Starting orchestrator manager...")
            self.state = SystemState.STARTING
            self.start_time = time.time()
            self.running = True
            
            # Start the pipeline orchestrator
            self.orchestrator.start()
            
            # Start monitoring
            self._start_monitoring()
            
            # Register for pipeline events
            self._setup_event_handlers()
            
            self.state = SystemState.RUNNING
            logger.info("✅ Orchestrator manager started successfully")
            
        except Exception as e:
            self.state = SystemState.ERROR
            self.last_error = str(e)
            self.error_count += 1
            logger.error(f"Failed to start orchestrator manager: {e}")
            raise
    
    def stop(self) -> None:
        """Stop the orchestrator manager and pipeline."""
        if not self.running:
            return
        
        logger.info("Stopping orchestrator manager...")
        self.state = SystemState.STOPPING
        self.running = False
        
        try:
            # Stop monitoring
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5.0)
            
            # Stop the orchestrator
            self.orchestrator.shutdown()
            
            self.state = SystemState.STOPPED
            logger.info("Orchestrator manager stopped")
            
        except Exception as e:
            self.state = SystemState.ERROR
            self.last_error = str(e)
            logger.error(f"Error stopping orchestrator manager: {e}")
    
    def get_health_status(self) -> SystemHealth:
        """Get comprehensive system health status."""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        # Get pipeline metrics
        pipeline_status = {}
        try:
            pipeline_status = self.orchestrator.get_pipeline_metrics()
        except Exception as e:
            logger.warning(f"Failed to get pipeline metrics: {e}")
            pipeline_status = {"error": str(e)}
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(pipeline_status)
        
        return SystemHealth(
            state=self.state,
            uptime=uptime,
            pipeline_status=pipeline_status,
            error_count=self.error_count,
            warning_count=self.warning_count,
            last_error=self.last_error,
            performance_score=performance_score
        )
    
    def register_event_callback(self, event_type: str, callback: Callable) -> None:
        """Register callback for system events."""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
    
    def trigger_doorbell(self) -> Dict[str, Any]:
        """Trigger doorbell event (for testing/web interface)."""
        try:
            # Create doorbell event
            event = PipelineEvent(
                event_type=EventType.DOORBELL_PRESSED,
                data={"trigger_source": "manual", "timestamp": time.time()},
                source="orchestrator_manager"
            )
            
            # Publish to message bus
            self.orchestrator.message_bus.publish('doorbell_events', event)
            
            return {
                "status": "success",
                "message": "Doorbell triggered",
                "event_id": event.event_id
            }
            
        except Exception as e:
            logger.error(f"Failed to trigger doorbell: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_legacy_interface(self):
        """Get legacy compatibility interface."""
        return self.legacy_adapter
    
    def _start_monitoring(self) -> None:
        """Start background monitoring thread."""
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
    
    def _monitoring_loop(self) -> None:
        """Background monitoring and recovery loop."""
        while self.running:
            try:
                # Perform health check
                health = self.get_health_status()
                
                # Check for issues
                if health.state == SystemState.ERROR:
                    self._handle_system_error(health)
                elif health.performance_score < 0.7:
                    self._handle_performance_degradation(health)
                
                # Sleep until next check
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(10.0)  # Longer sleep on error
    
    def _setup_event_handlers(self) -> None:
        """Setup handlers for pipeline events."""
        # Subscribe to critical events
        self.orchestrator.message_bus.subscribe(
            'system_errors', self._handle_system_event, 'orchestrator_manager'
        )
        self.orchestrator.message_bus.subscribe(
            'pipeline_health', self._handle_health_event, 'orchestrator_manager'
        )
    
    def _handle_system_event(self, message) -> None:
        """Handle system-level events."""
        event_data = message.data
        
        if event_data.get('type') == 'error':
            self.error_count += 1
            self.last_error = event_data.get('message', 'Unknown error')
            
            # Notify callbacks
            self._notify_callbacks('system_error', event_data)
    
    def _handle_health_event(self, message) -> None:
        """Handle health monitoring events."""
        # Notify callbacks
        self._notify_callbacks('health_update', message.data)
    
    def _handle_system_error(self, health: SystemHealth) -> None:
        """Handle system errors with recovery attempts."""
        if not self.auto_recovery_enabled:
            return
        
        if self.restart_attempts < self.max_restart_attempts:
            logger.warning(f"Attempting system recovery (attempt {self.restart_attempts + 1})")
            self.restart_attempts += 1
            
            try:
                # Attempt restart
                self.orchestrator.shutdown()
                time.sleep(5.0)  # Brief pause
                self.orchestrator.start()
                
                # Reset restart counter on success
                self.restart_attempts = 0
                
            except Exception as e:
                logger.error(f"Recovery attempt failed: {e}")
        else:
            logger.error("Maximum restart attempts reached, manual intervention required")
    
    def _handle_performance_degradation(self, health: SystemHealth) -> None:
        """Handle performance degradation."""
        logger.warning(f"Performance degradation detected: {health.performance_score:.2f}")
        
        # Notify callbacks
        self._notify_callbacks('performance_warning', {
            'score': health.performance_score,
            'timestamp': time.time()
        })
    
    def _calculate_performance_score(self, pipeline_status: Dict[str, Any]) -> float:
        """Calculate overall performance score (0.0 to 1.0)."""
        try:
            # Base score
            score = 1.0
            
            # Penalize for errors
            if 'error' in pipeline_status:
                score *= 0.5
            
            # Check queue health
            queue_status = pipeline_status.get('queue_status', {})
            if queue_status.get('backlog_warning'):
                score *= 0.8
            
            # Check worker health
            worker_status = pipeline_status.get('worker_status', {})
            if worker_status.get('high_cpu'):
                score *= 0.9
            
            return max(0.0, score)
            
        except Exception:
            return 0.5  # Default to moderate score on calculation error
    
    def _notify_callbacks(self, event_type: str, data: Any) -> None:
        """Notify registered callbacks."""
        callbacks = self.event_callbacks.get(event_type, [])
        for callback in callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event_type}: {e}")
```

### **Legacy Adapter Implementation**
```python
#!/usr/bin/env python3
"""
Legacy Adapter - Compatibility Layer

Provides backward compatibility with the existing DoorbellSecuritySystem API
while using the new pipeline orchestrator internally.
"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from src.pipeline.orchestrator import PipelineOrchestrator
from src.communication.events import PipelineEvent, EventType

logger = logging.getLogger(__name__)


class LegacyAdapter:
    """
    Adapter to maintain compatibility with legacy DoorbellSecuritySystem API.
    
    This allows existing code (especially web interface) to continue working
    without modification while using the new pipeline architecture internally.
    """
    
    def __init__(self, orchestrator: PipelineOrchestrator):
        """Initialize legacy adapter."""
        self.orchestrator = orchestrator
        self.last_trigger_time = 0
        self.running = False
        
        # Emulate legacy attributes that web interface expects
        self.settings = self._create_settings_proxy()
        self.face_manager = self._create_face_manager_proxy()
        self.camera = self._create_camera_proxy()
        self.gpio = self._create_gpio_proxy()
        
        logger.info("Legacy adapter initialized")
    
    def start(self) -> None:
        """Start the system (delegates to orchestrator)."""
        if not self.orchestrator.running:
            self.orchestrator.start()
        self.running = True
    
    def stop(self) -> None:
        """Stop the system (delegates to orchestrator)."""
        self.running = False
        self.orchestrator.shutdown()
    
    def on_doorbell_pressed(self, channel=None) -> None:
        """Handle doorbell press (legacy interface)."""
        current_time = time.time()
        
        # Simple debounce
        if current_time - self.last_trigger_time < 5.0:
            logger.debug("Ignoring rapid doorbell press (debounce)")
            return
        
        self.last_trigger_time = current_time
        
        # Create pipeline event
        event = PipelineEvent(
            event_type=EventType.DOORBELL_PRESSED,
            data={
                "trigger_source": "gpio",
                "channel": channel,
                "timestamp": current_time
            },
            source="legacy_adapter"
        )
        
        # Publish to pipeline
        self.orchestrator.message_bus.publish('doorbell_events', event)
    
    def get_recent_captures(self, limit: int = 10) -> list:
        """Get recent capture events (legacy interface)."""
        try:
            # Query event database through orchestrator
            return self.orchestrator.event_database.get_recent_events(
                event_type='face_recognition',
                limit=limit
            )
        except Exception as e:
            logger.error(f"Failed to get recent captures: {e}")
            return []
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status (legacy interface)."""
        try:
            metrics = self.orchestrator.get_pipeline_metrics()
            
            return {
                "status": "running" if self.running else "stopped",
                "uptime": metrics.get('uptime', 0),
                "pipeline_health": metrics.get('pipeline_status', 'unknown'),
                "events_processed": metrics.get('events_per_minute', 0),
                "last_activity": self.last_trigger_time
            }
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _create_settings_proxy(self):
        """Create proxy object for settings access."""
        class SettingsProxy:
            def __init__(self, orchestrator):
                self.orchestrator = orchestrator
                # Delegate to orchestrator config
                config = orchestrator.config
                
                # Common legacy attributes
                self.DEBOUNCE_TIME = getattr(config, 'debounce_time', 5.0)
                self.CAPTURES_DIR = getattr(config, 'captures_dir', 'data/captures')
                self.CROPPED_FACES_DIR = getattr(config, 'cropped_faces_dir', 'data/cropped_faces')
                self.KNOWN_FACES_DIR = getattr(config, 'known_faces_dir', 'data/known_faces')
                self.BLACKLIST_FACES_DIR = getattr(config, 'blacklist_faces_dir', 'data/blacklist_faces')
        
        return SettingsProxy(self.orchestrator)
    
    def _create_face_manager_proxy(self):
        """Create proxy object for face manager access."""
        class FaceManagerProxy:
            def __init__(self, orchestrator):
                self.orchestrator = orchestrator
            
            def load_known_faces(self):
                """Delegate to pipeline face database."""
                # Pipeline handles this automatically
                pass
            
            def load_blacklist_faces(self):
                """Delegate to pipeline face database."""
                # Pipeline handles this automatically
                pass
            
            def get_known_faces_count(self) -> int:
                """Get count of known faces."""
                try:
                    return self.orchestrator.event_database.get_known_faces_count()
                except Exception:
                    return 0
        
        return FaceManagerProxy(self.orchestrator)
    
    def _create_camera_proxy(self):
        """Create proxy object for camera access."""
        class CameraProxy:
            def __init__(self, orchestrator):
                self.orchestrator = orchestrator
            
            def initialize(self):
                """Camera initialization handled by pipeline."""
                pass
            
            def capture_image(self):
                """Legacy capture interface."""
                # This would need to be implemented if legacy code calls it directly
                # For now, pipeline handles all captures
                pass
            
            def cleanup(self):
                """Camera cleanup handled by pipeline."""
                pass
        
        return CameraProxy(self.orchestrator)
    
    def _create_gpio_proxy(self):
        """Create proxy object for GPIO access."""
        class GPIOProxy:
            def __init__(self, orchestrator):
                self.orchestrator = orchestrator
            
            def setup_doorbell_button(self, callback):
                """GPIO setup handled by pipeline."""
                # Store callback for potential use
                self.callback = callback
            
            def simulate_doorbell_press(self):
                """Simulate doorbell press for testing."""
                # Delegate to orchestrator manager
                event = PipelineEvent(
                    event_type=EventType.DOORBELL_PRESSED,
                    data={"trigger_source": "simulation", "timestamp": time.time()},
                    source="gpio_simulation"
                )
                self.orchestrator.message_bus.publish('doorbell_events', event)
            
            def cleanup(self):
                """GPIO cleanup handled by pipeline."""
                pass
        
        return GPIOProxy(self.orchestrator)
```

### **Main Entry Point Implementation**
```python
#!/usr/bin/env python3
"""
Main Entry Point - Pipeline Architecture

New main entry point using the PipelineOrchestrator instead of legacy system.
"""

import sys
import signal
import logging
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.integration.orchestrator_manager import OrchestratorManager, SystemState
from config.logging_config import setup_logging

logger = logging.getLogger(__name__)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    if hasattr(signal_handler, 'manager'):
        signal_handler.manager.stop()
    sys.exit(0)


def main(config: Optional[dict] = None) -> int:
    """
    Main entry point for the doorbell security system.
    
    Args:
        config: Optional configuration override
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Setup logging
        setup_logging(level=logging.INFO)
        logger.info("Starting Doorbell Security System with Pipeline Architecture")
        
        # Create and start orchestrator manager
        manager = OrchestratorManager(config)
        signal_handler.manager = manager  # Store for signal handler
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start the system
        manager.start()
        
        # Monitor and keep running
        try:
            while manager.state in [SystemState.STARTING, SystemState.RUNNING]:
                health = manager.get_health_status()
                
                if health.state == SystemState.ERROR:
                    logger.error(f"System error detected: {health.last_error}")
                    break
                
                # Log periodic status
                if hasattr(main, 'last_status_time'):
                    if health.uptime - main.last_status_time > 300:  # Every 5 minutes
                        logger.info(f"System running - Uptime: {health.uptime:.1f}s, "
                                  f"Performance: {health.performance_score:.2f}")
                        main.last_status_time = health.uptime
                else:
                    main.last_status_time = health.uptime
                
                # Brief sleep
                import time
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        
        # Cleanup
        manager.stop()
        logger.info("System shutdown complete")
        return 0
        
    except Exception as e:
        logger.error(f"System startup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

## 🔄 **Migration Strategy**

### **Phase 1: Compatibility Layer**
1. **Create Legacy Adapter**: Implement `LegacyAdapter` with full API compatibility
2. **Update app.py**: Switch to orchestrator with legacy adapter
3. **Test Compatibility**: Verify all existing functionality works
4. **Performance Baseline**: Establish performance metrics

### **Phase 2: Web Interface Integration**
1. **Event Stream Connection**: Connect web interface to pipeline events
2. **API Modernization**: Update internal APIs to use pipeline directly
3. **Real-time Updates**: Implement real-time status updates
4. **Testing**: Comprehensive web interface testing

### **Phase 3: Performance Optimization**
1. **Pipeline Tuning**: Optimize worker counts and queue sizes
2. **Resource Management**: Fine-tune memory and CPU usage
3. **Monitoring Enhancement**: Add detailed performance monitoring
4. **Load Testing**: Validate performance under various loads

### **Phase 4: Production Readiness**
1. **Error Recovery**: Implement automatic recovery mechanisms
2. **Health Monitoring**: Add comprehensive health checking
3. **Deployment Scripts**: Create production deployment tools
4. **Documentation**: Update all documentation

## 🧪 **Testing Requirements**

### **Integration Tests**
```python
#!/usr/bin/env python3
"""
Orchestrator Integration Tests

Comprehensive testing of the pipeline orchestrator integration.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch

from src.integration.orchestrator_manager import OrchestratorManager, SystemState
from src.integration.legacy_adapter import LegacyAdapter
from src.pipeline.orchestrator import PipelineOrchestrator


class TestOrchestratorIntegration:
    """Test orchestrator integration functionality."""
    
    @pytest.fixture
    def orchestrator_manager(self):
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
        
        with patch('src.hardware.camera_handler.CameraHandler'), \
             patch('src.hardware.gpio_handler.GPIOHandler'):
            manager = OrchestratorManager(config)
            yield manager
            if manager.running:
                manager.stop()
    
    def test_orchestrator_startup_shutdown(self, orchestrator_manager):
        """Test orchestrator startup and shutdown."""
        # Test startup
        orchestrator_manager.start()
        assert orchestrator_manager.state == SystemState.RUNNING
        assert orchestrator_manager.running is True
        
        # Test health status
        health = orchestrator_manager.get_health_status()
        assert health.state == SystemState.RUNNING
        assert health.uptime > 0
        
        # Test shutdown
        orchestrator_manager.stop()
        assert orchestrator_manager.state == SystemState.STOPPED
        assert orchestrator_manager.running is False
    
    def test_legacy_compatibility(self, orchestrator_manager):
        """Test legacy adapter compatibility."""
        orchestrator_manager.start()
        
        # Get legacy interface
        legacy = orchestrator_manager.get_legacy_interface()
        assert isinstance(legacy, LegacyAdapter)
        
        # Test legacy methods
        assert hasattr(legacy, 'settings')
        assert hasattr(legacy, 'face_manager')
        assert hasattr(legacy, 'camera')
        assert hasattr(legacy, 'gpio')
        
        # Test doorbell trigger
        legacy.on_doorbell_pressed()
        
        # Test status
        status = legacy.get_system_status()
        assert status['status'] == 'running'
    
    def test_doorbell_event_flow(self, orchestrator_manager):
        """Test complete doorbell event flow."""
        orchestrator_manager.start()
        
        # Setup event tracking
        events_received = []
        
        def event_callback(data):
            events_received.append(data)
        
        orchestrator_manager.register_event_callback('doorbell_trigger', event_callback)
        
        # Trigger doorbell
        result = orchestrator_manager.trigger_doorbell()
        assert result['status'] == 'success'
        assert 'event_id' in result
        
        # Wait for processing
        time.sleep(2.0)
        
        # Verify events were processed
        # (This would need actual event verification in real implementation)
    
    def test_health_monitoring(self, orchestrator_manager):
        """Test health monitoring functionality."""
        orchestrator_manager.start()
        
        # Get initial health
        health = orchestrator_manager.get_health_status()
        assert health.state == SystemState.RUNNING
        assert health.performance_score > 0.5
        
        # Test monitoring over time
        time.sleep(2.0)
        
        health2 = orchestrator_manager.get_health_status()
        assert health2.uptime > health.uptime


class TestLegacyCompatibility:
    """Test legacy compatibility layer."""
    
    @pytest.fixture
    def legacy_adapter(self):
        """Create test legacy adapter."""
        with patch('src.pipeline.orchestrator.PipelineOrchestrator') as mock_orchestrator:
            mock_orchestrator.running = True
            mock_orchestrator.message_bus = Mock()
            mock_orchestrator.event_database = Mock()
            mock_orchestrator.config = Mock()
            
            adapter = LegacyAdapter(mock_orchestrator)
            return adapter
    
    def test_legacy_interface_structure(self, legacy_adapter):
        """Test that legacy interface has expected structure."""
        # Verify all expected attributes exist
        assert hasattr(legacy_adapter, 'settings')
        assert hasattr(legacy_adapter, 'face_manager')
        assert hasattr(legacy_adapter, 'camera')
        assert hasattr(legacy_adapter, 'gpio')
        
        # Test settings proxy
        assert hasattr(legacy_adapter.settings, 'DEBOUNCE_TIME')
        assert hasattr(legacy_adapter.settings, 'CAPTURES_DIR')
        
        # Test face manager proxy
        assert hasattr(legacy_adapter.face_manager, 'load_known_faces')
        assert hasattr(legacy_adapter.face_manager, 'get_known_faces_count')
        
        # Test GPIO proxy
        assert hasattr(legacy_adapter.gpio, 'simulate_doorbell_press')
    
    def test_doorbell_press_handling(self, legacy_adapter):
        """Test doorbell press handling through legacy interface."""
        # Test doorbell press
        legacy_adapter.on_doorbell_pressed(channel=18)
        
        # Verify message bus was called
        legacy_adapter.orchestrator.message_bus.publish.assert_called()
        
        # Test debounce
        legacy_adapter.on_doorbell_pressed(channel=18)  # Should be debounced
    
    def test_system_status_reporting(self, legacy_adapter):
        """Test system status reporting."""
        # Mock orchestrator metrics
        legacy_adapter.orchestrator.get_pipeline_metrics.return_value = {
            'uptime': 100.0,
            'pipeline_status': 'running',
            'events_per_minute': 5
        }
        
        status = legacy_adapter.get_system_status()
        
        assert status['status'] == 'running'
        assert status['uptime'] == 100.0
        assert status['events_processed'] == 5


class TestEndToEndPipeline:
    """End-to-end pipeline testing."""
    
    def test_complete_doorbell_workflow(self):
        """Test complete workflow from doorbell press to notification."""
        # This would be a comprehensive integration test
        # covering the entire pipeline from GPIO event to final notification
        pass
    
    def test_performance_under_load(self):
        """Test system performance under load."""
        # Load testing with multiple concurrent doorbell events
        pass
    
    def test_error_recovery(self):
        """Test system behavior during errors and recovery."""
        # Test various failure scenarios and recovery mechanisms
        pass
```

### **Performance Testing**
```python
#!/usr/bin/env python3
"""
Performance Tests for Pipeline Integration

Tests to validate performance improvements and resource usage.
"""

import pytest
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor

from src.integration.orchestrator_manager import OrchestratorManager


class TestPerformanceIntegration:
    """Performance testing for pipeline integration."""
    
    def test_startup_time(self):
        """Test system startup time."""
        start_time = time.time()
        
        with patch_hardware_components():
            manager = OrchestratorManager()
            manager.start()
            
            startup_time = time.time() - start_time
            manager.stop()
        
        # Should start in under 5 seconds
        assert startup_time < 5.0
    
    def test_memory_usage(self):
        """Test memory usage during operation."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        with patch_hardware_components():
            manager = OrchestratorManager()
            manager.start()
            
            # Run for a while to reach steady state
            time.sleep(10.0)
            
            steady_memory = process.memory_info().rss
            memory_increase = steady_memory - initial_memory
            
            manager.stop()
        
        # Memory increase should be reasonable (< 100MB)
        assert memory_increase < 100 * 1024 * 1024
    
    def test_concurrent_doorbell_events(self):
        """Test handling multiple concurrent doorbell events."""
        with patch_hardware_components():
            manager = OrchestratorManager()
            manager.start()
            
            # Trigger multiple events concurrently
            start_time = time.time()
            
            def trigger_doorbell():
                return manager.trigger_doorbell()
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(trigger_doorbell) for _ in range(20)]
                results = [f.result() for f in futures]
            
            processing_time = time.time() - start_time
            manager.stop()
        
        # All events should succeed
        assert all(r['status'] == 'success' for r in results)
        
        # Should process quickly
        assert processing_time < 5.0


@pytest.fixture
def patch_hardware_components():
    """Patch hardware components for testing."""
    with patch('src.hardware.camera_handler.CameraHandler'), \
         patch('src.hardware.gpio_handler.GPIOHandler'), \
         patch('src.telegram_notifier.TelegramNotifier'):
        yield
```

## 📊 **Performance Metrics**

### **Target Performance Improvements**
- **Startup Time**: < 3 seconds (vs 5-8 seconds legacy)
- **Memory Usage**: < 150MB steady state (vs 200MB+ legacy)
- **Processing Latency**: < 2 seconds doorbell-to-notification (vs 3-5 seconds legacy)
- **Throughput**: > 10 concurrent doorbell events/minute
- **CPU Efficiency**: 25% improvement in CPU utilization
- **Error Recovery**: < 10 seconds automatic recovery time

### **Monitoring and Alerting**
- Real-time performance dashboards
- Automatic alerts for performance degradation
- Health check endpoints for external monitoring
- Comprehensive logging and debugging

## 🚀 **Deployment Strategy**

### **Blue-Green Deployment**
1. **Blue Environment**: Current legacy system
2. **Green Environment**: New pipeline system
3. **Traffic Routing**: Gradual migration of traffic
4. **Rollback Plan**: Immediate rollback capability

### **Configuration Management**
1. **Environment-Specific Configs**: Development, staging, production
2. **Feature Flags**: Enable/disable new features
3. **Hot Reloading**: Update configuration without restart
4. **Validation**: Comprehensive configuration validation

## 📋 **Acceptance Criteria**

### **Functional Requirements**
- [ ] Complete replacement of legacy `DoorbellSecuritySystem`
- [ ] All existing web interface functionality preserved
- [ ] Backward compatibility for external integrations
- [ ] Improved error handling and recovery
- [ ] Comprehensive monitoring and health checking

### **Performance Requirements**
- [ ] 25% improvement in processing speed
- [ ] 30% reduction in memory usage
- [ ] Sub-3-second startup time
- [ ] 99% uptime with automatic recovery

### **Quality Requirements**
- [ ] 95% test coverage for integration layer
- [ ] Complete performance test suite
- [ ] Production deployment documentation
- [ ] Monitoring and alerting setup

### **Documentation Requirements**
- [ ] Migration guide from legacy system
- [ ] Performance tuning guide
- [ ] Troubleshooting guide
- [ ] API documentation updates

---

**This issue represents the final major architectural transformation, completing the migration from a traditional monolithic design to a modern, high-performance pipeline architecture inspired by Frigate NVR. The implementation provides a robust foundation for future enhancements while maintaining full backward compatibility.**