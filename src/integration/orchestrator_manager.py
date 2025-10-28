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
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.pipeline.orchestrator import PipelineOrchestrator
from src.communication.message_bus import MessageBus
from src.communication.events import PipelineEvent, EventType

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
        # Load configuration
        from config.orchestrator_config import OrchestratorConfig
        self.config = OrchestratorConfig(config or {})
        
        # Initialize orchestrator
        self.orchestrator = PipelineOrchestrator(self.config.pipeline_config)
        
        # Legacy adapter will be created when needed
        self._legacy_adapter = None
        
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
            self.orchestrator.start_time = time.time()
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
        if self._legacy_adapter is None:
            from src.integration.legacy_adapter import LegacyAdapter
            self._legacy_adapter = LegacyAdapter(self.orchestrator)
        return self._legacy_adapter
    
    def _start_monitoring(self) -> None:
        """Start background monitoring thread."""
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="OrchestratorMonitor"
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
        try:
            self.orchestrator.message_bus.subscribe(
                'system_errors', self._handle_system_event, 'orchestrator_manager'
            )
            self.orchestrator.message_bus.subscribe(
                'pipeline_health', self._handle_health_event, 'orchestrator_manager'
            )
        except Exception as e:
            logger.warning(f"Failed to setup event handlers: {e}")
    
    def _handle_system_event(self, message) -> None:
        """Handle system-level events."""
        try:
            event_data = message.data if hasattr(message, 'data') else {}
            
            if event_data.get('type') == 'error':
                self.error_count += 1
                self.last_error = event_data.get('message', 'Unknown error')
                
                # Notify callbacks
                self._notify_callbacks('system_error', event_data)
        except Exception as e:
            logger.error(f"Error handling system event: {e}")
    
    def _handle_health_event(self, message) -> None:
        """Handle health monitoring events."""
        try:
            # Notify callbacks
            event_data = message.data if hasattr(message, 'data') else {}
            self._notify_callbacks('health_update', event_data)
        except Exception as e:
            logger.error(f"Error handling health event: {e}")
    
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
            if isinstance(queue_status, dict) and queue_status.get('backlog_warning'):
                score *= 0.8
            
            # Check worker health
            worker_status = pipeline_status.get('worker_status', {})
            if isinstance(worker_status, dict) and worker_status.get('high_cpu'):
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
