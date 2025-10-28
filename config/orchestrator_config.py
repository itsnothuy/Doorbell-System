#!/usr/bin/env python3
"""
Orchestrator Configuration - High-Level System Configuration

Configuration specific to the OrchestratorManager, extending pipeline configuration
with management and monitoring features.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from config.pipeline_config import PipelineConfig


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator manager."""
    
    # Health monitoring
    health_check_interval: float = 30.0
    metrics_collection_interval: float = 60.0
    
    # Auto-recovery settings
    auto_recovery_enabled: bool = True
    max_restart_attempts: int = 3
    restart_cooldown_seconds: float = 60.0
    
    # Performance thresholds
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    queue_backlog_threshold: int = 100
    
    # Alerting
    alert_on_high_cpu: bool = True
    alert_on_high_memory: bool = True
    alert_on_errors: bool = True
    error_threshold_per_minute: int = 10
    
    # Pipeline configuration
    pipeline_config: Optional[Dict[str, Any]] = None
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize orchestrator configuration."""
        # Set defaults
        self.health_check_interval = 30.0
        self.metrics_collection_interval = 60.0
        self.auto_recovery_enabled = True
        self.max_restart_attempts = 3
        self.restart_cooldown_seconds = 60.0
        self.cpu_threshold = 80.0
        self.memory_threshold = 85.0
        self.queue_backlog_threshold = 100
        self.alert_on_high_cpu = True
        self.alert_on_high_memory = True
        self.alert_on_errors = True
        self.error_threshold_per_minute = 10
        
        # Initialize pipeline config
        pipeline_config_dict = {}
        if config_dict:
            # Extract orchestrator-specific settings
            for key, value in config_dict.items():
                if hasattr(self, key) and key != 'pipeline_config':
                    setattr(self, key, value)
            
            # Extract pipeline configuration
            if 'pipeline_config' in config_dict:
                pipeline_config_dict = config_dict['pipeline_config']
        
        # Create pipeline configuration
        self.pipeline_config = PipelineConfig(pipeline_config_dict)
        
        # Load from environment
        self._load_from_environment()
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        # Health monitoring
        if os.getenv('HEALTH_CHECK_INTERVAL'):
            self.health_check_interval = float(os.getenv('HEALTH_CHECK_INTERVAL'))
        
        # Auto-recovery
        if os.getenv('AUTO_RECOVERY_ENABLED'):
            self.auto_recovery_enabled = os.getenv('AUTO_RECOVERY_ENABLED').lower() == 'true'
        
        if os.getenv('MAX_RESTART_ATTEMPTS'):
            self.max_restart_attempts = int(os.getenv('MAX_RESTART_ATTEMPTS'))
        
        # Performance thresholds
        if os.getenv('CPU_THRESHOLD'):
            self.cpu_threshold = float(os.getenv('CPU_THRESHOLD'))
        
        if os.getenv('MEMORY_THRESHOLD'):
            self.memory_threshold = float(os.getenv('MEMORY_THRESHOLD'))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'health_check_interval': self.health_check_interval,
            'metrics_collection_interval': self.metrics_collection_interval,
            'auto_recovery_enabled': self.auto_recovery_enabled,
            'max_restart_attempts': self.max_restart_attempts,
            'restart_cooldown_seconds': self.restart_cooldown_seconds,
            'cpu_threshold': self.cpu_threshold,
            'memory_threshold': self.memory_threshold,
            'queue_backlog_threshold': self.queue_backlog_threshold,
            'alert_on_high_cpu': self.alert_on_high_cpu,
            'alert_on_high_memory': self.alert_on_high_memory,
            'alert_on_errors': self.alert_on_errors,
            'error_threshold_per_minute': self.error_threshold_per_minute,
            'pipeline_config': self.pipeline_config.to_dict()
        }
