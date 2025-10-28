#!/usr/bin/env python3
"""
Configuration Hot-Reloading System

Provides real-time configuration updates without system restart.
"""

import os
import time
import threading
import logging
from typing import Dict, Any, Callable, Optional, List
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None

from config.pipeline_config import PipelineConfig
from config.validation import ConfigValidator, ValidationResult

logger = logging.getLogger(__name__)


class ConfigFileHandler:
    """Handles file system events for configuration files."""
    
    def __init__(self, reloader: 'ConfigurationReloader'):
        self.reloader = reloader
        self.debounce_timer: Optional[threading.Timer] = None
        self.debounce_delay = 1.0  # 1 second debounce
    
    def on_modified(self, event: Any) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return
        
        # Check if modified file is a configuration file
        modified_path = Path(event.src_path)
        if any(
            modified_path.samefile(config_path) 
            for config_path in self.reloader.config_paths
            if config_path.exists()
        ):
            logger.debug(f"Configuration file modified: {modified_path}")
            self._debounced_reload()
    
    def _debounced_reload(self) -> None:
        """Debounced configuration reload to handle rapid file changes."""
        if self.debounce_timer:
            self.debounce_timer.cancel()
        
        self.debounce_timer = threading.Timer(
            self.debounce_delay,
            self.reloader.reload_configuration
        )
        self.debounce_timer.start()


class ConfigurationReloader:
    """Manages hot-reloading of configuration files."""
    
    def __init__(
        self,
        config_paths: List[Path],
        message_bus: Optional[Any] = None,
        validator: Optional[ConfigValidator] = None
    ):
        """Initialize configuration reloader.
        
        Args:
            config_paths: List of configuration file paths to watch
            message_bus: Optional message bus for broadcasting events
            validator: Optional configuration validator
        """
        self.config_paths = config_paths
        self.message_bus = message_bus
        self.validator = validator or ConfigValidator()
        
        # Current configuration state
        self.current_config: Optional[PipelineConfig] = None
        self.config_version = 0
        
        # File watching
        self.observer: Optional[Any] = None
        self.event_handler: Optional[ConfigFileHandler] = None
        self._watching = False
        
        # Reload state
        self.reload_lock = threading.RLock()
        self.reload_callbacks: Dict[str, Callable[[Dict[str, Any]], None]] = {}
        
        # Statistics
        self.reload_count = 0
        self.last_reload_time: Optional[float] = None
        self.failed_reloads = 0
        
        # Load initial configuration
        self._load_initial_config()
    
    def _load_initial_config(self) -> None:
        """Load initial configuration."""
        try:
            self.current_config = PipelineConfig()
            self.config_version = 1
            self.last_reload_time = time.time()
            logger.info("Initial configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load initial configuration: {e}")
            raise
    
    def start_watching(self) -> bool:
        """Start watching configuration files for changes.
        
        Returns:
            bool: True if watching started successfully, False otherwise
        """
        if not WATCHDOG_AVAILABLE:
            logger.warning("Watchdog not available, hot-reloading disabled")
            return False
        
        if self._watching:
            logger.warning("Configuration watching already started")
            return False
        
        try:
            self.observer = Observer()
            self.event_handler = ConfigFileHandler(self)
            
            # Watch parent directories of config files
            watched_dirs = set()
            for config_path in self.config_paths:
                if config_path.exists():
                    parent_dir = config_path.parent
                    if parent_dir not in watched_dirs:
                        self.observer.schedule(
                            self.event_handler,
                            str(parent_dir),
                            recursive=False
                        )
                        watched_dirs.add(parent_dir)
                        logger.debug(f"Watching directory: {parent_dir}")
            
            self.observer.start()
            self._watching = True
            logger.info(f"Configuration hot-reloading started (watching {len(watched_dirs)} directories)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start configuration watching: {e}")
            return False
    
    def stop_watching(self) -> None:
        """Stop watching configuration files."""
        if not self._watching:
            return
        
        try:
            if self.observer:
                self.observer.stop()
                self.observer.join(timeout=5.0)
            self._watching = False
            logger.info("Configuration hot-reloading stopped")
        except Exception as e:
            logger.error(f"Error stopping configuration watching: {e}")
    
    def reload_configuration(self, force: bool = False) -> bool:
        """Reload configuration from files.
        
        Args:
            force: Force reload even if configuration hasn't changed
            
        Returns:
            bool: True if reload was successful, False otherwise
        """
        with self.reload_lock:
            try:
                logger.debug("Reloading configuration...")
                
                # Load new configuration
                new_config = PipelineConfig()
                
                # Convert to dictionaries for comparison
                new_config_dict = new_config.to_dict()
                
                # Validate new configuration
                validation_result = self.validator.validate_full_config(new_config_dict)
                if not validation_result.is_valid:
                    logger.error(f"Configuration validation failed: {validation_result.errors}")
                    self.failed_reloads += 1
                    return False
                
                # Log warnings if any
                if validation_result.has_warnings:
                    for warning in validation_result.warnings:
                        logger.warning(f"Configuration warning: {warning.message}")
                
                # Check if configuration actually changed
                if not force and self.current_config:
                    current_config_dict = self.current_config.to_dict()
                    if self._configs_equal(current_config_dict, new_config_dict):
                        logger.debug("Configuration unchanged, skipping reload")
                        return True
                
                # Create configuration diff
                config_diff = self._create_config_diff(
                    self.current_config.to_dict() if self.current_config else {},
                    new_config_dict
                )
                
                # Backup current configuration
                backup_config = self.current_config
                
                try:
                    # Update configuration
                    self.current_config = new_config
                    self.config_version += 1
                    self.last_reload_time = time.time()
                    self.reload_count += 1
                    
                    # Notify components of configuration change
                    self._notify_config_change(config_diff)
                    
                    # Broadcast configuration change event via message bus
                    if self.message_bus:
                        self._broadcast_config_event(config_diff)
                    
                    logger.info(
                        f"Configuration reloaded successfully "
                        f"(version {self.config_version}, "
                        f"{len(config_diff)} changes)"
                    )
                    return True
                    
                except Exception as e:
                    # Rollback on failure
                    logger.error(f"Configuration reload failed, rolling back: {e}")
                    self.current_config = backup_config
                    self.failed_reloads += 1
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to reload configuration: {e}")
                self.failed_reloads += 1
                return False
    
    def register_reload_callback(
        self,
        component_name: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Register callback for configuration changes.
        
        Args:
            component_name: Name of the component registering the callback
            callback: Callable that receives configuration diff
        """
        self.reload_callbacks[component_name] = callback
        logger.debug(f"Registered reload callback for {component_name}")
    
    def unregister_reload_callback(self, component_name: str) -> None:
        """Unregister callback for configuration changes.
        
        Args:
            component_name: Name of the component to unregister
        """
        if component_name in self.reload_callbacks:
            del self.reload_callbacks[component_name]
            logger.debug(f"Unregistered reload callback for {component_name}")
    
    def get_config(self) -> Optional[PipelineConfig]:
        """Get current configuration.
        
        Returns:
            Optional[PipelineConfig]: Current configuration or None
        """
        return self.current_config
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get hot-reloading statistics.
        
        Returns:
            Dict[str, Any]: Statistics about reloading operations
        """
        return {
            'config_version': self.config_version,
            'reload_count': self.reload_count,
            'failed_reloads': self.failed_reloads,
            'last_reload_time': self.last_reload_time,
            'watching': self._watching,
            'registered_callbacks': len(self.reload_callbacks),
            'success_rate': (
                self.reload_count / (self.reload_count + self.failed_reloads)
                if (self.reload_count + self.failed_reloads) > 0
                else 0.0
            )
        }
    
    def _notify_config_change(self, config_diff: Dict[str, Any]) -> None:
        """Notify registered components of configuration changes.
        
        Args:
            config_diff: Dictionary describing configuration changes
        """
        for component_name, callback in self.reload_callbacks.items():
            try:
                callback(config_diff)
                logger.debug(f"Notified {component_name} of configuration change")
            except Exception as e:
                logger.error(f"Failed to notify {component_name}: {e}")
    
    def _broadcast_config_event(self, config_diff: Dict[str, Any]) -> None:
        """Broadcast configuration change event via message bus.
        
        Args:
            config_diff: Dictionary describing configuration changes
        """
        try:
            # Import here to avoid circular dependency
            from src.communication.events import EventType
            
            config_event = {
                'event_type': EventType.CONFIGURATION_RELOADED,
                'config_diff': config_diff,
                'version': self.config_version,
                'timestamp': time.time()
            }
            
            self.message_bus.publish('config_events', config_event)
            logger.debug("Configuration change event broadcasted")
        except Exception as e:
            logger.warning(f"Failed to broadcast configuration event: {e}")
    
    def _configs_equal(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any]
    ) -> bool:
        """Compare two configurations for equality.
        
        Args:
            config1: First configuration dictionary
            config2: Second configuration dictionary
            
        Returns:
            bool: True if configurations are equal
        """
        return self._deep_equal(config1, config2)
    
    def _deep_equal(self, obj1: Any, obj2: Any) -> bool:
        """Deep comparison of two objects.
        
        Args:
            obj1: First object
            obj2: Second object
            
        Returns:
            bool: True if objects are deeply equal
        """
        if type(obj1) != type(obj2):
            return False
        
        if isinstance(obj1, dict):
            if set(obj1.keys()) != set(obj2.keys()):
                return False
            return all(self._deep_equal(obj1[k], obj2[k]) for k in obj1.keys())
        
        if isinstance(obj1, (list, tuple)):
            if len(obj1) != len(obj2):
                return False
            return all(self._deep_equal(a, b) for a, b in zip(obj1, obj2))
        
        return obj1 == obj2
    
    def _create_config_diff(
        self,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a diff between two configurations.
        
        Args:
            old_config: Old configuration dictionary
            new_config: New configuration dictionary
            
        Returns:
            Dict[str, Any]: Dictionary describing changes
        """
        diff: Dict[str, Any] = {
            'added': {},
            'removed': {},
            'modified': {},
            'unchanged': []
        }
        
        # Find added and modified keys
        for section, values in new_config.items():
            if section not in old_config:
                diff['added'][section] = values
            elif not self._deep_equal(old_config[section], values):
                diff['modified'][section] = {
                    'old': old_config[section],
                    'new': values
                }
            else:
                diff['unchanged'].append(section)
        
        # Find removed keys
        for section in old_config:
            if section not in new_config:
                diff['removed'][section] = old_config[section]
        
        return diff
    
    def __enter__(self) -> 'ConfigurationReloader':
        """Context manager entry."""
        self.start_watching()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop_watching()


def create_reloader(
    config_path: Optional[Path] = None,
    message_bus: Optional[Any] = None
) -> ConfigurationReloader:
    """Create a configuration reloader instance.
    
    Args:
        config_path: Optional specific configuration file path
        message_bus: Optional message bus for event broadcasting
        
    Returns:
        ConfigurationReloader: Configured reloader instance
    """
    if config_path is None:
        config_path = Path(__file__).parent / "pipeline_config.py"
    
    config_paths = [config_path]
    
    return ConfigurationReloader(
        config_paths=config_paths,
        message_bus=message_bus
    )
