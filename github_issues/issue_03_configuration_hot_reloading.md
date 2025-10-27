# Issue #3: Pipeline Configuration Hot-Reloading and Advanced Management

## 🎯 **Overview**
Enhance the existing Pipeline Configuration System with hot-reloading capabilities, environment-specific configuration management, advanced validation, and comprehensive testing to create a production-ready configuration management system.

## 📋 **Acceptance Criteria**

### Core Hot-Reloading Features
- [ ] **Configuration Hot-Reloading** (`config/hot_reload.py`)
  - [ ] File system watcher for configuration changes
  - [ ] Thread-safe configuration updates
  - [ ] Component notification system for config changes
  - [ ] Rollback mechanism for invalid configurations
  - [ ] Configuration change event broadcasting
  - [ ] Graceful component reconfiguration

- [ ] **Configuration Validation Enhanced** (`config/validation.py`)
  - [ ] Schema-based configuration validation
  - [ ] Cross-component dependency validation
  - [ ] Runtime configuration compatibility checks
  - [ ] Configuration migration support
  - [ ] Validation error reporting with suggestions
  - [ ] Configuration health checks

- [ ] **Environment Management** (`config/environment.py`)
  - [ ] Multi-environment configuration support (dev/staging/prod)
  - [ ] Environment-specific configuration overrides
  - [ ] Secure environment variable integration
  - [ ] Configuration template system
  - [ ] Environment validation and safety checks

### Advanced Configuration Features
- [ ] **Configuration API** (`config/api.py`)
  - [ ] REST API for configuration management
  - [ ] Real-time configuration monitoring
  - [ ] Configuration backup and restore
  - [ ] Configuration diff and comparison tools
  - [ ] Remote configuration management
  - [ ] Configuration audit logging

- [ ] **Dynamic Configuration** (`config/dynamic.py`)
  - [ ] Runtime configuration parameter adjustment
  - [ ] A/B testing configuration support
  - [ ] Performance-based configuration optimization
  - [ ] Load-based configuration scaling
  - [ ] Machine learning-driven configuration tuning

- [ ] **Configuration Security** (`config/security.py`)
  - [ ] Encrypted configuration storage
  - [ ] Configuration access control
  - [ ] Secure credential management
  - [ ] Configuration integrity verification
  - [ ] Audit trail for configuration changes

### Comprehensive Testing
- [ ] **Configuration Tests** (`tests/test_pipeline_config.py`)
  - [ ] Configuration loading and validation tests
  - [ ] Hot-reloading functionality tests
  - [ ] Environment-specific configuration tests
  - [ ] Configuration migration tests
  - [ ] Error handling and recovery tests

- [ ] **Integration Tests** (`tests/test_config_integration.py`)
  - [ ] End-to-end configuration management tests
  - [ ] Component reconfiguration tests
  - [ ] Multi-environment deployment tests
  - [ ] Configuration API tests
  - [ ] Performance impact tests

## 🔧 **Technical Implementation**

### Hot-Reloading System
```python
#!/usr/bin/env python3
"""
Configuration Hot-Reloading System

Provides real-time configuration updates without system restart.
"""

import os
import time
import threading
import logging
from typing import Dict, Any, Callable, Optional, Set
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from config.settings import load_configuration
from config.validation import ConfigValidator
from src.communication.message_bus import MessageBus
from src.communication.events import ConfigurationEvent, EventType


class ConfigurationReloader:
    """Manages hot-reloading of configuration files."""
    
    def __init__(self, message_bus: MessageBus, config_paths: List[Path]):
        self.message_bus = message_bus
        self.config_paths = config_paths
        self.validator = ConfigValidator()
        
        # Current configuration state
        self.current_config = self._load_initial_config()
        self.config_version = 1
        
        # File watching
        self.observer = Observer()
        self.event_handler = ConfigFileHandler(self)
        
        # Reload state
        self.reload_lock = threading.RLock()
        self.reload_callbacks: Dict[str, Callable[[Dict[str, Any]], None]] = {}
        
        # Statistics
        self.reload_count = 0
        self.last_reload_time = time.time()
        self.failed_reloads = 0
        
    def start_watching(self) -> None:
        """Start watching configuration files for changes."""
        for config_path in self.config_paths:
            self.observer.schedule(
                self.event_handler,
                str(config_path.parent),
                recursive=False
            )
        
        self.observer.start()
        logger.info("Configuration hot-reloading started")
    
    def stop_watching(self) -> None:
        """Stop watching configuration files."""
        self.observer.stop()
        self.observer.join()
        logger.info("Configuration hot-reloading stopped")
    
    def reload_configuration(self, force: bool = False) -> bool:
        """Reload configuration from files."""
        with self.reload_lock:
            try:
                # Load new configuration
                new_config = self._load_configuration()
                
                # Validate new configuration
                validation_result = self.validator.validate_full_config(new_config)
                if not validation_result.is_valid:
                    logger.error(f"Configuration validation failed: {validation_result.errors}")
                    self.failed_reloads += 1
                    return False
                
                # Check if configuration actually changed
                if not force and self._configs_equal(self.current_config, new_config):
                    logger.debug("Configuration unchanged, skipping reload")
                    return True
                
                # Create configuration diff
                config_diff = self._create_config_diff(self.current_config, new_config)
                
                # Backup current configuration
                backup_config = self.current_config.copy()
                
                try:
                    # Update configuration
                    self.current_config = new_config
                    self.config_version += 1
                    self.last_reload_time = time.time()
                    self.reload_count += 1
                    
                    # Notify components of configuration change
                    self._notify_config_change(config_diff)
                    
                    # Broadcast configuration change event
                    config_event = ConfigurationEvent(
                        event_type=EventType.CONFIGURATION_RELOADED,
                        config_diff=config_diff,
                        version=self.config_version,
                        timestamp=time.time()
                    )
                    self.message_bus.publish('config_events', config_event)
                    
                    logger.info(f"Configuration reloaded successfully (version {self.config_version})")
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
        """Register callback for configuration changes."""
        self.reload_callbacks[component_name] = callback
        logger.debug(f"Registered reload callback for {component_name}")
    
    def _notify_config_change(self, config_diff: Dict[str, Any]) -> None:
        """Notify registered components of configuration changes."""
        for component_name, callback in self.reload_callbacks.items():
            try:
                callback(config_diff)
                logger.debug(f"Notified {component_name} of configuration change")
            except Exception as e:
                logger.error(f"Failed to notify {component_name}: {e}")


class ConfigFileHandler(FileSystemEventHandler):
    """Handles file system events for configuration files."""
    
    def __init__(self, reloader: ConfigurationReloader):
        self.reloader = reloader
        self.debounce_timer = None
        self.debounce_delay = 1.0  # 1 second debounce
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        # Check if modified file is a configuration file
        modified_path = Path(event.src_path)
        if any(modified_path.samefile(config_path) for config_path in self.reloader.config_paths):
            self._debounced_reload()
    
    def _debounced_reload(self):
        """Debounced configuration reload to handle rapid file changes."""
        if self.debounce_timer:
            self.debounce_timer.cancel()
        
        self.debounce_timer = threading.Timer(
            self.debounce_delay,
            self.reloader.reload_configuration
        )
        self.debounce_timer.start()
```

### Advanced Configuration Validation
```python
#!/usr/bin/env python3
"""
Advanced Configuration Validation System

Comprehensive validation with schema support and dependency checking.
"""

import jsonschema
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from config.pipeline_config import PipelineConfig


@dataclass
class ValidationError:
    """Represents a configuration validation error."""
    path: str
    message: str
    severity: str  # 'error', 'warning', 'info'
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


class ConfigValidator:
    """Advanced configuration validator with schema support."""
    
    def __init__(self, schema_path: Optional[Path] = None):
        self.schema_path = schema_path or Path(__file__).parent / "schemas"
        self.schemas = self._load_schemas()
        
        # Dependency rules
        self.dependency_rules = self._load_dependency_rules()
    
    def validate_full_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Perform comprehensive configuration validation."""
        errors = []
        warnings = []
        
        # Schema validation
        schema_errors = self._validate_schema(config)
        errors.extend(schema_errors)
        
        # Dependency validation
        dep_errors, dep_warnings = self._validate_dependencies(config)
        errors.extend(dep_errors)
        warnings.extend(dep_warnings)
        
        # Hardware compatibility validation
        hw_errors, hw_warnings = self._validate_hardware_compatibility(config)
        errors.extend(hw_errors)
        warnings.extend(hw_warnings)
        
        # Performance validation
        perf_warnings = self._validate_performance_settings(config)
        warnings.extend(perf_warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_schema(self, config: Dict[str, Any]) -> List[ValidationError]:
        """Validate configuration against JSON schema."""
        errors = []
        
        try:
            # Load main schema
            main_schema = self.schemas.get('main_config')
            if main_schema:
                jsonschema.validate(config, main_schema)
        except jsonschema.ValidationError as e:
            errors.append(ValidationError(
                path=e.absolute_path,
                message=e.message,
                severity='error',
                suggestion=self._get_schema_suggestion(e)
            ))
        
        return errors
    
    def _validate_dependencies(
        self,
        config: Dict[str, Any]
    ) -> Tuple[List[ValidationError], List[ValidationError]]:
        """Validate configuration dependencies."""
        errors = []
        warnings = []
        
        # Example: GPU detector requires CUDA configuration
        if config.get('detectors', {}).get('gpu', {}).get('enabled'):
            if not config.get('hardware', {}).get('cuda', {}).get('available'):
                errors.append(ValidationError(
                    path='detectors.gpu.enabled',
                    message='GPU detector enabled but CUDA not available',
                    severity='error',
                    suggestion='Install CUDA or disable GPU detector'
                ))
        
        # Example: High FPS requires sufficient processing power
        fps = config.get('frame_capture', {}).get('fps', 10)
        if fps > 30:
            warnings.append(ValidationError(
                path='frame_capture.fps',
                message=f'High FPS ({fps}) may impact performance',
                severity='warning',
                suggestion='Consider reducing FPS for better stability'
            ))
        
        return errors, warnings
```

### Environment Management System
```python
#!/usr/bin/env python3
"""
Environment-Specific Configuration Management

Handles multiple environments with secure overrides and validation.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from enum import Enum

from config.settings import BaseConfig


class Environment(Enum):
    """Supported deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class EnvironmentManager:
    """Manages environment-specific configurations."""
    
    def __init__(self, base_config_path: Path):
        self.base_config_path = base_config_path
        self.current_env = self._detect_environment()
        
    def load_environment_config(self) -> Dict[str, Any]:
        """Load configuration for current environment."""
        # Load base configuration
        base_config = self._load_base_config()
        
        # Load environment-specific overrides
        env_overrides = self._load_environment_overrides()
        
        # Merge configurations
        merged_config = self._merge_configs(base_config, env_overrides)
        
        # Apply environment variables
        final_config = self._apply_env_variables(merged_config)
        
        return final_config
    
    def _detect_environment(self) -> Environment:
        """Auto-detect current environment."""
        env_name = os.getenv('DOORBELL_ENV', 'development').lower()
        
        try:
            return Environment(env_name)
        except ValueError:
            logger.warning(f"Unknown environment '{env_name}', defaulting to development")
            return Environment.DEVELOPMENT
    
    def _load_environment_overrides(self) -> Dict[str, Any]:
        """Load environment-specific configuration overrides."""
        env_config_path = self.base_config_path.parent / f"env_{self.current_env.value}.py"
        
        if env_config_path.exists():
            # Load environment-specific configuration
            return self._load_config_from_file(env_config_path)
        
        return {}
    
    def _apply_env_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        # Define environment variable mappings
        env_mappings = {
            'DOORBELL_LOG_LEVEL': 'logging.level',
            'DOORBELL_DETECTOR_TYPE': 'detectors.primary_type',
            'DOORBELL_CAMERA_FPS': 'frame_capture.fps',
            'DOORBELL_WEB_PORT': 'web_interface.port',
            # Add more mappings as needed
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                self._set_nested_config(config, config_path, env_value)
        
        return config
```

## 🧪 **Comprehensive Testing Strategy**

### Hot-Reloading Tests
```python
def test_configuration_hot_reload():
    """Test hot-reloading functionality."""
    # Setup
    reloader = ConfigurationReloader(message_bus, config_paths)
    reloader.start_watching()
    
    # Modify configuration file
    modify_config_file('frame_capture.fps', 15)
    
    # Wait for reload
    time.sleep(2)
    
    # Verify configuration updated
    assert reloader.current_config['frame_capture']['fps'] == 15
    
    reloader.stop_watching()
```

### Environment Testing
```python
@pytest.mark.parametrize("environment", [
    Environment.DEVELOPMENT,
    Environment.STAGING,
    Environment.PRODUCTION
])
def test_environment_specific_config(environment):
    """Test environment-specific configuration loading."""
    with mock.patch.dict(os.environ, {'DOORBELL_ENV': environment.value}):
        env_manager = EnvironmentManager(config_path)
        config = env_manager.load_environment_config()
        
        # Verify environment-specific settings
        assert config['environment'] == environment.value
        verify_environment_constraints(config, environment)
```

## 📊 **Performance Targets**

### Hot-Reloading Targets
- **Reload Detection Time**: <500ms from file change
- **Configuration Application**: <1 second for component updates
- **Memory Overhead**: <10MB for monitoring
- **Validation Time**: <100ms for full configuration

## 📁 **File Structure**
```
config/
├── hot_reload.py            # Hot-reloading system
├── validation.py            # Advanced validation
├── environment.py           # Environment management
├── api.py                  # Configuration API
├── dynamic.py              # Dynamic configuration
├── security.py             # Configuration security
├── schemas/                # JSON schemas
│   ├── main_config.json
│   ├── detector_config.json
│   └── pipeline_config.json
└── environments/           # Environment configs
    ├── env_development.py
    ├── env_staging.py
    └── env_production.py

tests/
├── test_pipeline_config.py           # Configuration tests
├── test_hot_reload.py                # Hot-reload tests
├── test_config_validation.py         # Validation tests
├── test_environment_management.py    # Environment tests
└── test_config_integration.py        # Integration tests
```

## ⚡ **Implementation Timeline**
- **Phase 1** (Days 1-3): Hot-Reloading System
- **Phase 2** (Days 4-6): Advanced Validation
- **Phase 3** (Days 7-9): Environment Management
- **Phase 4** (Days 10-12): Configuration API & Security
- **Phase 5** (Days 13-14): Testing & Documentation

## 🎯 **Definition of Done**
- [ ] Hot-reloading functionality operational
- [ ] Advanced validation with comprehensive error reporting
- [ ] Multi-environment support complete
- [ ] Configuration API functional
- [ ] Security features implemented
- [ ] All tests pass with >95% coverage
- [ ] Performance targets met
- [ ] Documentation updated

## 🔗 **Related Issues**
- Enhances: Pipeline Configuration System (Already Complete)
- Integrates with: All pipeline components
- Enables: Production deployment flexibility

## 📚 **References**
- [Configuration Management Best Practices](docs/IMPLEMENTATION_GUIDE.md)
- [Security Guidelines](docs/SECURITY.md)
- [Environment Setup Guide](docs/installation.md)