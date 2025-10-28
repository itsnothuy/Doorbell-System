# Configuration Management System

## Overview

The Doorbell Security System features a sophisticated configuration management system inspired by modern DevOps practices. It provides hot-reloading, comprehensive validation, and environment-specific configuration management without requiring system restarts.

## Features

### 🔄 Hot-Reloading
- **Zero-Downtime Updates**: Change configuration without restarting the system
- **File Watching**: Automatically detects configuration file changes
- **Rollback Protection**: Invalid configurations are rejected and rolled back
- **Component Notifications**: Registered components are notified of configuration changes
- **Thread-Safe**: Safe for concurrent access in multi-threaded applications

### ✅ Advanced Validation
- **Schema Validation**: JSON Schema-based configuration validation
- **Dependency Checking**: Validates cross-component dependencies (e.g., GPU detector requires CUDA)
- **Performance Warnings**: Alerts on potentially problematic settings
- **Resource Validation**: Ensures reasonable resource allocation
- **Helpful Suggestions**: Provides actionable suggestions for fixing errors

### 🌍 Environment Management
- **Multi-Environment Support**: Development, Testing, Staging, Production
- **Auto-Detection**: Automatically detects environment from variables
- **Environment-Specific Overrides**: Different settings per environment
- **Secure Variable Integration**: Type-safe environment variable loading

### 📊 Performance
- **Fast Reload**: Sub-second configuration reload times
- **Efficient Validation**: 100 validations in under 1 second
- **Low Memory**: Minimal overhead suitable for edge devices
- **Concurrent Safe**: Thread-safe operations with proper locking

## Installation

The configuration management system is included by default. For hot-reloading support, install the optional watchdog dependency:

```bash
pip install watchdog
```

For schema validation support:

```bash
pip install jsonschema
```

Or install all optional dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Configuration

```python
from config.pipeline_config import PipelineConfig

# Load default configuration
config = PipelineConfig()

# Access configuration sections
print(f"Worker count: {config.face_detection.worker_count}")
print(f"FPS: {config.frame_capture.fps}")
```

### Configuration with Overrides

```python
# Override specific settings
config = PipelineConfig({
    'face_detection': {
        'worker_count': 4,
        'detector_type': 'gpu'
    },
    'face_recognition': {
        'tolerance': 0.5
    }
})
```

### Environment-Specific Configuration

```python
from config.environment import EnvironmentManager, Environment

# Auto-detect environment
env_manager = EnvironmentManager()
config = env_manager.load_environment_config()

# Or load specific environment
from config.environment import load_config_for_environment
config = load_config_for_environment(Environment.PRODUCTION)
```

### Configuration Validation

```python
from config.validation import ConfigValidator

validator = ConfigValidator()
result = validator.validate_full_config(config.to_dict())

if not result.is_valid:
    for error in result.errors:
        print(f"Error: {error.path}: {error.message}")
        if error.suggestion:
            print(f"  Suggestion: {error.suggestion}")
```

### Hot-Reloading

```python
from config.hot_reload import create_reloader

# Create reloader
reloader = create_reloader(message_bus=message_bus)

# Register callback for configuration changes
def on_config_change(config_diff):
    print(f"Configuration changed: {config_diff}")

reloader.register_reload_callback("my_component", on_config_change)

# Start watching for changes
reloader.start_watching()

# Get current configuration
current_config = reloader.get_config()

# Stop watching when done
reloader.stop_watching()
```

## Configuration Structure

### Main Sections

The configuration is organized into logical sections:

- **frame_capture**: Frame capture settings (FPS, resolution, quality)
- **motion_detection**: Motion detection parameters
- **face_detection**: Face detection configuration (workers, model, detector type)
- **face_recognition**: Face recognition settings (tolerance, workers, cache)
- **event_processing**: Event processing and enrichment
- **storage**: Database and file storage settings
- **notifications**: Notification system configuration
- **hardware**: Hardware interface settings (GPIO, camera)
- **monitoring**: System monitoring and health checks

### Example Configuration

```python
config = {
    'frame_capture': {
        'fps': 30,
        'resolution': (1280, 720),
        'quality': 85
    },
    'face_detection': {
        'enabled': True,
        'worker_count': 2,
        'detector_type': 'cpu',
        'model': 'hog'
    },
    'face_recognition': {
        'enabled': True,
        'tolerance': 0.6,
        'worker_count': 2
    },
    'monitoring': {
        'enabled': True,
        'health_check_interval': 30.0
    }
}
```

## Environment Configuration

### Supported Environments

- **Development**: Verbose logging, more resources, debugging enabled
- **Testing**: Minimal resources, fast startup, mocked hardware
- **Staging**: Production-like with extra logging
- **Production**: Optimized for reliability and performance

### Environment Detection

The system automatically detects the environment from:

1. `DOORBELL_ENV` environment variable (highest priority)
2. `ENV` environment variable
3. `CI` variable (sets to testing)
4. Cloud platform variables (Railway, Render, Vercel)
5. Defaults to development if none found

### Environment-Specific Settings

```bash
# Development
export DOORBELL_ENV=development
# - DEBUG logging
# - More workers
# - Mock hardware enabled

# Production
export DOORBELL_ENV=production
# - INFO logging
# - Conservative worker counts
# - Monitoring enabled
# - Backups enabled
```

## Environment Variables

The system supports loading configuration from environment variables:

```bash
# Worker configuration
export DOORBELL_WORKER_COUNT=4

# Face recognition
export DOORBELL_FACE_TOLERANCE=0.7

# Hardware
export DOORBELL_PIN=18
export DOORBELL_DETECTOR_TYPE=gpu

# Logging
export DOORBELL_LOG_LEVEL=DEBUG
```

## Configuration Files

### Loading from Files

```python
from config.pipeline_config import load_config_from_file

# Load from JSON
config = load_config_from_file('config/production.json')

# Load from YAML (requires PyYAML)
config = load_config_from_file('config/production.yaml')
```

### Example JSON Configuration

```json
{
  "face_detection": {
    "worker_count": 4,
    "detector_type": "cpu",
    "model": "hog"
  },
  "face_recognition": {
    "tolerance": 0.6,
    "worker_count": 2
  },
  "monitoring": {
    "enabled": true,
    "health_check_interval": 30.0
  }
}
```

## Validation

### Validation Rules

The validator checks for:

1. **Hardware Dependencies**
   - GPU detector requires CUDA
   - EdgeTPU detector requires Coral hardware
   - Telegram requires bot token

2. **Performance Settings**
   - High FPS warnings (>30)
   - Excessive worker counts (>8)
   - Large queue sizes (>1000)
   - Short timeouts (<5s)

3. **Resource Allocation**
   - Total worker count limits
   - Cache size warnings
   - Memory considerations

### Custom Validation

```python
from config.validation import ConfigValidator, ValidationError

# Create custom validator
validator = ConfigValidator()

# Validate specific section
result = validator.validate_section('face_detection', {
    'enabled': True,
    'worker_count': 2
})

if not result.is_valid:
    print(f"Validation failed: {result.errors}")
```

## Performance Optimization

### Hardware-Specific Optimization

```python
# Optimize for available hardware
config = PipelineConfig()
config.optimize_for_hardware(cpu_cores=4, memory_gb=8.0)
```

### Performance Profiles

The system automatically detects and reports performance profiles:

- **low_power**: 1-2 total workers (Raspberry Pi)
- **balanced**: 3-4 total workers (Standard)
- **performance**: 5-8 total workers (High-end)
- **high_performance**: 9+ total workers (Server)

```python
profile = config.get_performance_profile()
print(f"Current profile: {profile}")
```

## Configuration Factories

### Pre-configured Setups

```python
from config.pipeline_config import (
    create_minimal_config,
    create_production_config
)

# Minimal configuration (testing)
minimal = create_minimal_config()

# Production-ready configuration
production = create_production_config()
```

## Event System

Configuration changes trigger events that can be subscribed to:

```python
# Events are broadcast via message bus
# - CONFIGURATION_LOADED: Initial load
# - CONFIGURATION_RELOADED: Successful reload
# - CONFIGURATION_VALIDATION_FAILED: Validation error
# - CONFIGURATION_CHANGED: Configuration changed
```

## Best Practices

### 1. Use Environment-Specific Configurations

```python
# Always load environment-specific config
env_manager = EnvironmentManager()
config = env_manager.load_environment_config()
```

### 2. Validate Before Applying

```python
# Validate configuration before use
validator = ConfigValidator()
result = validator.validate_full_config(config.to_dict())

if not result.is_valid:
    raise ValueError(f"Invalid configuration: {result.errors}")
```

### 3. Register for Change Notifications

```python
# Get notified of configuration changes
def on_config_change(diff):
    # Reload component configuration
    self.reload_configuration()

reloader.register_reload_callback("my_component", on_config_change)
```

### 4. Use Appropriate Profiles

```python
# Development: High resources, verbose logging
if environment.is_development():
    config.storage.log_level = 'DEBUG'
    config.monitoring.enabled = True

# Production: Conservative, reliable
if environment.is_production():
    config.motion_detection.enabled = True
    config.storage.backup_enabled = True
```

## Testing

Comprehensive test suite included:

```bash
# Run all configuration tests
python tests/test_hot_reload.py          # 22 tests
python tests/test_config_validation.py   # 28 tests
python tests/test_environment_management.py  # 17+ tests
python tests/test_pipeline_config.py     # 25 tests
python tests/test_config_integration.py  # 12 tests
```

## Examples

Complete usage examples available in:

```bash
python examples/config_management_examples.py
```

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────┐
│         Configuration Management            │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐  ┌──────────────────┐   │
│  │ Hot Reload   │  │   Validation     │   │
│  │  - File      │  │   - Schema       │   │
│  │    Watching  │  │   - Dependencies │   │
│  │  - Callbacks │  │   - Performance  │   │
│  └──────────────┘  └──────────────────┘   │
│                                             │
│  ┌──────────────┐  ┌──────────────────┐   │
│  │ Environment  │  │  Pipeline Config │   │
│  │  - Auto      │  │   - Sections     │   │
│  │    Detection │  │   - Factories    │   │
│  │  - Overrides │  │   - Serialization│   │
│  └──────────────┘  └──────────────────┘   │
│                                             │
└─────────────────────────────────────────────┘
```

## Contributing

When adding new configuration options:

1. Add to appropriate dataclass in `pipeline_config.py`
2. Update JSON schema in `config/schemas/`
3. Add validation rules if needed in `validation.py`
4. Add tests for new options
5. Update environment templates if applicable
6. Document in this README

## Troubleshooting

### Hot-Reloading Not Working

- Check if `watchdog` is installed: `pip install watchdog`
- Verify file permissions on configuration files
- Check logs for file watching errors

### Validation Errors

- Review error messages and suggestions
- Check hardware availability for detector types
- Verify environment variables are set correctly

### Performance Issues

- Use `get_statistics()` to monitor reload performance
- Check validation time with performance tests
- Reduce worker counts for constrained hardware

## License

MIT License - See main project LICENSE file
