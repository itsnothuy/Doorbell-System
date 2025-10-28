# Pipeline Orchestrator Integration Guide

## Overview

This document describes the migration from the legacy `DoorbellSecuritySystem` to the new Frigate-inspired pipeline orchestrator architecture.

## Architecture Changes

### Before (Legacy Architecture)
```
app.py → DoorbellSecuritySystem → [FaceManager, CameraHandler, etc.] → Web Interface
```

### After (Pipeline Architecture)
```
app.py → OrchestratorManager → PipelineOrchestrator → [Frame Capture → Face Detection → Recognition → Event Processing] → Web Interface + Notifications
```

## Key Components

### OrchestratorManager
High-level management interface providing:
- Health monitoring and alerting
- Automatic recovery mechanisms
- Performance optimization
- Legacy compatibility layer
- Production deployment support

**Location**: `src/integration/orchestrator_manager.py`

**Usage**:
```python
from src.integration.orchestrator_manager import OrchestratorManager

# Create and start
manager = OrchestratorManager(config)
manager.start()

# Get health status
health = manager.get_health_status()

# Trigger doorbell (for testing)
result = manager.trigger_doorbell()

# Stop gracefully
manager.stop()
```

### LegacyAdapter
Provides backward compatibility with existing code that expects the old `DoorbellSecuritySystem` API.

**Location**: `src/integration/legacy_adapter.py`

**Usage**:
```python
# Get legacy interface from manager
manager = OrchestratorManager()
legacy_system = manager.get_legacy_interface()

# Use like old system
legacy_system.start()
legacy_system.on_doorbell_pressed(channel=18)
status = legacy_system.get_system_status()
legacy_system.stop()
```

**Compatible Attributes**:
- `settings` - Configuration settings
- `face_manager` - Face recognition interface
- `camera` - Camera handler interface
- `gpio` - GPIO handler interface

**Compatible Methods**:
- `start()` - Start the system
- `stop()` - Stop the system
- `on_doorbell_pressed(channel)` - Handle doorbell press
- `get_recent_captures(limit)` - Get recent capture events
- `get_system_status()` - Get system status

### MigrationUtils
Utilities for migrating from legacy to pipeline architecture.

**Location**: `src/integration/migration_utils.py`

**Usage**:
```python
from src.integration.migration_utils import MigrationUtils

# Validate compatibility
results = MigrationUtils.validate_migration_compatibility()
if results['compatible']:
    print("System is ready for migration")

# Migrate data
migration_results = MigrationUtils.migrate_legacy_data()

# Create backup
backup_results = MigrationUtils.create_backup()

# Verify pipeline health
health_results = MigrationUtils.verify_pipeline_health()
```

## Migration Steps

### Step 1: Backup Existing System
```python
from src.integration.migration_utils import MigrationUtils

# Create backup
backup = MigrationUtils.create_backup()
print(f"Backup created at: {backup['backup_path']}")
```

### Step 2: Validate Compatibility
```python
# Check compatibility
results = MigrationUtils.validate_migration_compatibility()
if not results['compatible']:
    print("Issues found:", results['issues'])
    # Resolve issues before proceeding
```

### Step 3: Update Application Code

#### Option A: Use app.py (Recommended for Web Deployment)
The `app.py` file has been updated to use the new architecture automatically:

```python
# app.py now uses OrchestratorManager internally
# No changes needed for existing deployments
```

#### Option B: Use src/main.py (Standalone Execution)
For standalone execution without web interface:

```bash
python src/main.py
```

Or programmatically:
```python
from src.main import main

# Run with custom config
exit_code = main(config={'health_check_interval': 60.0})
```

#### Option C: Direct Integration
For custom integrations:

```python
from src.integration.orchestrator_manager import OrchestratorManager

# Create manager
manager = OrchestratorManager({
    'health_check_interval': 30.0,
    'auto_recovery_enabled': True,
    'pipeline_config': {
        'face_detection': {'worker_count': 2},
        'face_recognition': {'worker_count': 2}
    }
})

# Start system
manager.start()

# Use legacy interface if needed
legacy = manager.get_legacy_interface()
```

### Step 4: Update Web Interface (If Custom)
The web interface automatically works with the legacy adapter. No changes needed unless you have custom integrations.

If you have custom code that directly uses `DoorbellSecuritySystem`:

**Before**:
```python
from src.doorbell_security import DoorbellSecuritySystem

system = DoorbellSecuritySystem()
system.start()
```

**After**:
```python
from src.integration.orchestrator_manager import OrchestratorManager

manager = OrchestratorManager()
manager.start()

# Get legacy interface for backward compatibility
system = manager.get_legacy_interface()
```

### Step 5: Verify Operation
```python
# Check health
health = manager.get_health_status()
print(f"State: {health.state}")
print(f"Performance Score: {health.performance_score}")
print(f"Uptime: {health.uptime}s")

# Test doorbell trigger
result = manager.trigger_doorbell()
print(f"Trigger result: {result}")
```

## Configuration

### Orchestrator Configuration
Configure the `OrchestratorManager` via `config/orchestrator_config.py`:

```python
config = {
    # Health monitoring
    'health_check_interval': 30.0,
    'metrics_collection_interval': 60.0,
    
    # Auto-recovery
    'auto_recovery_enabled': True,
    'max_restart_attempts': 3,
    'restart_cooldown_seconds': 60.0,
    
    # Performance thresholds
    'cpu_threshold': 80.0,
    'memory_threshold': 85.0,
    
    # Pipeline configuration
    'pipeline_config': {
        'face_detection': {
            'worker_count': 2,
            'detector_type': 'cpu'
        },
        'face_recognition': {
            'worker_count': 2,
            'tolerance': 0.6
        }
    }
}
```

### Environment Variables
Configure via environment variables:

```bash
# Health monitoring
export HEALTH_CHECK_INTERVAL=30.0

# Auto-recovery
export AUTO_RECOVERY_ENABLED=true
export MAX_RESTART_ATTEMPTS=3

# Performance thresholds
export CPU_THRESHOLD=80.0
export MEMORY_THRESHOLD=85.0

# Pipeline settings
export WORKER_COUNT=2
export FACE_RECOGNITION_TOLERANCE=0.6
```

## Monitoring and Health Checks

### Health Status
```python
health = manager.get_health_status()

# Check state
if health.state == SystemState.RUNNING:
    print("System is healthy")
elif health.state == SystemState.ERROR:
    print(f"Error: {health.last_error}")

# Check performance
if health.performance_score < 0.7:
    print("Performance degradation detected")

# Check pipeline status
print(f"Pipeline status: {health.pipeline_status}")
```

### Event Callbacks
Register callbacks for system events:

```python
def on_error(data):
    print(f"Error occurred: {data}")

def on_performance_warning(data):
    print(f"Performance issue: {data['score']}")

manager.register_event_callback('system_error', on_error)
manager.register_event_callback('performance_warning', on_performance_warning)
```

### Pipeline Metrics
```python
# Get detailed metrics
metrics = manager.orchestrator.get_pipeline_metrics()

print(f"Pipeline Status: {metrics['pipeline_status']}")
print(f"Uptime: {metrics['uptime']}s")
print(f"Events/minute: {metrics['events_per_minute']}")
print(f"Queue Status: {metrics['queue_status']}")
print(f"Worker Status: {metrics['worker_status']}")
```

## Troubleshooting

### Issue: System Won't Start
**Symptoms**: Manager fails during `start()`

**Solutions**:
1. Check hardware initialization:
   ```python
   from src.integration.migration_utils import MigrationUtils
   health = MigrationUtils.verify_pipeline_health()
   print(health['issues'])
   ```

2. Check configuration:
   ```python
   results = MigrationUtils.validate_migration_compatibility()
   print(results['issues'])
   ```

3. Check logs:
   ```bash
   tail -f data/logs/doorbell.log
   ```

### Issue: Performance Degradation
**Symptoms**: `performance_score < 0.7`

**Solutions**:
1. Check worker counts:
   ```python
   config = {
       'pipeline_config': {
           'face_detection': {'worker_count': 4},  # Increase workers
           'face_recognition': {'worker_count': 3}
       }
   }
   ```

2. Enable motion detection (for performance):
   ```python
   config = {
       'pipeline_config': {
           'motion_detection': {'enabled': True}
       }
   }
   ```

3. Check queue backlogs:
   ```python
   metrics = manager.orchestrator.get_pipeline_metrics()
   if metrics['queue_status'].get('backlog_warning'):
       # Increase worker counts or optimize processing
   ```

### Issue: Legacy Code Not Working
**Symptoms**: AttributeError or compatibility issues

**Solutions**:
1. Ensure using legacy adapter:
   ```python
   legacy = manager.get_legacy_interface()
   # Use 'legacy' instead of direct orchestrator
   ```

2. Check available attributes:
   ```python
   print(dir(legacy))
   print(dir(legacy.settings))
   print(dir(legacy.face_manager))
   ```

3. Report missing compatibility in issue tracker

### Issue: Auto-Recovery Not Working
**Symptoms**: System stays in ERROR state

**Solutions**:
1. Check auto-recovery is enabled:
   ```python
   config = {'auto_recovery_enabled': True}
   ```

2. Check restart attempts:
   ```python
   health = manager.get_health_status()
   if manager.restart_attempts >= manager.max_restart_attempts:
       # Manual intervention needed
       manager.restart_attempts = 0  # Reset counter
   ```

3. Check logs for recovery attempts:
   ```bash
   grep "recovery" data/logs/doorbell.log
   ```

## Performance Optimization

### Raspberry Pi
```python
config = {
    'pipeline_config': {
        'face_detection': {
            'worker_count': 1,
            'detector_type': 'cpu',
            'model': 'hog'
        },
        'face_recognition': {
            'worker_count': 1,
            'encoding_model': 'small'
        },
        'motion_detection': {'enabled': True}  # Reduces load
    }
}
```

### High-Performance Systems
```python
config = {
    'pipeline_config': {
        'face_detection': {
            'worker_count': 4,
            'detector_type': 'gpu',
            'model': 'cnn'
        },
        'face_recognition': {
            'worker_count': 3,
            'encoding_model': 'large'
        },
        'motion_detection': {'enabled': False}  # Not needed
    }
}
```

## Rollback Plan

If issues occur, you can rollback to legacy system:

### Step 1: Restore app.py
```python
# In app.py, comment out new code and uncomment legacy:
# from src.doorbell_security import DoorbellSecuritySystem
# doorbell_system = DoorbellSecuritySystem()
# doorbell_system.start()
```

### Step 2: Restore from Backup
```python
# Restore data from backup if needed
import shutil
from pathlib import Path

backup_dir = Path('data/backups/backup_YYYYMMDD_HHMMSS')
shutil.copytree(backup_dir / 'known_faces', 'data/known_faces', dirs_exist_ok=True)
# ... restore other directories
```

### Step 3: Restart
```bash
python app.py
```

## Testing

Run integration tests to verify the system:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run integration tests
pytest tests/integration/ -v

# Run specific test suites
pytest tests/integration/test_orchestrator_integration.py -v
pytest tests/integration/test_legacy_compatibility.py -v
pytest tests/integration/test_end_to_end_pipeline.py -v

# Run with coverage
pytest tests/integration/ --cov=src/integration --cov-report=html
```

## Best Practices

1. **Always create backups** before migration
2. **Test in development** before production deployment
3. **Monitor health metrics** after deployment
4. **Use legacy adapter** for gradual migration
5. **Enable auto-recovery** in production
6. **Configure appropriate worker counts** for your hardware
7. **Register event callbacks** for critical alerts
8. **Review logs regularly** for issues

## Additional Resources

- [Pipeline Architecture Documentation](ARCHITECTURE.md)
- [Configuration Management Guide](CONFIGURATION_MANAGEMENT.md)
- [Testing Documentation](TESTING.md)
- [Security Guidelines](SECURITY.md)
- [Development Roadmap](DEVELOPMENT_ROADMAP.md)

## Support

For issues or questions:
1. Check existing issues on GitHub
2. Review troubleshooting section above
3. Check logs in `data/logs/`
4. Create a new issue with:
   - System health output
   - Relevant log excerpts
   - Configuration used
   - Steps to reproduce
