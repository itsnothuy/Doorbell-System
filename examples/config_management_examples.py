#!/usr/bin/env python3
"""
Configuration Management System - Usage Examples

This file demonstrates how to use the hot-reloading, validation,
and environment management features of the configuration system.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Example 1: Basic Configuration Loading
# =======================================

from config.pipeline_config import PipelineConfig

# Load default configuration
config = PipelineConfig()
print(f"Default worker count: {config.face_detection.worker_count}")
print(f"Default FPS: {config.frame_capture.fps}")


# Example 2: Loading Configuration with Overrides
# ================================================

config_overrides = {
    'face_detection': {
        'worker_count': 4,
        'detector_type': 'cpu'
    },
    'face_recognition': {
        'tolerance': 0.5,
        'worker_count': 3
    }
}

config = PipelineConfig(config_overrides)
print(f"\nOverridden worker count: {config.face_detection.worker_count}")
print(f"Custom tolerance: {config.face_recognition.tolerance}")


# Example 3: Environment-Specific Configuration
# ==============================================

from config.environment import (
    EnvironmentManager,
    Environment,
    load_config_for_environment
)

# Auto-detect environment from DOORBELL_ENV variable
os.environ['DOORBELL_ENV'] = 'production'

env_manager = EnvironmentManager()
print(f"\nCurrent environment: {env_manager.get_environment().value}")
print(f"Is production: {env_manager.is_production()}")

# Load environment-specific configuration
env_config = env_manager.load_environment_config()
print(f"Production log level: {env_config.storage.log_level}")
print(f"Production monitoring: {env_config.monitoring.enabled}")

# Load config for specific environment (without changing current environment)
testing_config = load_config_for_environment(Environment.TESTING)
print(f"\nTesting worker count: {testing_config.face_detection.worker_count}")


# Example 4: Configuration Validation
# ====================================

from config.validation import ConfigValidator, create_validator

validator = create_validator()

# Validate a configuration
test_config = {
    'face_detection': {
        'enabled': True,
        'worker_count': 2,
        'detector_type': 'cpu'
    },
    'face_recognition': {
        'enabled': True,
        'timeout': 15.0
    }
}

result = validator.validate_full_config(test_config)

print(f"\nValidation result:")
print(f"  Is valid: {result.is_valid}")
print(f"  Has errors: {result.has_errors}")
print(f"  Has warnings: {result.has_warnings}")

if result.errors:
    print("  Errors:")
    for error in result.errors:
        print(f"    - {error.path}: {error.message}")

if result.warnings:
    print("  Warnings:")
    for warning in result.warnings:
        print(f"    - {warning.path}: {warning.message}")


# Example 5: Hot-Reloading Configuration
# =======================================

from config.hot_reload import ConfigurationReloader, create_reloader

# Create a mock message bus (in real usage, use actual message bus)
class MockMessageBus:
    def publish(self, topic, message):
        print(f"Event published to {topic}")

message_bus = MockMessageBus()

# Create reloader
config_path = Path(__file__).parent.parent / "config" / "pipeline_config.py"
reloader = create_reloader(config_path=config_path, message_bus=message_bus)

print(f"\nHot-reloader initialized:")
print(f"  Version: {reloader.config_version}")
print(f"  Reload count: {reloader.reload_count}")

# Register a callback to be notified of configuration changes
def on_config_change(config_diff):
    print(f"\n⚡ Configuration changed!")
    print(f"  Modified sections: {list(config_diff['modified'].keys())}")
    print(f"  Added sections: {list(config_diff['added'].keys())}")
    print(f"  Removed sections: {list(config_diff['removed'].keys())}")

reloader.register_reload_callback("example_component", on_config_change)

# Get current configuration
current_config = reloader.get_config()
print(f"\nCurrent configuration FPS: {current_config.frame_capture.fps}")

# Manually trigger a reload (in production, this happens automatically on file changes)
print("\n⚙️  Triggering configuration reload...")
success = reloader.reload_configuration(force=True)
print(f"  Reload success: {success}")

# Get reload statistics
stats = reloader.get_statistics()
print(f"\nReload statistics:")
print(f"  Version: {stats['config_version']}")
print(f"  Reload count: {stats['reload_count']}")
print(f"  Failed reloads: {stats['failed_reloads']}")
print(f"  Success rate: {stats['success_rate']:.1%}")


# Example 6: Hardware-Specific Optimization
# ==========================================

config = PipelineConfig()

# Optimize for low-end hardware (e.g., Raspberry Pi)
config.optimize_for_hardware(cpu_cores=2, memory_gb=1.0)
print(f"\nLow-end optimization:")
print(f"  Face detection workers: {config.face_detection.worker_count}")
print(f"  Motion detection enabled: {config.motion_detection.enabled}")

# Optimize for high-end hardware
config = PipelineConfig()
config.optimize_for_hardware(cpu_cores=16, memory_gb=32.0)
print(f"\nHigh-end optimization:")
print(f"  Face detection workers: {config.face_detection.worker_count}")
print(f"  Face recognition workers: {config.face_recognition.worker_count}")


# Example 7: Configuration Performance Profiles
# ==============================================

config = PipelineConfig()
profile = config.get_performance_profile()
print(f"\nCurrent performance profile: {profile}")

# Create configurations for different profiles
minimal = PipelineConfig({'face_detection': {'worker_count': 1}})
print(f"Minimal profile: {minimal.get_performance_profile()}")

balanced = PipelineConfig({
    'face_detection': {'worker_count': 2},
    'face_recognition': {'worker_count': 2}
})
print(f"Balanced profile: {balanced.get_performance_profile()}")


# Example 8: Environment Variable Integration
# ============================================

# Set environment variables
os.environ['DOORBELL_WORKER_COUNT'] = '8'
os.environ['DOORBELL_FACE_TOLERANCE'] = '0.7'
os.environ['DOORBELL_LOG_LEVEL'] = 'DEBUG'

# Configuration automatically loads from environment variables
config = PipelineConfig()
print(f"\nEnvironment variable loading:")
print(f"  Worker count from env: {config.face_detection.worker_count}")


# Example 9: Configuration File Loading
# ======================================

from config.pipeline_config import load_config_from_file, create_minimal_config, create_production_config
import tempfile
import json

# Create a temporary config file
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    config_data = {
        'face_detection': {
            'worker_count': 5,
            'detector_type': 'cpu'
        }
    }
    json.dump(config_data, f)
    temp_config_path = f.name

# Load from file
loaded_config = load_config_from_file(temp_config_path)
print(f"\nLoaded from file:")
print(f"  Worker count: {loaded_config.face_detection.worker_count}")

# Cleanup
Path(temp_config_path).unlink()

# Use factory functions
minimal_config = create_minimal_config()
print(f"\nMinimal config:")
print(f"  Workers: {minimal_config.face_detection.worker_count}")
print(f"  Monitoring: {minimal_config.monitoring.enabled}")

production_config = create_production_config()
print(f"\nProduction config:")
print(f"  Motion detection: {production_config.motion_detection.enabled}")
print(f"  Backups: {production_config.storage.backup_enabled}")


# Example 10: Advanced Validation with Dependencies
# ==================================================

# Test invalid configuration (GPU without CUDA)
invalid_config = {
    'face_detection': {
        'enabled': True,
        'detector_type': 'gpu'  # GPU detector
    },
    'hardware': {
        'gpu_available': False  # But no GPU available!
    }
}

result = validator.validate_full_config(invalid_config)
print(f"\nValidating GPU config without CUDA:")
print(f"  Is valid: {result.is_valid}")
if result.errors:
    print("  Errors found:")
    for error in result.errors:
        print(f"    - {error.message}")
        if error.suggestion:
            print(f"      Suggestion: {error.suggestion}")


# Example 11: Context Manager for Hot-Reloading
# ==============================================

print("\n" + "="*60)
print("Example: Using hot-reloader as context manager")
print("="*60)

with create_reloader(config_path=config_path) as reloader:
    print(f"Reloader active, version: {reloader.config_version}")
    # Reloader is watching for changes
    # When context exits, watching will stop automatically
print("Reloader context exited, watching stopped")


print("\n" + "="*60)
print("✅ All examples completed successfully!")
print("="*60)
