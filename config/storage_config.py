#!/usr/bin/env python3
"""
Storage Configuration

Configuration settings for the storage layer including database paths,
retention policies, and performance tuning parameters.
"""

import os
from pathlib import Path
from typing import Any, Dict

# Base data directory
BASE_DATA_DIR = os.getenv('DATA_DIR', 'data')
BASE_BACKUP_DIR = os.getenv('BACKUP_DIR', f'{BASE_DATA_DIR}/backups')

# Ensure directories exist
Path(BASE_DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(BASE_BACKUP_DIR).mkdir(parents=True, exist_ok=True)


# Storage Configuration
STORAGE_CONFIG: Dict[str, Any] = {
    # General storage settings
    "data_directory": BASE_DATA_DIR,
    "backup_directory": BASE_BACKUP_DIR,
    "backup_enabled": True,
    "enable_wal_mode": True,
    "connection_pool_size": 10,
    "auto_vacuum": True,
    
    # Event database configuration
    "event_db_config": {
        "database_path": f"{BASE_DATA_DIR}/events.db",
        "table_name": "events",
        "enable_full_text_search": False,  # Can be enabled for advanced search
        "retention_days": 90,
        "wal_mode": True,
        "connection_pool_size": 10,
        "batch_insert_size": 100,
        "auto_vacuum": True
    },
    
    # Face database configuration
    "face_db_config": {
        "database_path": f"{BASE_DATA_DIR}/faces.db",
        "encoding_version": "1.0",
        "max_faces_per_person": 10,
        "backup_encodings": True,
        "wal_mode": True,
        "connection_pool_size": 5
    },
    
    # Metrics database configuration
    "metrics_db_config": {
        "database_path": f"{BASE_DATA_DIR}/metrics.db",
        "retention_days": 30,
        "aggregation_enabled": True,
        "hourly_aggregation": True,
        "daily_aggregation": True,
        "wal_mode": True,
        "connection_pool_size": 5,
        "auto_vacuum": True
    },
    
    # Configuration database
    "config_db_config": {
        "database_path": f"{BASE_DATA_DIR}/config.db",
        "enable_versioning": True,
        "max_versions_per_key": 50,
        "audit_trail": True,
        "wal_mode": True,
        "connection_pool_size": 3
    },
    
    # Notification database (existing)
    "notification_db_config": {
        "database_path": f"{BASE_DATA_DIR}/notifications.db",
        "retention_days": 30,
        "wal_mode": True
    },
    
    # Backup configuration
    "backup_config": {
        "enabled": True,
        "backup_directory": BASE_BACKUP_DIR,
        "backup_retention_days": 7,
        "auto_backup_interval_hours": 24,
        "compress_backups": True,
        "backup_on_shutdown": True
    },
    
    # Migration configuration
    "migration_config": {
        "migrations_directory": "src/storage/migrations",
        "auto_migrate": True,
        "backup_before_migration": True
    },
    
    # Performance tuning
    "performance_config": {
        "query_timeout_seconds": 30,
        "batch_size": 100,
        "cache_size_mb": 64,
        "enable_query_optimization": True,
        "connection_timeout_seconds": 10
    },
    
    # Maintenance configuration
    "maintenance_config": {
        "auto_vacuum_enabled": True,
        "auto_cleanup_enabled": True,
        "cleanup_schedule_hour": 3,  # 3 AM daily cleanup
        "vacuum_schedule_day": 0,     # Sunday weekly vacuum
        "integrity_check_enabled": True
    }
}


# Development/Testing configuration overrides
if os.getenv('DEVELOPMENT_MODE', '').lower() == 'true':
    STORAGE_CONFIG.update({
        "event_db_config": {
            **STORAGE_CONFIG["event_db_config"],
            "retention_days": 7,  # Shorter retention in dev
            "wal_mode": False  # Simpler mode for testing
        },
        "metrics_db_config": {
            **STORAGE_CONFIG["metrics_db_config"],
            "retention_days": 3,
            "wal_mode": False
        },
        "backup_config": {
            **STORAGE_CONFIG["backup_config"],
            "enabled": False  # Disable auto-backup in dev
        }
    })


# Production configuration overrides
if os.getenv('PRODUCTION_MODE', '').lower() == 'true':
    STORAGE_CONFIG.update({
        "event_db_config": {
            **STORAGE_CONFIG["event_db_config"],
            "retention_days": 180,  # Longer retention in production
        },
        "metrics_db_config": {
            **STORAGE_CONFIG["metrics_db_config"],
            "retention_days": 90,
        },
        "backup_config": {
            **STORAGE_CONFIG["backup_config"],
            "enabled": True,
            "backup_retention_days": 30,
            "auto_backup_interval_hours": 12  # More frequent backups
        }
    })


def get_storage_config() -> Dict[str, Any]:
    """
    Get storage configuration.
    
    Returns:
        Storage configuration dictionary
    """
    return STORAGE_CONFIG.copy()


def get_database_config(db_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific database.
    
    Args:
        db_name: Database name (event_db, face_db, metrics_db, config_db)
        
    Returns:
        Database-specific configuration
    """
    config_key = f"{db_name}_config"
    return STORAGE_CONFIG.get(config_key, {}).copy()


def update_storage_config(updates: Dict[str, Any]) -> None:
    """
    Update storage configuration at runtime.
    
    Args:
        updates: Dictionary of configuration updates
    """
    STORAGE_CONFIG.update(updates)


# Export configuration
__all__ = [
    'STORAGE_CONFIG',
    'get_storage_config',
    'get_database_config',
    'update_storage_config',
    'BASE_DATA_DIR',
    'BASE_BACKUP_DIR'
]
