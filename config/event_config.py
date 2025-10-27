#!/usr/bin/env python3
"""
Event Processing Configuration

Configuration settings for the event processing system including enrichment
processors, database settings, and performance tuning parameters.
"""

from pathlib import Path
from typing import Dict, Any

# Base event processing configuration
BASE_CONFIG = {
    'worker_count': 3,
    'queue_size': 1000,
    'timeout': 30.0,
    'enabled': True
}

# Database configuration
DATABASE_CONFIG = {
    'type': 'sqlite',
    'path': 'data/events.db',
    'connection_pool_size': 10,
    'batch_insert_size': 100,
    'wal_mode': True,  # Write-Ahead Logging for better concurrency
    'auto_vacuum': True
}

# Enrichment processor configuration
ENRICHMENT_CONFIG = {
    'enabled_processors': [
        'metadata_enrichment',
        'alert_manager',
        'notification_handler',
        'web_events'
    ],
    'max_enrichment_time': 5.0,
    'retry_failed_enrichments': True,
    'max_retries': 3,
    'retry_delay': 1.0,  # seconds
    'timeout_per_processor': 2.0
}

# Notification routing configuration
NOTIFICATION_CONFIG = {
    'internal_alerts': {
        'enabled': True,
        'priority_threshold': 'MEDIUM'
    },
    'web_notifications': {
        'enabled': True,
        'real_time_streaming': True,
        'max_connections': 50,
        'heartbeat_interval': 30.0
    }
}

# Performance configuration
PERFORMANCE_CONFIG = {
    'max_concurrent_events': 100,
    'event_timeout': 60.0,
    'metrics_collection': True,
    'health_check_interval': 30.0,
    'batch_processing': True,
    'batch_size': 10,
    'batch_timeout': 1.0
}

# Event retention and cleanup
RETENTION_CONFIG = {
    'retention_days': 30,
    'cleanup_interval_hours': 24,
    'archive_old_events': False,
    'archive_path': 'data/archived_events'
}

# Event priority configuration
PRIORITY_CONFIG = {
    'blacklist_detected': 'CRITICAL',
    'unknown_person': 'HIGH',
    'known_person': 'NORMAL',
    'system_event': 'NORMAL',
    'motion_detected': 'LOW'
}

# Web streaming configuration
WEB_STREAMING_CONFIG = {
    'enabled': True,
    'port': 5001,
    'host': '0.0.0.0',
    'cors_enabled': True,
    'cors_origins': ['*'],
    'compression': True,
    'buffer_size': 100
}


def get_event_config() -> Dict[str, Any]:
    """
    Get complete event processing configuration.
    
    Returns:
        Dictionary with all event processing configuration
    """
    return {
        'base_config': BASE_CONFIG,
        'database_config': DATABASE_CONFIG,
        'enrichment_config': ENRICHMENT_CONFIG,
        'notification_config': NOTIFICATION_CONFIG,
        'performance_config': PERFORMANCE_CONFIG,
        'retention_config': RETENTION_CONFIG,
        'priority_config': PRIORITY_CONFIG,
        'web_streaming_config': WEB_STREAMING_CONFIG
    }


def get_database_path() -> Path:
    """Get the database path as a Path object."""
    db_path = Path(DATABASE_CONFIG['path'])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


# Development/test overrides
DEV_CONFIG_OVERRIDES = {
    'database_config': {
        'path': 'data/test_events.db'
    },
    'performance_config': {
        'metrics_collection': True,
        'health_check_interval': 10.0
    },
    'retention_config': {
        'retention_days': 7
    }
}


def get_dev_config() -> Dict[str, Any]:
    """Get configuration for development/testing."""
    config = get_event_config()
    
    # Apply dev overrides
    for key, overrides in DEV_CONFIG_OVERRIDES.items():
        if key in config:
            config[key].update(overrides)
    
    return config
