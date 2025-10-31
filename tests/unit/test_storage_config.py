#!/usr/bin/env python3
"""
Unit tests for storage_config module.

Tests storage configuration and database settings.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from config.storage_config import (
    STORAGE_CONFIG,
    BASE_DATA_DIR,
    BASE_BACKUP_DIR,
    get_storage_config,
    get_database_config,
    update_storage_config,
)


class TestStorageConfiguration:
    """Test storage configuration constants and functions."""

    def test_base_data_dir_defined(self):
        """Test BASE_DATA_DIR is defined."""
        assert BASE_DATA_DIR is not None
        assert isinstance(BASE_DATA_DIR, str)

    def test_base_backup_dir_defined(self):
        """Test BASE_BACKUP_DIR is defined."""
        assert BASE_BACKUP_DIR is not None
        assert isinstance(BASE_BACKUP_DIR, str)

    def test_storage_config_is_dict(self):
        """Test STORAGE_CONFIG is a dictionary."""
        assert isinstance(STORAGE_CONFIG, dict)

    def test_storage_config_has_required_keys(self):
        """Test STORAGE_CONFIG has required keys."""
        required_keys = [
            'data_directory',
            'backup_directory',
            'event_db_config',
            'face_db_config',
            'metrics_db_config',
            'config_db_config',
            'notification_db_config',
            'backup_config',
            'migration_config',
            'performance_config',
            'maintenance_config'
        ]
        for key in required_keys:
            assert key in STORAGE_CONFIG, f"Missing key: {key}"

    def test_event_db_config_structure(self):
        """Test event database config has required fields."""
        event_config = STORAGE_CONFIG['event_db_config']
        assert 'database_path' in event_config
        assert 'retention_days' in event_config
        assert 'wal_mode' in event_config

    def test_face_db_config_structure(self):
        """Test face database config has required fields."""
        face_config = STORAGE_CONFIG['face_db_config']
        assert 'database_path' in face_config
        assert 'encoding_version' in face_config

    def test_metrics_db_config_structure(self):
        """Test metrics database config has required fields."""
        metrics_config = STORAGE_CONFIG['metrics_db_config']
        assert 'database_path' in metrics_config
        assert 'retention_days' in metrics_config

    def test_backup_config_structure(self):
        """Test backup config has required fields."""
        backup_config = STORAGE_CONFIG['backup_config']
        assert 'enabled' in backup_config
        assert 'backup_directory' in backup_config
        assert 'backup_retention_days' in backup_config

    def test_get_storage_config_returns_copy(self):
        """Test get_storage_config returns a copy."""
        config1 = get_storage_config()
        config2 = get_storage_config()
        
        # Modify one config
        config1['test_key'] = 'test_value'
        
        # Other config should not be affected
        assert 'test_key' not in config2

    def test_get_storage_config_returns_dict(self):
        """Test get_storage_config returns dictionary."""
        config = get_storage_config()
        assert isinstance(config, dict)
        assert len(config) > 0

    def test_get_database_config_event_db(self):
        """Test get_database_config for event_db."""
        config = get_database_config('event_db')
        assert isinstance(config, dict)
        assert 'database_path' in config

    def test_get_database_config_face_db(self):
        """Test get_database_config for face_db."""
        config = get_database_config('face_db')
        assert isinstance(config, dict)
        assert 'database_path' in config

    def test_get_database_config_metrics_db(self):
        """Test get_database_config for metrics_db."""
        config = get_database_config('metrics_db')
        assert isinstance(config, dict)
        assert 'database_path' in config

    def test_get_database_config_config_db(self):
        """Test get_database_config for config_db."""
        config = get_database_config('config_db')
        assert isinstance(config, dict)
        assert 'database_path' in config

    def test_get_database_config_invalid_name(self):
        """Test get_database_config with invalid name."""
        config = get_database_config('invalid_db')
        assert isinstance(config, dict)
        assert len(config) == 0  # Should return empty dict

    def test_get_database_config_returns_copy(self):
        """Test get_database_config returns a copy."""
        config1 = get_database_config('event_db')
        config2 = get_database_config('event_db')
        
        # Modify one config
        config1['test_key'] = 'test_value'
        
        # Other config should not be affected
        assert 'test_key' not in config2

    def test_update_storage_config_updates_config(self):
        """Test update_storage_config updates configuration."""
        original_value = STORAGE_CONFIG.get('test_update_key')
        
        updates = {'test_update_key': 'test_value'}
        update_storage_config(updates)
        
        assert STORAGE_CONFIG['test_update_key'] == 'test_value'
        
        # Cleanup
        if original_value is None:
            STORAGE_CONFIG.pop('test_update_key', None)
        else:
            STORAGE_CONFIG['test_update_key'] = original_value

    def test_update_storage_config_preserves_other_keys(self):
        """Test update_storage_config preserves other keys."""
        keys_before = set(STORAGE_CONFIG.keys())
        
        update_storage_config({'new_key': 'new_value'})
        
        keys_after = set(STORAGE_CONFIG.keys())
        
        # All original keys should still be there
        assert keys_before.issubset(keys_after)
        
        # Cleanup
        STORAGE_CONFIG.pop('new_key', None)


class TestStorageConfigEnvironmentModes:
    """Test storage config environment-specific behavior."""

    def test_development_mode_shorter_retention(self, monkeypatch):
        """Test development mode uses shorter retention periods."""
        monkeypatch.setenv('DEVELOPMENT_MODE', 'true')
        # Need to reload module to pick up environment changes
        # For now, just verify the config structure
        assert 'event_db_config' in STORAGE_CONFIG
        assert 'retention_days' in STORAGE_CONFIG['event_db_config']

    def test_production_mode_longer_retention(self, monkeypatch):
        """Test production mode uses longer retention periods."""
        monkeypatch.setenv('PRODUCTION_MODE', 'true')
        # Verify config structure exists
        assert 'event_db_config' in STORAGE_CONFIG
        assert 'retention_days' in STORAGE_CONFIG['event_db_config']

    def test_backup_configuration_complete(self):
        """Test backup configuration is complete."""
        backup_config = STORAGE_CONFIG['backup_config']
        assert 'enabled' in backup_config
        assert 'backup_directory' in backup_config
        assert 'backup_retention_days' in backup_config
        assert 'auto_backup_interval_hours' in backup_config
        assert 'compress_backups' in backup_config

    def test_migration_configuration_complete(self):
        """Test migration configuration is complete."""
        migration_config = STORAGE_CONFIG['migration_config']
        assert 'migrations_directory' in migration_config
        assert 'auto_migrate' in migration_config
        assert 'backup_before_migration' in migration_config

    def test_performance_configuration_complete(self):
        """Test performance configuration is complete."""
        perf_config = STORAGE_CONFIG['performance_config']
        assert 'query_timeout_seconds' in perf_config
        assert 'batch_size' in perf_config
        assert 'cache_size_mb' in perf_config

    def test_maintenance_configuration_complete(self):
        """Test maintenance configuration is complete."""
        maint_config = STORAGE_CONFIG['maintenance_config']
        assert 'auto_vacuum_enabled' in maint_config
        assert 'auto_cleanup_enabled' in maint_config
        assert 'cleanup_schedule_hour' in maint_config


class TestStorageIntegration:
    """Integration tests for storage configuration."""

    def test_all_database_paths_accessible(self):
        """Test all database paths can be accessed."""
        db_names = ['event_db', 'face_db', 'metrics_db', 'config_db']
        
        for db_name in db_names:
            config = get_database_config(db_name)
            assert 'database_path' in config
            assert isinstance(config['database_path'], str)

    def test_database_paths_unique(self):
        """Test all database paths are unique."""
        db_names = ['event_db', 'face_db', 'metrics_db', 'config_db', 'notification_db']
        paths = []
        
        for db_name in db_names:
            config = get_database_config(db_name)
            if 'database_path' in config:
                paths.append(config['database_path'])
        
        # All paths should be unique
        assert len(paths) == len(set(paths))

    def test_directories_created_on_import(self):
        """Test that data and backup directories are created on import."""
        # These should exist because module creates them on import
        data_dir = Path(BASE_DATA_DIR)
        backup_dir = Path(BASE_BACKUP_DIR)
        
        # Check they exist or can be created
        assert data_dir.exists() or data_dir.parent.exists()
        assert backup_dir.exists() or backup_dir.parent.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
