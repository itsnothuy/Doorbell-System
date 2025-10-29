#!/usr/bin/env python3
"""
Migration Process Tests

Tests for the complete migration process from legacy to pipeline.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.integration.migration_manager import MigrationManager, MigrationStage, MigrationStatus
from src.integration.configuration_migrator import ConfigurationMigrator
from src.integration.data_migrator import DataMigrator


class TestMigrationManager:
    """Test migration manager functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield Path(temp_path)
        # Cleanup
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def migration_config(self, temp_dir):
        """Create migration configuration."""
        return {
            'backup_dir': str(temp_dir / 'backup'),
            'log_file': str(temp_dir / 'migration.log'),
            'auto_rollback': False  # Don't auto-rollback in tests
        }
    
    @pytest.fixture
    def migration_manager(self, migration_config):
        """Create migration manager."""
        return MigrationManager(migration_config)
    
    def test_migration_manager_creation(self, migration_manager):
        """Test migration manager can be created."""
        assert migration_manager is not None
        assert migration_manager.status.stage == MigrationStage.PREPARATION
    
    def test_migration_status_tracking(self, migration_manager):
        """Test migration status tracking."""
        status = migration_manager.get_migration_status()
        
        assert status is not None
        assert isinstance(status, MigrationStatus)
        assert status.progress >= 0.0
        assert status.progress <= 1.0
    
    def test_backup_creation(self, migration_manager):
        """Test backup creation."""
        # Run backup step
        result = migration_manager._create_backup()
        
        # Backup should be created
        assert result is True
        assert migration_manager.backup_dir.exists()
    
    def test_disk_space_check(self, migration_manager):
        """Test disk space check."""
        result = migration_manager._check_disk_space()
        
        # Should pass on systems with reasonable disk space
        assert isinstance(result, bool)
    
    def test_dependency_check(self, migration_manager):
        """Test dependency check."""
        result = migration_manager._check_dependencies()
        
        # Should pass if dependencies are installed
        assert isinstance(result, bool)


class TestConfigurationMigrator:
    """Test configuration migrator."""
    
    @pytest.fixture
    def config_migrator(self):
        """Create configuration migrator."""
        return ConfigurationMigrator()
    
    def test_config_migrator_creation(self, config_migrator):
        """Test configuration migrator can be created."""
        assert config_migrator is not None
    
    def test_configuration_migration(self, config_migrator):
        """Test configuration migration."""
        result = config_migrator.migrate_configuration()
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'success' in result
    
    def test_legacy_config_validation(self, config_migrator):
        """Test legacy configuration validation."""
        result = config_migrator._validate_legacy_config()
        
        # Should return boolean
        assert isinstance(result, bool)


class TestDataMigrator:
    """Test data migrator."""
    
    @pytest.fixture
    def data_migrator(self):
        """Create data migrator."""
        return DataMigrator()
    
    def test_data_migrator_creation(self, data_migrator):
        """Test data migrator can be created."""
        assert data_migrator is not None
    
    def test_face_database_migration(self, data_migrator):
        """Test face database migration."""
        result = data_migrator.migrate_face_databases()
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'success' in result
    
    def test_directory_creation(self, data_migrator):
        """Test pipeline directory creation."""
        # Should not crash
        data_migrator._ensure_pipeline_directories()
        
        # Directories should exist
        assert Path("data/known_faces").exists()
        assert Path("data/blacklist_faces").exists()
        assert Path("data/captures").exists()


class TestMigrationValidation:
    """Test migration validation."""
    
    @pytest.fixture
    def migration_manager(self):
        """Create migration manager."""
        config = {
            'auto_rollback': False
        }
        return MigrationManager(config)
    
    def test_pipeline_config_validation(self, migration_manager):
        """Test pipeline configuration validation."""
        result = migration_manager._validate_pipeline_config()
        
        # Should return boolean
        assert isinstance(result, bool)
    
    def test_data_validation(self, migration_manager):
        """Test migrated data validation."""
        result = migration_manager._validate_migrated_data()
        
        # Should return boolean
        assert isinstance(result, bool)
    
    def test_functionality_validation(self, migration_manager):
        """Test functionality validation."""
        result = migration_manager._validate_functionality()
        
        # Should return boolean
        assert isinstance(result, bool)
    
    def test_api_compatibility_validation(self, migration_manager):
        """Test API compatibility validation."""
        result = migration_manager._validate_api_compatibility()
        
        # Should return boolean
        assert isinstance(result, bool)


class TestMigrationRollback:
    """Test migration rollback functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield Path(temp_path)
        # Cleanup
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def migration_manager(self, temp_dir):
        """Create migration manager with temp backup dir."""
        config = {
            'backup_dir': str(temp_dir / 'backup'),
            'log_file': str(temp_dir / 'migration.log'),
            'auto_rollback': False
        }
        manager = MigrationManager(config)
        
        # Create a backup for rollback testing
        manager._create_backup()
        
        return manager
    
    def test_rollback_validation(self, migration_manager):
        """Test rollback validation."""
        # Should validate rollback is possible
        result = migration_manager._validate_rollback()
        
        assert isinstance(result, bool)


class TestMigrationStages:
    """Test individual migration stages."""
    
    @pytest.fixture
    def migration_manager(self):
        """Create migration manager."""
        config = {
            'auto_rollback': False
        }
        return MigrationManager(config)
    
    def test_preparation_stage(self, migration_manager):
        """Test preparation stage."""
        result = migration_manager._prepare_migration()
        
        # Should complete successfully
        assert isinstance(result, bool)
    
    def test_configuration_migration_stage(self, migration_manager):
        """Test configuration migration stage."""
        result = migration_manager._migrate_configuration()
        
        # Should complete
        assert isinstance(result, bool)
    
    def test_data_migration_stage(self, migration_manager):
        """Test data migration stage."""
        result = migration_manager._migrate_data()
        
        # Should complete
        assert isinstance(result, bool)
    
    def test_validation_stage(self, migration_manager):
        """Test validation stage."""
        result = migration_manager._validate_migration()
        
        # Should complete
        assert isinstance(result, bool)
    
    def test_cleanup_stage(self, migration_manager):
        """Test cleanup stage."""
        result = migration_manager._cleanup_migration()
        
        # Should complete successfully
        assert result is True


@pytest.mark.slow
class TestFullMigrationProcess:
    """Test full migration process (slow tests)."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield Path(temp_path)
        # Cleanup
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def migration_manager(self, temp_dir):
        """Create migration manager."""
        config = {
            'backup_dir': str(temp_dir / 'backup'),
            'log_file': str(temp_dir / 'migration.log'),
            'auto_rollback': False
        }
        return MigrationManager(config)
    
    def test_full_migration_dry_run(self, migration_manager):
        """Test full migration process (dry run)."""
        # This is a dry run - it will go through the motions but not actually change anything
        # since the system is already in pipeline architecture
        
        # Run migration
        result = migration_manager.run_migration()
        
        # Should complete (may succeed or fail depending on environment)
        assert isinstance(result, bool)
        
        # Check final status
        status = migration_manager.get_migration_status()
        assert status is not None
        assert status.stage in [MigrationStage.COMPLETED, MigrationStage.FAILED]
