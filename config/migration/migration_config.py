#!/usr/bin/env python3
"""
Migration Configuration - Settings for Migration Process

Configuration specific to the migration process from legacy to pipeline architecture.
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class MigrationConfig:
    """Configuration for migration process."""
    
    # Backup settings
    backup_dir: str = "data/migration_backup"
    compress_backup: bool = False
    keep_backup_days: int = 30
    
    # Migration settings
    auto_rollback: bool = True
    validate_before_cleanup: bool = True
    
    # Validation thresholds
    min_disk_space_gb: float = 1.0
    max_migration_time_minutes: int = 30
    
    # Logging
    log_file: str = "data/logs/migration.log"
    log_level: str = "INFO"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'backup_dir': self.backup_dir,
            'compress_backup': self.compress_backup,
            'keep_backup_days': self.keep_backup_days,
            'auto_rollback': self.auto_rollback,
            'validate_before_cleanup': self.validate_before_cleanup,
            'min_disk_space_gb': self.min_disk_space_gb,
            'max_migration_time_minutes': self.max_migration_time_minutes,
            'log_file': self.log_file,
            'log_level': self.log_level
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MigrationConfig':
        """Create from dictionary."""
        return cls(**config_dict)
