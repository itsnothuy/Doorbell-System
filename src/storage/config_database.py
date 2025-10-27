#!/usr/bin/env python3
"""
Configuration Database - Versioned Configuration Storage

Manages system configuration with version history, rollback capability,
and audit trail. Supports hierarchical configuration structure.
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.storage.base_storage import BaseDatabase, DatabaseConfig

logger = logging.getLogger(__name__)


@dataclass
class ConfigVersion:
    """Represents a configuration version."""
    version_id: int
    config_key: str
    config_value: Dict[str, Any]
    description: str
    created_by: str
    created_at: float
    is_active: bool = False


@dataclass
class ConfigHistory:
    """Configuration change history entry."""
    change_id: int
    config_key: str
    old_value: Optional[Dict[str, Any]]
    new_value: Dict[str, Any]
    changed_by: str
    change_reason: str
    timestamp: float


class ConfigDatabase(BaseDatabase):
    """Database for versioned configuration management."""
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize configuration database.
        
        Args:
            config: Database configuration
        """
        super().__init__(config)
        self.config_table = "configurations"
        self.config_versions_table = "configuration_versions"
        self.config_history_table = "configuration_history"
    
    def initialize(self) -> bool:
        """Initialize configuration database schema."""
        try:
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            
            # Configure database
            self._configure_database()
            
            # Create schema
            self._create_tables()
            
            logger.info("Configuration database initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize configuration database: {e}")
            return False
    
    def _create_tables(self) -> None:
        """Create configuration database tables."""
        with self.get_cursor() as cursor:
            # Main configurations table (current active configurations)
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.config_table} (
                    config_key TEXT PRIMARY KEY,
                    config_value TEXT NOT NULL,
                    config_type TEXT DEFAULT 'general',
                    description TEXT,
                    is_sensitive INTEGER DEFAULT 0,
                    version INTEGER DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    updated_by TEXT
                )
            ''')
            
            # Configuration versions table (version history)
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.config_versions_table} (
                    version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_key TEXT NOT NULL,
                    config_value TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    description TEXT,
                    created_by TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    is_active INTEGER DEFAULT 0,
                    FOREIGN KEY (config_key) REFERENCES {self.config_table}(config_key)
                )
            ''')
            
            # Configuration change history table (audit trail)
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.config_history_table} (
                    change_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_key TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT NOT NULL,
                    changed_by TEXT NOT NULL,
                    change_reason TEXT,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (config_key) REFERENCES {self.config_table}(config_key)
                )
            ''')
            
            # Create indexes
            self._create_indexes(cursor)
    
    def _create_indexes(self, cursor: sqlite3.Cursor) -> None:
        """Create performance indexes."""
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_config_type ON {self.config_table}(config_type)",
            f"CREATE INDEX IF NOT EXISTS idx_config_updated ON {self.config_table}(updated_at DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_version_key ON {self.config_versions_table}(config_key)",
            f"CREATE INDEX IF NOT EXISTS idx_version_active ON {self.config_versions_table}(is_active)",
            f"CREATE INDEX IF NOT EXISTS idx_history_key ON {self.config_history_table}(config_key)",
            f"CREATE INDEX IF NOT EXISTS idx_history_timestamp ON {self.config_history_table}(timestamp DESC)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
    
    def set_config(self, config_key: str, config_value: Dict[str, Any],
                   description: Optional[str] = None, updated_by: str = "system",
                   change_reason: Optional[str] = None) -> bool:
        """
        Set or update a configuration value.
        
        Args:
            config_key: Configuration key
            config_value: Configuration value dictionary
            description: Optional description
            updated_by: User/system making the change
            change_reason: Reason for the change
            
        Returns:
            True if successful
        """
        try:
            with self.get_cursor() as cursor:
                current_time = time.time()
                config_value_json = json.dumps(config_value)
                
                # Get current value if exists
                cursor.execute(f'''
                    SELECT config_value, version FROM {self.config_table}
                    WHERE config_key = ?
                ''', (config_key,))
                
                row = cursor.fetchone()
                old_value = json.loads(row['config_value']) if row else None
                new_version = (row['version'] + 1) if row else 1
                
                # Update or insert configuration
                cursor.execute(f'''
                    INSERT OR REPLACE INTO {self.config_table}
                    (config_key, config_value, description, version, 
                     created_at, updated_at, updated_by)
                    VALUES (?, ?, ?, ?, 
                            COALESCE((SELECT created_at FROM {self.config_table} 
                                     WHERE config_key = ?), ?),
                            ?, ?)
                ''', (config_key, config_value_json, description, new_version,
                      config_key, current_time, current_time, updated_by))
                
                # Create version entry
                cursor.execute(f'''
                    UPDATE {self.config_versions_table}
                    SET is_active = 0
                    WHERE config_key = ?
                ''', (config_key,))
                
                cursor.execute(f'''
                    INSERT INTO {self.config_versions_table}
                    (config_key, config_value, version_number, description, 
                     created_by, created_at, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                ''', (config_key, config_value_json, new_version, description,
                      updated_by, current_time))
                
                # Record history
                cursor.execute(f'''
                    INSERT INTO {self.config_history_table}
                    (config_key, old_value, new_value, changed_by, change_reason, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (config_key, json.dumps(old_value) if old_value else None,
                      config_value_json, updated_by, change_reason, current_time))
                
                self.updates_executed += 1
                logger.info(f"Configuration updated: {config_key} (version {new_version})")
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to set configuration {config_key}: {e}")
            return False
    
    def get_config(self, config_key: str, default: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Get current configuration value.
        
        Args:
            config_key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value dictionary or default
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute(f'''
                    SELECT config_value FROM {self.config_table}
                    WHERE config_key = ?
                ''', (config_key,))
                
                row = cursor.fetchone()
                if row:
                    self.queries_executed += 1
                    return json.loads(row['config_value'])
                
                return default
                
        except Exception as e:
            logger.error(f"Failed to get configuration {config_key}: {e}")
            return default
    
    def get_all_configs(self, config_type: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get all configurations, optionally filtered by type.
        
        Args:
            config_type: Optional configuration type filter
            
        Returns:
            Dictionary of all configurations
        """
        try:
            with self.get_cursor() as cursor:
                if config_type:
                    cursor.execute(f'''
                        SELECT config_key, config_value FROM {self.config_table}
                        WHERE config_type = ?
                    ''', (config_type,))
                else:
                    cursor.execute(f'''
                        SELECT config_key, config_value FROM {self.config_table}
                    ''')
                
                configs = {}
                for row in cursor.fetchall():
                    configs[row['config_key']] = json.loads(row['config_value'])
                
                self.queries_executed += 1
                return configs
                
        except Exception as e:
            logger.error(f"Failed to get all configurations: {e}")
            return {}
    
    def get_config_version(self, config_key: str, version_number: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific version of a configuration.
        
        Args:
            config_key: Configuration key
            version_number: Version number to retrieve
            
        Returns:
            Configuration value for that version
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute(f'''
                    SELECT config_value FROM {self.config_versions_table}
                    WHERE config_key = ? AND version_number = ?
                ''', (config_key, version_number))
                
                row = cursor.fetchone()
                if row:
                    self.queries_executed += 1
                    return json.loads(row['config_value'])
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get configuration version: {e}")
            return None
    
    def get_version_history(self, config_key: str) -> List[ConfigVersion]:
        """
        Get version history for a configuration.
        
        Args:
            config_key: Configuration key
            
        Returns:
            List of configuration versions
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute(f'''
                    SELECT version_id, config_key, config_value, version_number,
                           description, created_by, created_at, is_active
                    FROM {self.config_versions_table}
                    WHERE config_key = ?
                    ORDER BY version_number DESC
                ''', (config_key,))
                
                versions = []
                for row in cursor.fetchall():
                    versions.append(ConfigVersion(
                        version_id=row['version_id'],
                        config_key=row['config_key'],
                        config_value=json.loads(row['config_value']),
                        description=row['description'] or "",
                        created_by=row['created_by'],
                        created_at=row['created_at'],
                        is_active=bool(row['is_active'])
                    ))
                
                self.queries_executed += 1
                return versions
                
        except Exception as e:
            logger.error(f"Failed to get version history: {e}")
            return []
    
    def get_change_history(self, config_key: Optional[str] = None,
                          limit: int = 100) -> List[ConfigHistory]:
        """
        Get configuration change history.
        
        Args:
            config_key: Optional key to filter by
            limit: Maximum number of entries to return
            
        Returns:
            List of configuration changes
        """
        try:
            with self.get_cursor() as cursor:
                if config_key:
                    cursor.execute(f'''
                        SELECT * FROM {self.config_history_table}
                        WHERE config_key = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ''', (config_key, limit))
                else:
                    cursor.execute(f'''
                        SELECT * FROM {self.config_history_table}
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ''', (limit,))
                
                history = []
                for row in cursor.fetchall():
                    history.append(ConfigHistory(
                        change_id=row['change_id'],
                        config_key=row['config_key'],
                        old_value=json.loads(row['old_value']) if row['old_value'] else None,
                        new_value=json.loads(row['new_value']),
                        changed_by=row['changed_by'],
                        change_reason=row['change_reason'] or "",
                        timestamp=row['timestamp']
                    ))
                
                self.queries_executed += 1
                return history
                
        except Exception as e:
            logger.error(f"Failed to get change history: {e}")
            return []
    
    def rollback_to_version(self, config_key: str, version_number: int,
                           rolled_back_by: str = "system") -> bool:
        """
        Rollback configuration to a previous version.
        
        Args:
            config_key: Configuration key
            version_number: Version to rollback to
            rolled_back_by: User performing rollback
            
        Returns:
            True if successful
        """
        try:
            # Get the version to rollback to
            version_value = self.get_config_version(config_key, version_number)
            if version_value is None:
                logger.error(f"Version {version_number} not found for {config_key}")
                return False
            
            # Set as current configuration
            return self.set_config(
                config_key=config_key,
                config_value=version_value,
                description=f"Rolled back to version {version_number}",
                updated_by=rolled_back_by,
                change_reason=f"Rollback to version {version_number}"
            )
            
        except Exception as e:
            logger.error(f"Failed to rollback configuration: {e}")
            return False
    
    def delete_config(self, config_key: str, deleted_by: str = "system") -> bool:
        """
        Delete a configuration (soft delete by marking as inactive).
        
        Args:
            config_key: Configuration key to delete
            deleted_by: User performing deletion
            
        Returns:
            True if successful
        """
        try:
            with self.get_cursor() as cursor:
                # Record deletion in history
                cursor.execute(f'''
                    SELECT config_value FROM {self.config_table}
                    WHERE config_key = ?
                ''', (config_key,))
                
                row = cursor.fetchone()
                if row:
                    cursor.execute(f'''
                        INSERT INTO {self.config_history_table}
                        (config_key, old_value, new_value, changed_by, 
                         change_reason, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (config_key, row['config_value'], 'null', deleted_by,
                          "Configuration deleted", time.time()))
                
                # Delete from main table
                cursor.execute(f'''
                    DELETE FROM {self.config_table}
                    WHERE config_key = ?
                ''', (config_key,))
                
                # Mark versions as inactive
                cursor.execute(f'''
                    UPDATE {self.config_versions_table}
                    SET is_active = 0
                    WHERE config_key = ?
                ''', (config_key,))
                
                self.deletes_executed += 1
                logger.info(f"Configuration deleted: {config_key}")
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete configuration: {e}")
            return False
