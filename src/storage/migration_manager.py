#!/usr/bin/env python3
"""
Migration Manager - Database Schema Migration System

Handles database schema migrations with version control, rollback capability,
and transactional safety. Supports both forward and backward migrations.
"""

import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Migration:
    """Represents a database migration."""
    version: str
    name: str
    description: str
    up_statements: List[str]
    down_statements: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    applied_at: Optional[float] = None


@dataclass
class MigrationResult:
    """Result of a migration operation."""
    success: bool
    message: str = ""
    migrations_applied: List[str] = field(default_factory=list)
    migrations_rolled_back: List[str] = field(default_factory=list)
    error: Optional[str] = None
    duration: float = 0.0


class MigrationManager:
    """
    Manages database schema migrations.
    
    Provides version control for database schemas with support for
    forward migrations, rollbacks, and dependency management.
    """
    
    def __init__(self, db_path: str, migrations_dir: Optional[str] = None):
        """
        Initialize migration manager.
        
        Args:
            db_path: Path to database file
            migrations_dir: Directory containing migration files
        """
        self.db_path = Path(db_path)
        self.migrations_dir = Path(migrations_dir) if migrations_dir else Path("src/storage/migrations")
        self.migration_table = "schema_migrations"
        
        # Ensure migrations directory exists
        self.migrations_dir.mkdir(parents=True, exist_ok=True)
        
        # Connection
        self.conn: Optional[sqlite3.Connection] = None
        
        logger.info(f"Migration manager initialized for {db_path}")
    
    def initialize(self) -> bool:
        """
        Initialize migration tracking.
        
        Returns:
            True if successful
        """
        try:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.row_factory = sqlite3.Row
            
            # Create migration tracking table
            self._create_migration_table()
            
            logger.info("Migration tracking initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize migration tracking: {e}")
            return False
    
    def _create_migration_table(self) -> None:
        """Create table for tracking applied migrations."""
        cursor = self.conn.cursor()
        
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.migration_table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                applied_at REAL NOT NULL,
                applied_by TEXT DEFAULT 'system',
                duration_ms REAL,
                checksum TEXT
            )
        ''')
        
        # Create index on version
        cursor.execute(f'''
            CREATE INDEX IF NOT EXISTS idx_migration_version 
            ON {self.migration_table}(version)
        ''')
        
        self.conn.commit()
        cursor.close()
    
    def get_applied_migrations(self) -> List[str]:
        """
        Get list of applied migration versions.
        
        Returns:
            List of migration versions that have been applied
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(f'''
                SELECT version FROM {self.migration_table}
                ORDER BY applied_at ASC
            ''')
            
            versions = [row['version'] for row in cursor.fetchall()]
            cursor.close()
            
            return versions
            
        except Exception as e:
            logger.error(f"Failed to get applied migrations: {e}")
            return []
    
    def get_pending_migrations(self) -> List[Migration]:
        """
        Get list of pending migrations.
        
        Returns:
            List of migrations that haven't been applied yet
        """
        applied_versions = set(self.get_applied_migrations())
        available_migrations = self._load_migrations_from_directory()
        
        pending = [
            m for m in available_migrations
            if m.version not in applied_versions
        ]
        
        # Sort by version
        pending.sort(key=lambda m: m.version)
        
        return pending
    
    def _load_migrations_from_directory(self) -> List[Migration]:
        """
        Load migration definitions from files.
        
        Returns:
            List of available migrations
        """
        migrations = []
        
        # Look for .sql migration files
        for sql_file in sorted(self.migrations_dir.glob("*.sql")):
            try:
                migration = self._parse_migration_file(sql_file)
                if migration:
                    migrations.append(migration)
            except Exception as e:
                logger.warning(f"Failed to parse migration file {sql_file}: {e}")
        
        return migrations
    
    def _parse_migration_file(self, file_path: Path) -> Optional[Migration]:
        """
        Parse a migration SQL file.
        
        Expected format:
        -- version: 001
        -- name: initial_schema
        -- description: Create initial database schema
        
        -- up
        CREATE TABLE ...;
        
        -- down
        DROP TABLE ...;
        
        Args:
            file_path: Path to migration file
            
        Returns:
            Migration object or None if parsing fails
        """
        content = file_path.read_text()
        
        # Extract metadata from comments
        version_match = re.search(r'--\s*version:\s*(\S+)', content, re.IGNORECASE)
        name_match = re.search(r'--\s*name:\s*(.+)', content, re.IGNORECASE)
        desc_match = re.search(r'--\s*description:\s*(.+)', content, re.IGNORECASE)
        
        if not (version_match and name_match):
            logger.warning(f"Migration file {file_path} missing required metadata")
            return None
        
        version = version_match.group(1).strip()
        name = name_match.group(1).strip()
        description = desc_match.group(1).strip() if desc_match else ""
        
        # Split into up and down sections
        up_match = re.search(r'--\s*up\s*\n(.*?)(?:--\s*down|$)', content, re.DOTALL | re.IGNORECASE)
        down_match = re.search(r'--\s*down\s*\n(.*?)$', content, re.DOTALL | re.IGNORECASE)
        
        if not up_match:
            logger.warning(f"Migration file {file_path} missing 'up' section")
            return None
        
        # Parse SQL statements
        up_statements = self._parse_sql_statements(up_match.group(1))
        down_statements = self._parse_sql_statements(down_match.group(1)) if down_match else []
        
        return Migration(
            version=version,
            name=name,
            description=description,
            up_statements=up_statements,
            down_statements=down_statements
        )
    
    def _parse_sql_statements(self, sql_text: str) -> List[str]:
        """
        Parse SQL text into individual statements.
        
        Args:
            sql_text: SQL text with multiple statements
            
        Returns:
            List of SQL statements
        """
        # Remove comments
        lines = []
        for line in sql_text.split('\n'):
            # Remove line comments
            line = re.sub(r'--.*$', '', line)
            if line.strip():
                lines.append(line)
        
        sql_text = '\n'.join(lines)
        
        # Split by semicolon (basic parsing, doesn't handle all edge cases)
        statements = []
        current = []
        
        for line in sql_text.split('\n'):
            current.append(line)
            if ';' in line:
                stmt = '\n'.join(current).strip()
                if stmt:
                    statements.append(stmt)
                current = []
        
        # Add any remaining statement
        if current:
            stmt = '\n'.join(current).strip()
            if stmt and not stmt.startswith('--'):
                statements.append(stmt)
        
        return statements
    
    def run_migrations(self, target_version: Optional[str] = None) -> MigrationResult:
        """
        Run pending migrations up to target version.
        
        Args:
            target_version: Optional target version (runs all if None)
            
        Returns:
            Migration result
        """
        if not self.conn:
            return MigrationResult(
                success=False,
                error="Migration manager not initialized"
            )
        
        start_time = time.time()
        pending = self.get_pending_migrations()
        
        if not pending:
            return MigrationResult(
                success=True,
                message="No pending migrations"
            )
        
        # Filter by target version if specified
        if target_version:
            pending = [m for m in pending if m.version <= target_version]
        
        applied = []
        
        try:
            for migration in pending:
                result = self._apply_migration(migration)
                if result.success:
                    applied.append(migration.version)
                    logger.info(f"Applied migration {migration.version}: {migration.name}")
                else:
                    return MigrationResult(
                        success=False,
                        message=f"Migration {migration.version} failed",
                        migrations_applied=applied,
                        error=result.error,
                        duration=time.time() - start_time
                    )
            
            duration = time.time() - start_time
            return MigrationResult(
                success=True,
                message=f"Successfully applied {len(applied)} migration(s)",
                migrations_applied=applied,
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"Migration failed: {e}", exc_info=True)
            return MigrationResult(
                success=False,
                message="Migration failed with exception",
                migrations_applied=applied,
                error=str(e),
                duration=time.time() - start_time
            )
    
    def _apply_migration(self, migration: Migration) -> MigrationResult:
        """
        Apply a single migration.
        
        Args:
            migration: Migration to apply
            
        Returns:
            Migration result
        """
        start_time = time.time()
        
        try:
            cursor = self.conn.cursor()
            
            # Begin transaction
            cursor.execute("BEGIN TRANSACTION")
            
            try:
                # Execute up statements
                for statement in migration.up_statements:
                    if statement.strip():
                        cursor.execute(statement)
                
                # Record migration
                cursor.execute(f'''
                    INSERT INTO {self.migration_table}
                    (version, name, description, applied_at, duration_ms)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    migration.version,
                    migration.name,
                    migration.description,
                    time.time(),
                    (time.time() - start_time) * 1000
                ))
                
                # Commit transaction
                cursor.execute("COMMIT")
                cursor.close()
                
                return MigrationResult(
                    success=True,
                    message=f"Applied migration {migration.version}",
                    duration=time.time() - start_time
                )
                
            except Exception as e:
                # Rollback on error
                cursor.execute("ROLLBACK")
                cursor.close()
                raise e
                
        except Exception as e:
            logger.error(f"Failed to apply migration {migration.version}: {e}")
            return MigrationResult(
                success=False,
                error=str(e),
                duration=time.time() - start_time
            )
    
    def rollback_migration(self, version: str) -> MigrationResult:
        """
        Rollback a specific migration.
        
        Args:
            version: Migration version to rollback
            
        Returns:
            Migration result
        """
        if not self.conn:
            return MigrationResult(
                success=False,
                error="Migration manager not initialized"
            )
        
        # Find the migration
        migrations = self._load_migrations_from_directory()
        migration = next((m for m in migrations if m.version == version), None)
        
        if not migration:
            return MigrationResult(
                success=False,
                error=f"Migration {version} not found"
            )
        
        if not migration.down_statements:
            return MigrationResult(
                success=False,
                error=f"Migration {version} has no rollback statements"
            )
        
        try:
            cursor = self.conn.cursor()
            
            # Begin transaction
            cursor.execute("BEGIN TRANSACTION")
            
            try:
                # Execute down statements
                for statement in migration.down_statements:
                    if statement.strip():
                        cursor.execute(statement)
                
                # Remove migration record
                cursor.execute(f'''
                    DELETE FROM {self.migration_table}
                    WHERE version = ?
                ''', (version,))
                
                # Commit transaction
                cursor.execute("COMMIT")
                cursor.close()
                
                logger.info(f"Rolled back migration {version}")
                
                return MigrationResult(
                    success=True,
                    message=f"Rolled back migration {version}",
                    migrations_rolled_back=[version]
                )
                
            except Exception as e:
                cursor.execute("ROLLBACK")
                cursor.close()
                raise e
                
        except Exception as e:
            logger.error(f"Failed to rollback migration {version}: {e}")
            return MigrationResult(
                success=False,
                error=str(e)
            )
    
    def get_migration_status(self) -> Dict[str, Any]:
        """
        Get current migration status.
        
        Returns:
            Status information
        """
        applied = self.get_applied_migrations()
        pending = self.get_pending_migrations()
        
        status = {
            'total_applied': len(applied),
            'total_pending': len(pending),
            'applied_versions': applied,
            'pending_versions': [m.version for m in pending],
            'current_version': applied[-1] if applied else None
        }
        
        return status
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
