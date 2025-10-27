#!/usr/bin/env python3
"""
Base Storage - Abstract Interfaces for Storage Layer

Provides abstract base classes and interfaces for consistent database
implementations across the storage layer.
"""

import logging
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class DatabaseEngine(Enum):
    """Supported database engines."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


class StorageStatus(Enum):
    """Storage component status."""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    READY = auto()
    ERROR = auto()
    CLOSED = auto()


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""
    database_path: str
    engine: DatabaseEngine = DatabaseEngine.SQLITE
    connection_pool_size: int = 5
    wal_mode: bool = True
    auto_vacuum: bool = True
    batch_insert_size: int = 100
    query_timeout: int = 30
    enable_foreign_keys: bool = True
    cache_size_kb: int = 64000
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Result from a database query."""
    success: bool
    rows: List[Dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class StorageHealthStatus:
    """Health status of storage component."""
    status: StorageStatus
    is_healthy: bool
    database_size: int = 0
    connection_count: int = 0
    last_error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class BaseDatabase(ABC):
    """
    Abstract base class for database implementations.
    
    Provides common functionality for database operations with
    consistent error handling and connection management.
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize database with configuration.
        
        Args:
            config: Database configuration
        """
        self.config = config
        self.db_path = Path(config.database_path)
        self.conn: Optional[sqlite3.Connection] = None
        self.status = StorageStatus.UNINITIALIZED
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Performance metrics
        self.queries_executed = 0
        self.inserts_executed = 0
        self.updates_executed = 0
        self.deletes_executed = 0
        self.query_errors = 0
        
        logger.info(f"Initialized {self.__class__.__name__} with path: {config.database_path}")
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize database schema and connections.
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    def _create_tables(self) -> None:
        """Create database tables for this storage component."""
        pass
    
    def _configure_database(self) -> None:
        """Configure SQLite performance and reliability settings."""
        if not self.conn:
            raise RuntimeError("Database connection not established")
        
        cursor = self.conn.cursor()
        
        # Enable foreign keys
        if self.config.enable_foreign_keys:
            cursor.execute("PRAGMA foreign_keys=ON")
        
        # Enable Write-Ahead Logging for better concurrency
        if self.config.wal_mode:
            cursor.execute("PRAGMA journal_mode=WAL")
            logger.debug(f"Enabled WAL mode for {self.__class__.__name__}")
        
        # Enable auto-vacuum
        if self.config.auto_vacuum:
            cursor.execute("PRAGMA auto_vacuum=INCREMENTAL")
        
        # Performance optimizations
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute(f"PRAGMA cache_size=-{self.config.cache_size_kb}")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.execute(f"PRAGMA busy_timeout={self.config.query_timeout * 1000}")
        
        self.conn.commit()
        cursor.close()
    
    @contextmanager
    def get_connection(self):
        """
        Get database connection with context manager.
        
        Yields:
            Database connection
        """
        if not self.conn:
            raise RuntimeError("Database not initialized")
        
        try:
            yield self.conn
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
    
    @contextmanager
    def get_cursor(self):
        """
        Get database cursor with context manager.
        
        Yields:
            Database cursor
        """
        if not self.conn:
            raise RuntimeError("Database not initialized")
        
        cursor = self.conn.cursor()
        try:
            yield cursor
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            self.query_errors += 1
            logger.error(f"Database query failed: {e}")
            raise
        finally:
            cursor.close()
    
    def health_check(self) -> StorageHealthStatus:
        """
        Perform health check on database.
        
        Returns:
            Health status information
        """
        try:
            if not self.conn:
                return StorageHealthStatus(
                    status=StorageStatus.UNINITIALIZED,
                    is_healthy=False,
                    last_error="Database not initialized"
                )
            
            # Test query
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            
            # Get database size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            metrics = {
                'queries_executed': self.queries_executed,
                'inserts_executed': self.inserts_executed,
                'updates_executed': self.updates_executed,
                'deletes_executed': self.deletes_executed,
                'query_errors': self.query_errors,
                'error_rate': self.query_errors / max(1, self.queries_executed)
            }
            
            return StorageHealthStatus(
                status=self.status,
                is_healthy=True,
                database_size=db_size,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return StorageHealthStatus(
                status=StorageStatus.ERROR,
                is_healthy=False,
                last_error=str(e)
            )
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            try:
                self.conn.close()
                self.conn = None
                self.status = StorageStatus.CLOSED
                logger.info(f"Closed {self.__class__.__name__}")
            except Exception as e:
                logger.error(f"Error closing database: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get database performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            'database_path': str(self.db_path),
            'status': self.status.name,
            'queries_executed': self.queries_executed,
            'inserts_executed': self.inserts_executed,
            'updates_executed': self.updates_executed,
            'deletes_executed': self.deletes_executed,
            'query_errors': self.query_errors,
            'error_rate': self.query_errors / max(1, self.queries_executed)
        }
    
    def vacuum(self) -> bool:
        """
        Vacuum database to reclaim space.
        
        Returns:
            True if successful
        """
        try:
            if not self.conn:
                return False
            
            cursor = self.conn.cursor()
            cursor.execute("VACUUM")
            self.conn.commit()
            cursor.close()
            
            logger.info(f"Vacuumed {self.__class__.__name__}")
            return True
            
        except Exception as e:
            logger.error(f"Vacuum failed: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
