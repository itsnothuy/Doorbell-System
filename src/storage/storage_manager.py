#!/usr/bin/env python3
"""
Storage Manager - Central Storage Coordination

Manages all storage databases and provides unified interface for
storage operations, health monitoring, and maintenance tasks.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.storage.base_storage import DatabaseConfig, DatabaseEngine, StorageHealthStatus
from src.storage.event_database import EventDatabase
from src.storage.face_database import FaceDatabase
from src.storage.metrics_database import MetricsDatabase
from src.storage.config_database import ConfigDatabase

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """Configuration for storage manager."""
    data_directory: str = "data"
    
    # Event database config
    event_db_path: str = "data/events.db"
    event_db_wal_mode: bool = True
    event_retention_days: int = 90
    
    # Face database config
    face_db_path: str = "data/faces.db"
    face_db_wal_mode: bool = True
    max_faces_per_person: int = 10
    
    # Metrics database config
    metrics_db_path: str = "data/metrics.db"
    metrics_db_wal_mode: bool = True
    metrics_retention_days: int = 30
    
    # Config database config
    config_db_path: str = "data/config.db"
    config_db_wal_mode: bool = True
    
    # General settings
    connection_pool_size: int = 10
    auto_vacuum: bool = True
    backup_enabled: bool = True
    backup_directory: str = "data/backups"


@dataclass
class StorageMetrics:
    """Storage layer metrics."""
    total_queries: int = 0
    total_inserts: int = 0
    total_updates: int = 0
    total_deletes: int = 0
    total_errors: int = 0
    database_sizes: Dict[str, int] = field(default_factory=dict)
    health_status: Dict[str, bool] = field(default_factory=dict)


class StorageManager:
    """
    Central storage manager coordinating all databases.
    
    Provides unified interface for storage operations and manages
    lifecycle of all storage components.
    """
    
    def __init__(self, config: Optional[StorageConfig] = None):
        """
        Initialize storage manager.
        
        Args:
            config: Storage configuration (uses defaults if None)
        """
        self.config = config or StorageConfig()
        
        # Ensure directories exist
        Path(self.config.data_directory).mkdir(parents=True, exist_ok=True)
        if self.config.backup_enabled:
            Path(self.config.backup_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize database configurations
        self.event_db_config = DatabaseConfig(
            database_path=self.config.event_db_path,
            engine=DatabaseEngine.SQLITE,
            connection_pool_size=self.config.connection_pool_size,
            wal_mode=self.config.event_db_wal_mode,
            auto_vacuum=self.config.auto_vacuum
        )
        
        self.face_db_config = {
            'max_faces_per_person': self.config.max_faces_per_person,
            'backup_enabled': self.config.backup_enabled,
            'wal_mode': self.config.face_db_wal_mode
        }
        
        self.metrics_db_config = DatabaseConfig(
            database_path=self.config.metrics_db_path,
            engine=DatabaseEngine.SQLITE,
            connection_pool_size=self.config.connection_pool_size,
            wal_mode=self.config.metrics_db_wal_mode,
            auto_vacuum=self.config.auto_vacuum
        )
        
        self.config_db_config = DatabaseConfig(
            database_path=self.config.config_db_path,
            engine=DatabaseEngine.SQLITE,
            connection_pool_size=self.config.connection_pool_size,
            wal_mode=self.config.config_db_wal_mode,
            auto_vacuum=self.config.auto_vacuum
        )
        
        # Database instances
        self.event_db: Optional[EventDatabase] = None
        self.face_db: Optional[FaceDatabase] = None
        self.metrics_db: Optional[MetricsDatabase] = None
        self.config_db: Optional[ConfigDatabase] = None
        
        self._initialized = False
        
        logger.info("Storage manager created")
    
    def initialize(self) -> bool:
        """
        Initialize all storage components.
        
        Returns:
            True if all components initialized successfully
        """
        if self._initialized:
            logger.warning("Storage manager already initialized")
            return True
        
        try:
            logger.info("Initializing storage manager...")
            
            # Initialize event database
            self.event_db = EventDatabase(
                self.config.event_db_path,
                vars(self.event_db_config)
            )
            self.event_db.initialize()
            logger.info("✓ Event database initialized")
            
            # Initialize face database
            self.face_db = FaceDatabase(
                self.config.face_db_path,
                self.face_db_config
            )
            self.face_db.initialize()
            logger.info("✓ Face database initialized")
            
            # Initialize metrics database
            self.metrics_db = MetricsDatabase(self.metrics_db_config)
            self.metrics_db.initialize()
            logger.info("✓ Metrics database initialized")
            
            # Initialize config database
            self.config_db = ConfigDatabase(self.config_db_config)
            self.config_db.initialize()
            logger.info("✓ Config database initialized")
            
            self._initialized = True
            logger.info("Storage manager initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Storage manager initialization failed: {e}", exc_info=True)
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive storage health check.
        
        Returns:
            Health status for all storage components
        """
        if not self._initialized:
            return {
                'initialized': False,
                'healthy': False,
                'error': 'Storage manager not initialized'
            }
        
        health = {
            'initialized': True,
            'healthy': True,
            'databases': {}
        }
        
        # Check event database (may not have health_check yet)
        if self.event_db:
            if hasattr(self.event_db, 'health_check'):
                event_health = self.event_db.health_check()
                health['databases']['event_db'] = {
                    'healthy': event_health.is_healthy,
                    'status': event_health.status.name,
                    'size_bytes': event_health.database_size,
                    'metrics': event_health.metrics
                }
                if not event_health.is_healthy:
                    health['healthy'] = False
            else:
                # Basic check for event database
                health['databases']['event_db'] = {
                    'healthy': self.event_db.conn is not None,
                    'initialized': True
                }
        
        # Check face database
        if self.face_db:
            # Face database doesn't have health_check yet, use basic check
            face_healthy = self.face_db.is_initialized()
            health['databases']['face_db'] = {
                'healthy': face_healthy,
                'initialized': face_healthy
            }
            if not face_healthy:
                health['healthy'] = False
        
        # Check metrics database
        if self.metrics_db:
            metrics_health = self.metrics_db.health_check()
            health['databases']['metrics_db'] = {
                'healthy': metrics_health.is_healthy,
                'status': metrics_health.status.name,
                'size_bytes': metrics_health.database_size,
                'metrics': metrics_health.metrics
            }
            if not metrics_health.is_healthy:
                health['healthy'] = False
        
        # Check config database
        if self.config_db:
            config_health = self.config_db.health_check()
            health['databases']['config_db'] = {
                'healthy': config_health.is_healthy,
                'status': config_health.status.name,
                'size_bytes': config_health.database_size,
                'metrics': config_health.metrics
            }
            if not config_health.is_healthy:
                health['healthy'] = False
        
        return health
    
    def get_metrics(self) -> StorageMetrics:
        """
        Get comprehensive storage metrics.
        
        Returns:
            Storage metrics across all databases
        """
        metrics = StorageMetrics()
        
        if not self._initialized:
            return metrics
        
        # Aggregate metrics from all databases that have get_metrics
        for db_name, db in [
            ('event_db', self.event_db),
            ('metrics_db', self.metrics_db),
            ('config_db', self.config_db)
        ]:
            if db and hasattr(db, 'get_metrics'):
                db_metrics = db.get_metrics()
                metrics.total_queries += db_metrics.get('queries_executed', 0)
                metrics.total_inserts += db_metrics.get('inserts_executed', 0)
                metrics.total_updates += db_metrics.get('updates_executed', 0)
                metrics.total_deletes += db_metrics.get('deletes_executed', 0)
                metrics.total_errors += db_metrics.get('query_errors', 0)
                
                # Get database size
                db_path = Path(db_metrics.get('database_path', ''))
                if db_path.exists():
                    metrics.database_sizes[db_name] = db_path.stat().st_size
            elif db and hasattr(db, 'db_path'):
                # For databases without get_metrics, just get size
                db_path = Path(db.db_path) if hasattr(db, 'db_path') else None
                if db_path and db_path.exists():
                    metrics.database_sizes[db_name] = db_path.stat().st_size
        
        # Get face database size separately
        if self.face_db:
            face_db_path = Path(self.face_db.db_path)
            if face_db_path.exists():
                metrics.database_sizes['face_db'] = face_db_path.stat().st_size
        
        return metrics
    
    def cleanup_old_data(self) -> Dict[str, int]:
        """
        Clean up old data according to retention policies.
        
        Returns:
            Dictionary of deleted counts per database
        """
        if not self._initialized:
            logger.warning("Cannot cleanup: Storage manager not initialized")
            return {}
        
        deleted_counts = {}
        
        # Cleanup event database
        if self.event_db:
            try:
                deleted = self.event_db.cleanup_old_events(self.config.event_retention_days)
                deleted_counts['events'] = deleted
                logger.info(f"Cleaned up {deleted} old events")
            except Exception as e:
                logger.error(f"Event cleanup failed: {e}")
                deleted_counts['events'] = 0
        
        # Cleanup metrics database
        if self.metrics_db:
            try:
                deleted = self.metrics_db.cleanup_old_metrics(self.config.metrics_retention_days)
                deleted_counts['metrics'] = deleted
                logger.info(f"Cleaned up {deleted} old metrics")
            except Exception as e:
                logger.error(f"Metrics cleanup failed: {e}")
                deleted_counts['metrics'] = 0
        
        return deleted_counts
    
    def vacuum_all(self) -> Dict[str, bool]:
        """
        Vacuum all databases to reclaim space.
        
        Returns:
            Dictionary of vacuum results per database
        """
        if not self._initialized:
            logger.warning("Cannot vacuum: Storage manager not initialized")
            return {}
        
        results = {}
        
        for db_name, db in [
            ('event_db', self.event_db),
            ('metrics_db', self.metrics_db),
            ('config_db', self.config_db)
        ]:
            if db:
                try:
                    result = db.vacuum()
                    results[db_name] = result
                except Exception as e:
                    logger.error(f"Vacuum failed for {db_name}: {e}")
                    results[db_name] = False
        
        return results
    
    def close(self) -> None:
        """Close all database connections."""
        if not self._initialized:
            return
        
        logger.info("Closing storage manager...")
        
        if self.event_db:
            self.event_db.close()
        
        if self.face_db:
            # Face database doesn't have close() yet
            if hasattr(self.face_db, 'conn') and self.face_db.conn:
                self.face_db.conn.close()
        
        if self.metrics_db:
            self.metrics_db.close()
        
        if self.config_db:
            self.config_db.close()
        
        self._initialized = False
        logger.info("Storage manager closed")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
