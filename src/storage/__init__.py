#!/usr/bin/env python3
"""
Storage Module

Provides comprehensive database management for events, face encodings,
metrics, and configuration with centralized storage coordination.
"""

from src.storage.base_storage import (
    BaseDatabase,
    DatabaseConfig,
    DatabaseEngine,
    StorageStatus,
    StorageHealthStatus,
    QueryResult
)
from src.storage.event_database import EventDatabase
from src.storage.face_database import FaceDatabase
from src.storage.metrics_database import MetricsDatabase, SystemMetric, MetricType
from src.storage.config_database import ConfigDatabase
from src.storage.storage_manager import StorageManager, StorageConfig
from src.storage.migration_manager import MigrationManager, Migration, MigrationResult
from src.storage.backup_manager import BackupManager, BackupResult, RestoreResult
from src.storage.query_builder import QueryBuilder, Operator, SortOrder

__all__ = [
    # Base classes
    'BaseDatabase',
    'DatabaseConfig',
    'DatabaseEngine',
    'StorageStatus',
    'StorageHealthStatus',
    'QueryResult',
    
    # Database implementations
    'EventDatabase',
    'FaceDatabase',
    'MetricsDatabase',
    'ConfigDatabase',
    
    # Metrics types
    'SystemMetric',
    'MetricType',
    
    # Storage manager
    'StorageManager',
    'StorageConfig',
    
    # Management components
    'MigrationManager',
    'Migration',
    'MigrationResult',
    'BackupManager',
    'BackupResult',
    'RestoreResult',
    
    # Query utilities
    'QueryBuilder',
    'Operator',
    'SortOrder',
]
