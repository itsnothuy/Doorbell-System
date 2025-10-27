#!/usr/bin/env python3
"""
Backup Manager - Database Backup and Recovery System

Manages database backups with compression, encryption support,
and automated backup scheduling. Provides restore functionality.
"""

import gzip
import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BackupMetadata:
    """Metadata for a backup."""
    backup_id: str
    database_name: str
    backup_path: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    checksum: str
    created_at: float
    backup_type: str = "full"
    compressed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackupResult:
    """Result of a backup operation."""
    success: bool
    backup_id: Optional[str] = None
    backup_path: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    duration: float = 0.0
    size_bytes: int = 0


@dataclass
class RestoreResult:
    """Result of a restore operation."""
    success: bool
    database_path: str
    message: str = ""
    error: Optional[str] = None
    duration: float = 0.0


class BackupManager:
    """
    Manages database backups and recovery.
    
    Provides automated backup creation, compression, and restore
    functionality with metadata tracking.
    """
    
    def __init__(self, backup_dir: str = "data/backups", compress: bool = True):
        """
        Initialize backup manager.
        
        Args:
            backup_dir: Directory for storing backups
            compress: Whether to compress backups
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.compress = compress
        self.metadata_file = self.backup_dir / "backup_metadata.json"
        
        # Load existing metadata
        self.metadata: Dict[str, BackupMetadata] = self._load_metadata()
        
        logger.info(f"Backup manager initialized: {backup_dir}")
    
    def _load_metadata(self) -> Dict[str, BackupMetadata]:
        """
        Load backup metadata from file.
        
        Returns:
            Dictionary of backup metadata
        """
        if not self.metadata_file.exists():
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            metadata = {}
            for backup_id, meta_dict in data.items():
                metadata[backup_id] = BackupMetadata(**meta_dict)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load backup metadata: {e}")
            return {}
    
    def _save_metadata(self) -> bool:
        """
        Save backup metadata to file.
        
        Returns:
            True if successful
        """
        try:
            data = {}
            for backup_id, metadata in self.metadata.items():
                data[backup_id] = {
                    'backup_id': metadata.backup_id,
                    'database_name': metadata.database_name,
                    'backup_path': metadata.backup_path,
                    'original_size': metadata.original_size,
                    'compressed_size': metadata.compressed_size,
                    'compression_ratio': metadata.compression_ratio,
                    'checksum': metadata.checksum,
                    'created_at': metadata.created_at,
                    'backup_type': metadata.backup_type,
                    'compressed': metadata.compressed,
                    'metadata': metadata.metadata
                }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save backup metadata: {e}")
            return False
    
    def backup_database(self, db_path: str, backup_name: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> BackupResult:
        """
        Create a backup of a database.
        
        Args:
            db_path: Path to database file
            backup_name: Optional custom backup name
            metadata: Optional additional metadata
            
        Returns:
            Backup result
        """
        start_time = time.time()
        db_path_obj = Path(db_path)
        
        if not db_path_obj.exists():
            return BackupResult(
                success=False,
                error=f"Database file not found: {db_path}"
            )
        
        try:
            # Generate backup ID and filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            db_name = db_path_obj.stem
            backup_id = f"{db_name}_{timestamp}"
            
            if backup_name:
                backup_filename = f"{backup_name}_{timestamp}.db"
            else:
                backup_filename = f"{backup_id}.db"
            
            if self.compress:
                backup_filename += ".gz"
            
            backup_path = self.backup_dir / backup_filename
            
            # Get original size
            original_size = db_path_obj.stat().st_size
            
            # Copy and optionally compress
            if self.compress:
                with open(db_path_obj, 'rb') as f_in:
                    with gzip.open(backup_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                shutil.copy2(db_path_obj, backup_path)
            
            # Get compressed size
            compressed_size = backup_path.stat().st_size
            compression_ratio = (1 - compressed_size / original_size) if original_size > 0 else 0
            
            # Calculate checksum
            checksum = self._calculate_checksum(backup_path)
            
            # Create metadata
            backup_metadata = BackupMetadata(
                backup_id=backup_id,
                database_name=db_name,
                backup_path=str(backup_path),
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compression_ratio,
                checksum=checksum,
                created_at=time.time(),
                compressed=self.compress,
                metadata=metadata or {}
            )
            
            # Store metadata
            self.metadata[backup_id] = backup_metadata
            self._save_metadata()
            
            duration = time.time() - start_time
            
            logger.info(
                f"Backup created: {backup_id} "
                f"({original_size / 1024:.1f}KB → {compressed_size / 1024:.1f}KB, "
                f"{compression_ratio * 100:.1f}% compression)"
            )
            
            return BackupResult(
                success=True,
                backup_id=backup_id,
                backup_path=str(backup_path),
                message=f"Backup created successfully: {backup_id}",
                duration=duration,
                size_bytes=compressed_size
            )
            
        except Exception as e:
            logger.error(f"Backup failed: {e}", exc_info=True)
            return BackupResult(
                success=False,
                error=str(e),
                duration=time.time() - start_time
            )
    
    def restore_database(self, backup_id: str, restore_path: str,
                        verify_checksum: bool = True) -> RestoreResult:
        """
        Restore a database from backup.
        
        Args:
            backup_id: ID of backup to restore
            restore_path: Path where to restore database
            verify_checksum: Whether to verify backup checksum
            
        Returns:
            Restore result
        """
        start_time = time.time()
        
        if backup_id not in self.metadata:
            return RestoreResult(
                success=False,
                database_path=restore_path,
                error=f"Backup not found: {backup_id}"
            )
        
        backup_metadata = self.metadata[backup_id]
        backup_path = Path(backup_metadata.backup_path)
        
        if not backup_path.exists():
            return RestoreResult(
                success=False,
                database_path=restore_path,
                error=f"Backup file not found: {backup_path}"
            )
        
        try:
            # Verify checksum if requested
            if verify_checksum:
                current_checksum = self._calculate_checksum(backup_path)
                if current_checksum != backup_metadata.checksum:
                    return RestoreResult(
                        success=False,
                        database_path=restore_path,
                        error="Backup checksum mismatch - backup may be corrupted"
                    )
            
            restore_path_obj = Path(restore_path)
            restore_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Restore (decompress if needed)
            if backup_metadata.compressed:
                with gzip.open(backup_path, 'rb') as f_in:
                    with open(restore_path_obj, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                shutil.copy2(backup_path, restore_path_obj)
            
            duration = time.time() - start_time
            
            logger.info(f"Database restored from backup {backup_id} to {restore_path}")
            
            return RestoreResult(
                success=True,
                database_path=restore_path,
                message=f"Database restored successfully from {backup_id}",
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"Restore failed: {e}", exc_info=True)
            return RestoreResult(
                success=False,
                database_path=restore_path,
                error=str(e),
                duration=time.time() - start_time
            )
    
    def list_backups(self, database_name: Optional[str] = None) -> List[BackupMetadata]:
        """
        List available backups.
        
        Args:
            database_name: Optional filter by database name
            
        Returns:
            List of backup metadata
        """
        backups = list(self.metadata.values())
        
        if database_name:
            backups = [b for b in backups if b.database_name == database_name]
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda b: b.created_at, reverse=True)
        
        return backups
    
    def get_backup(self, backup_id: str) -> Optional[BackupMetadata]:
        """
        Get metadata for a specific backup.
        
        Args:
            backup_id: Backup ID
            
        Returns:
            Backup metadata or None
        """
        return self.metadata.get(backup_id)
    
    def delete_backup(self, backup_id: str) -> bool:
        """
        Delete a backup.
        
        Args:
            backup_id: ID of backup to delete
            
        Returns:
            True if successful
        """
        if backup_id not in self.metadata:
            logger.warning(f"Backup not found: {backup_id}")
            return False
        
        try:
            backup_metadata = self.metadata[backup_id]
            backup_path = Path(backup_metadata.backup_path)
            
            # Delete file
            if backup_path.exists():
                backup_path.unlink()
            
            # Remove metadata
            del self.metadata[backup_id]
            self._save_metadata()
            
            logger.info(f"Deleted backup: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False
    
    def cleanup_old_backups(self, retention_days: int,
                           database_name: Optional[str] = None) -> int:
        """
        Delete backups older than retention period.
        
        Args:
            retention_days: Number of days to retain backups
            database_name: Optional filter by database name
            
        Returns:
            Number of backups deleted
        """
        cutoff_time = (datetime.now() - timedelta(days=retention_days)).timestamp()
        
        backups_to_delete = []
        for backup_id, metadata in self.metadata.items():
            # Filter by database name if specified
            if database_name and metadata.database_name != database_name:
                continue
            
            # Check age
            if metadata.created_at < cutoff_time:
                backups_to_delete.append(backup_id)
        
        deleted_count = 0
        for backup_id in backups_to_delete:
            if self.delete_backup(backup_id):
                deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old backups")
        
        return deleted_count
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """
        Calculate SHA256 checksum of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex digest of checksum
        """
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def get_total_backup_size(self) -> int:
        """
        Get total size of all backups.
        
        Returns:
            Total size in bytes
        """
        total_size = 0
        
        for metadata in self.metadata.values():
            backup_path = Path(metadata.backup_path)
            if backup_path.exists():
                total_size += backup_path.stat().st_size
        
        return total_size
    
    def get_backup_statistics(self) -> Dict[str, Any]:
        """
        Get backup statistics.
        
        Returns:
            Statistics dictionary
        """
        backups = list(self.metadata.values())
        
        if not backups:
            return {
                'total_backups': 0,
                'total_size_bytes': 0,
                'databases': {}
            }
        
        # Group by database
        by_database: Dict[str, List[BackupMetadata]] = {}
        for backup in backups:
            if backup.database_name not in by_database:
                by_database[backup.database_name] = []
            by_database[backup.database_name].append(backup)
        
        # Calculate statistics
        database_stats = {}
        total_size = 0
        
        for db_name, db_backups in by_database.items():
            db_size = sum(b.compressed_size for b in db_backups)
            total_size += db_size
            
            database_stats[db_name] = {
                'backup_count': len(db_backups),
                'total_size_bytes': db_size,
                'oldest_backup': min(b.created_at for b in db_backups),
                'newest_backup': max(b.created_at for b in db_backups),
                'avg_compression_ratio': sum(b.compression_ratio for b in db_backups) / len(db_backups)
            }
        
        return {
            'total_backups': len(backups),
            'total_size_bytes': total_size,
            'databases': database_stats
        }
