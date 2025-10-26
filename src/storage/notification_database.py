#!/usr/bin/env python3
"""
Notification Database - Persistent Storage for Notification History

Provides persistent storage for notification history with SQLite backend.
"""

import sqlite3
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class NotificationDatabase:
    """Manages persistent storage of notification history."""
    
    def __init__(self, db_path: str = 'data/notifications.db'):
        """
        Initialize notification database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn: Optional[sqlite3.Connection] = None
        
        logger.info(f"Notification database initialized: {db_path}")
    
    def initialize(self):
        """Initialize database schema."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            
            self._create_tables()
            
            logger.info("Notification database schema created")
            
        except Exception as e:
            logger.error(f"Failed to initialize notification database: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables."""
        cursor = self.conn.cursor()
        
        # Notifications table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT UNIQUE NOT NULL,
                alert_type TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                content TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Indexes for common queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_notifications_timestamp 
            ON notifications(timestamp DESC)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_notifications_alert_type 
            ON notifications(alert_type)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_notifications_status 
            ON notifications(status)
        ''')
        
        self.conn.commit()
    
    @contextmanager
    def _get_cursor(self):
        """Get database cursor with context manager."""
        if not self.conn:
            raise RuntimeError("Database not initialized")
        
        cursor = self.conn.cursor()
        try:
            yield cursor
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e
        finally:
            cursor.close()
    
    def store_notification(self, notification_record: Dict[str, Any]) -> bool:
        """
        Store a notification record in the database.
        
        Args:
            notification_record: Notification data to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_cursor() as cursor:
                # Serialize content and metadata as JSON
                content_json = json.dumps(notification_record.get('content'))
                metadata_json = json.dumps(notification_record.get('metadata', {}))
                
                cursor.execute('''
                    INSERT INTO notifications 
                    (alert_id, alert_type, priority, status, timestamp, content, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    notification_record['alert_id'],
                    notification_record['alert_type'],
                    notification_record['priority'],
                    notification_record['status'],
                    notification_record['timestamp'],
                    content_json,
                    metadata_json
                ))
            
            logger.debug(f"Stored notification: {notification_record['alert_id']}")
            return True
            
        except sqlite3.IntegrityError:
            logger.warning(f"Duplicate notification: {notification_record['alert_id']}")
            return False
        except Exception as e:
            logger.error(f"Failed to store notification: {e}")
            return False
    
    def get_notifications(self, limit: int = 100, offset: int = 0,
                         alert_type: Optional[str] = None,
                         status: Optional[str] = None,
                         since: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve notifications from database with optional filters.
        
        Args:
            limit: Maximum number of notifications to return
            offset: Number of notifications to skip
            alert_type: Filter by alert type
            status: Filter by delivery status
            since: Filter by timestamp (ISO format)
            
        Returns:
            List of notification records
        """
        try:
            with self._get_cursor() as cursor:
                query = "SELECT * FROM notifications WHERE 1=1"
                params = []
                
                if alert_type:
                    query += " AND alert_type = ?"
                    params.append(alert_type)
                
                if status:
                    query += " AND status = ?"
                    params.append(status)
                
                if since:
                    query += " AND timestamp >= ?"
                    params.append(since)
                
                query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                notifications = []
                for row in rows:
                    notification = dict(row)
                    
                    # Deserialize JSON fields
                    if notification['content']:
                        notification['content'] = json.loads(notification['content'])
                    if notification['metadata']:
                        notification['metadata'] = json.loads(notification['metadata'])
                    
                    notifications.append(notification)
                
                return notifications
                
        except Exception as e:
            logger.error(f"Failed to retrieve notifications: {e}")
            return []
    
    def get_notification_by_alert_id(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific notification by alert ID.
        
        Args:
            alert_id: Alert ID to look up
            
        Returns:
            Notification record or None if not found
        """
        try:
            with self._get_cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM notifications WHERE alert_id = ?",
                    (alert_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    notification = dict(row)
                    
                    # Deserialize JSON fields
                    if notification['content']:
                        notification['content'] = json.loads(notification['content'])
                    if notification['metadata']:
                        notification['metadata'] = json.loads(notification['metadata'])
                    
                    return notification
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get notification: {e}")
            return None
    
    def get_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get notification statistics for the specified period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Statistics dictionary
        """
        try:
            since = (datetime.now() - timedelta(days=days)).isoformat()
            
            with self._get_cursor() as cursor:
                # Total notifications
                cursor.execute(
                    "SELECT COUNT(*) as count FROM notifications WHERE timestamp >= ?",
                    (since,)
                )
                total = cursor.fetchone()['count']
                
                # By alert type
                cursor.execute('''
                    SELECT alert_type, COUNT(*) as count 
                    FROM notifications 
                    WHERE timestamp >= ?
                    GROUP BY alert_type
                ''', (since,))
                by_type = {row['alert_type']: row['count'] for row in cursor.fetchall()}
                
                # By status
                cursor.execute('''
                    SELECT status, COUNT(*) as count 
                    FROM notifications 
                    WHERE timestamp >= ?
                    GROUP BY status
                ''', (since,))
                by_status = {row['status']: row['count'] for row in cursor.fetchall()}
                
                # By priority
                cursor.execute('''
                    SELECT priority, COUNT(*) as count 
                    FROM notifications 
                    WHERE timestamp >= ?
                    GROUP BY priority
                ''', (since,))
                by_priority = {row['priority']: row['count'] for row in cursor.fetchall()}
                
                return {
                    'total_notifications': total,
                    'by_alert_type': by_type,
                    'by_status': by_status,
                    'by_priority': by_priority,
                    'period_days': days
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def cleanup_old_notifications(self, retention_days: int = 30) -> int:
        """
        Delete notifications older than retention period.
        
        Args:
            retention_days: Number of days to retain notifications
            
        Returns:
            Number of notifications deleted
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=retention_days)).isoformat()
            
            with self._get_cursor() as cursor:
                cursor.execute(
                    "DELETE FROM notifications WHERE timestamp < ?",
                    (cutoff_date,)
                )
                deleted_count = cursor.rowcount
            
            logger.info(f"Cleaned up {deleted_count} old notifications (older than {retention_days} days)")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup notifications: {e}")
            return 0
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Notification database closed")
