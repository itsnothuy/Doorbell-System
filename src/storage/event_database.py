#!/usr/bin/env python3
"""
Event Database - Persistent Storage for Security Events

Provides comprehensive event storage, querying, and management with SQLite backend.
Supports full event lifecycle tracking, enrichment data, and performance optimization.
"""

import sqlite3
import json
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager
from dataclasses import asdict

logger = logging.getLogger(__name__)


class EventDatabase:
    """Manages persistent storage of security events."""
    
    def __init__(self, db_path: str = 'data/events.db', config: Optional[Dict[str, Any]] = None):
        """
        Initialize event database.
        
        Args:
            db_path: Path to SQLite database file
            config: Database configuration options
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.config = config or {}
        self.connection_pool_size = self.config.get('connection_pool_size', 5)
        self.wal_mode = self.config.get('wal_mode', True)
        self.auto_vacuum = self.config.get('auto_vacuum', True)
        self.batch_insert_size = self.config.get('batch_insert_size', 100)
        
        # Connection management
        self.conn: Optional[sqlite3.Connection] = None
        self.conn_lock = threading.RLock()
        
        # Statistics
        self.events_stored = 0
        self.storage_errors = 0
        self.queries_executed = 0
        
        logger.info(f"Event database initialized: {db_path}")
    
    def initialize(self) -> None:
        """Initialize database schema and configure SQLite."""
        try:
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            
            # Configure SQLite
            self._configure_database()
            
            # Create schema
            self._create_tables()
            
            logger.info("Event database schema created successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize event database: {e}")
            raise
    
    def _configure_database(self) -> None:
        """Configure SQLite performance and reliability settings."""
        cursor = self.conn.cursor()
        
        # Enable Write-Ahead Logging for better concurrency
        if self.wal_mode:
            cursor.execute("PRAGMA journal_mode=WAL")
            logger.debug("Enabled WAL mode for event database")
        
        # Enable auto-vacuum
        if self.auto_vacuum:
            cursor.execute("PRAGMA auto_vacuum=INCREMENTAL")
        
        # Performance optimizations
        cursor.execute("PRAGMA synchronous=NORMAL")  # Balance between safety and speed
        cursor.execute("PRAGMA cache_size=-64000")    # 64MB cache
        cursor.execute("PRAGMA temp_store=MEMORY")    # Use memory for temp tables
        
        self.conn.commit()
        cursor.close()
    
    def _create_tables(self) -> None:
        """Create database tables for event storage."""
        cursor = self.conn.cursor()
        
        # Main events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL,
                event_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                source TEXT,
                priority TEXT,
                confidence REAL,
                correlation_id TEXT,
                parent_event_id TEXT,
                
                -- Event data (JSON)
                event_data TEXT,
                enriched_data TEXT,
                
                -- Processing metadata
                processing_stage TEXT,
                enrichment_count INTEGER DEFAULT 0,
                error_count INTEGER DEFAULT 0,
                
                -- Timestamps
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Indexes
                FOREIGN KEY (parent_event_id) REFERENCES events(event_id)
            )
        ''')
        
        # Face recognition events table (for detailed face data)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_recognition_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                recognition_count INTEGER DEFAULT 0,
                known_count INTEGER DEFAULT 0,
                unknown_count INTEGER DEFAULT 0,
                blacklisted_count INTEGER DEFAULT 0,
                recognition_time REAL DEFAULT 0,
                recognition_data TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE
            )
        ''')
        
        # Event enrichment results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS event_enrichments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                processor_name TEXT NOT NULL,
                success INTEGER DEFAULT 1,
                processing_time REAL,
                enrichment_data TEXT,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE,
                UNIQUE(event_id, processor_name)
            )
        ''')
        
        # Event metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS event_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_unit TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE
            )
        ''')
        
        # Create indexes for performance
        self._create_indexes(cursor)
        
        self.conn.commit()
        cursor.close()
    
    def _create_indexes(self, cursor: sqlite3.Cursor) -> None:
        """Create database indexes for query performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type)",
            "CREATE INDEX IF NOT EXISTS idx_events_source ON events(source)",
            "CREATE INDEX IF NOT EXISTS idx_events_priority ON events(priority)",
            "CREATE INDEX IF NOT EXISTS idx_events_correlation_id ON events(correlation_id)",
            "CREATE INDEX IF NOT EXISTS idx_events_parent_event_id ON events(parent_event_id)",
            "CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at DESC)",
            
            "CREATE INDEX IF NOT EXISTS idx_face_recognition_event_id ON face_recognition_events(event_id)",
            "CREATE INDEX IF NOT EXISTS idx_enrichments_event_id ON event_enrichments(event_id)",
            "CREATE INDEX IF NOT EXISTS idx_enrichments_processor ON event_enrichments(processor_name)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_event_id ON event_metrics(event_id)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
    
    @contextmanager
    def _get_cursor(self):
        """Get database cursor with context manager."""
        if not self.conn:
            raise RuntimeError("Database not initialized")
        
        with self.conn_lock:
            cursor = self.conn.cursor()
            try:
                yield cursor
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                raise e
            finally:
                cursor.close()
    
    def store_event(self, event: Any, enrichment_results: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a processed event with enrichment data.
        
        Args:
            event: Event to store (PipelineEvent or subclass)
            enrichment_results: Dictionary of enrichment results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_cursor() as cursor:
                # Extract event data
                event_data = self._extract_event_data(event)
                
                # Serialize data as JSON
                event_data_json = json.dumps(event_data.get('data', {}))
                enriched_data_json = json.dumps(enrichment_results or {})
                
                # Calculate enrichment count
                enrichment_count = len(enrichment_results) if enrichment_results else 0
                
                # Insert main event record
                cursor.execute('''
                    INSERT INTO events 
                    (event_id, event_type, timestamp, source, priority, confidence,
                     correlation_id, parent_event_id, event_data, enriched_data,
                     processing_stage, enrichment_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event_data['event_id'],
                    event_data['event_type'],
                    event_data['timestamp'],
                    event_data.get('source'),
                    event_data.get('priority'),
                    event_data.get('confidence'),
                    event_data.get('correlation_id'),
                    event_data.get('parent_event_id'),
                    event_data_json,
                    enriched_data_json,
                    event_data.get('processing_stage', 'completed'),
                    enrichment_count
                ))
                
                # Store face recognition specific data if applicable
                self._store_face_recognition_data(cursor, event, event_data['event_id'])
                
                # Store enrichment results
                if enrichment_results:
                    self._store_enrichment_results(cursor, event_data['event_id'], enrichment_results)
                
                self.events_stored += 1
                logger.debug(f"Stored event: {event_data['event_id']}")
                
                return True
                
        except sqlite3.IntegrityError as e:
            logger.warning(f"Duplicate event or constraint violation: {e}")
            self.storage_errors += 1
            return False
        except Exception as e:
            logger.error(f"Failed to store event: {e}", exc_info=True)
            self.storage_errors += 1
            return False
    
    def _extract_event_data(self, event: Any) -> Dict[str, Any]:
        """Extract data from event object."""
        # Try to use to_dict method
        if hasattr(event, 'to_dict'):
            return event.to_dict()
        
        # Fallback: extract common attributes
        return {
            'event_id': getattr(event, 'event_id', None),
            'event_type': getattr(event, 'event_type', None).name if hasattr(event, 'event_type') else None,
            'timestamp': getattr(event, 'timestamp', None),
            'source': getattr(event, 'source', None),
            'priority': getattr(event, 'priority', None).name if hasattr(event, 'priority') else None,
            'confidence': getattr(event, 'confidence', None),
            'correlation_id': getattr(event, 'correlation_id', None),
            'parent_event_id': getattr(event, 'parent_event_id', None),
            'data': getattr(event, 'data', {})
        }
    
    def _store_face_recognition_data(self, cursor: sqlite3.Cursor, event: Any, event_id: str) -> None:
        """Store face recognition specific data."""
        from src.communication.events import FaceRecognitionEvent
        
        if isinstance(event, FaceRecognitionEvent):
            recognition_data = {
                'recognitions': [r.to_dict() for r in event.recognitions] if hasattr(event, 'recognitions') else []
            }
            
            cursor.execute('''
                INSERT INTO face_recognition_events
                (event_id, recognition_count, known_count, unknown_count, 
                 blacklisted_count, recognition_time, recognition_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                event_id,
                len(event.recognitions) if hasattr(event, 'recognitions') else 0,
                getattr(event, 'known_count', 0),
                getattr(event, 'unknown_count', 0),
                getattr(event, 'blacklisted_count', 0),
                getattr(event, 'recognition_time', 0),
                json.dumps(recognition_data)
            ))
    
    def _store_enrichment_results(self, cursor: sqlite3.Cursor, event_id: str, 
                                  enrichment_results: Dict[str, Any]) -> None:
        """Store enrichment processing results."""
        for processor_name, result in enrichment_results.items():
            # Handle both EnrichmentResult objects and dictionaries
            if hasattr(result, 'to_dict'):
                result_dict = result.to_dict()
            elif isinstance(result, dict):
                result_dict = result
            else:
                continue
            
            cursor.execute('''
                INSERT OR REPLACE INTO event_enrichments
                (event_id, processor_name, success, processing_time, 
                 enrichment_data, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                event_id,
                processor_name,
                1 if result_dict.get('success', False) else 0,
                result_dict.get('processing_time', 0.0),
                json.dumps(result_dict.get('enriched_data', {})),
                result_dict.get('error_message')
            ))
    
    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific event by ID.
        
        Args:
            event_id: Event ID to retrieve
            
        Returns:
            Event data dictionary or None if not found
        """
        try:
            with self._get_cursor() as cursor:
                cursor.execute('''
                    SELECT * FROM events WHERE event_id = ?
                ''', (event_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                event = dict(row)
                
                # Deserialize JSON fields
                event['event_data'] = json.loads(event['event_data']) if event['event_data'] else {}
                event['enriched_data'] = json.loads(event['enriched_data']) if event['enriched_data'] else {}
                
                # Get enrichment results
                event['enrichments'] = self._get_enrichment_results(cursor, event_id)
                
                self.queries_executed += 1
                return event
                
        except Exception as e:
            logger.error(f"Failed to retrieve event {event_id}: {e}")
            return None
    
    def _get_enrichment_results(self, cursor: sqlite3.Cursor, event_id: str) -> List[Dict[str, Any]]:
        """Get enrichment results for an event."""
        cursor.execute('''
            SELECT processor_name, success, processing_time, enrichment_data, error_message
            FROM event_enrichments WHERE event_id = ?
        ''', (event_id,))
        
        enrichments = []
        for row in cursor.fetchall():
            enrichment = dict(row)
            enrichment['enrichment_data'] = json.loads(enrichment['enrichment_data']) if enrichment['enrichment_data'] else {}
            enrichments.append(enrichment)
        
        return enrichments
    
    def query_events(self, filters: Optional[Dict[str, Any]] = None, 
                    limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Query events with optional filters.
        
        Args:
            filters: Filter criteria (event_type, source, priority, since, until)
            limit: Maximum number of events to return
            offset: Number of events to skip
            
        Returns:
            List of event records
        """
        try:
            filters = filters or {}
            
            with self._get_cursor() as cursor:
                query = "SELECT * FROM events WHERE 1=1"
                params = []
                
                # Apply filters
                if 'event_type' in filters:
                    query += " AND event_type = ?"
                    params.append(filters['event_type'])
                
                if 'source' in filters:
                    query += " AND source = ?"
                    params.append(filters['source'])
                
                if 'priority' in filters:
                    query += " AND priority = ?"
                    params.append(filters['priority'])
                
                if 'since' in filters:
                    query += " AND timestamp >= ?"
                    params.append(filters['since'])
                
                if 'until' in filters:
                    query += " AND timestamp <= ?"
                    params.append(filters['until'])
                
                if 'correlation_id' in filters:
                    query += " AND correlation_id = ?"
                    params.append(filters['correlation_id'])
                
                query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                events = []
                for row in rows:
                    event = dict(row)
                    event['event_data'] = json.loads(event['event_data']) if event['event_data'] else {}
                    event['enriched_data'] = json.loads(event['enriched_data']) if event['enriched_data'] else {}
                    events.append(event)
                
                self.queries_executed += 1
                return events
                
        except Exception as e:
            logger.error(f"Failed to query events: {e}")
            return []
    
    def get_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get event statistics for the specified period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Statistics dictionary
        """
        try:
            since = (datetime.now() - timedelta(days=days)).timestamp()
            
            with self._get_cursor() as cursor:
                # Total events
                cursor.execute(
                    "SELECT COUNT(*) as count FROM events WHERE timestamp >= ?",
                    (since,)
                )
                total = cursor.fetchone()['count']
                
                # By event type
                cursor.execute('''
                    SELECT event_type, COUNT(*) as count 
                    FROM events 
                    WHERE timestamp >= ?
                    GROUP BY event_type
                ''', (since,))
                by_type = {row['event_type']: row['count'] for row in cursor.fetchall()}
                
                # By priority
                cursor.execute('''
                    SELECT priority, COUNT(*) as count 
                    FROM events 
                    WHERE timestamp >= ?
                    GROUP BY priority
                ''', (since,))
                by_priority = {row['priority']: row['count'] for row in cursor.fetchall()}
                
                self.queries_executed += 1
                
                return {
                    'total_events': total,
                    'by_event_type': by_type,
                    'by_priority': by_priority,
                    'period_days': days,
                    'events_stored': self.events_stored,
                    'storage_errors': self.storage_errors,
                    'queries_executed': self.queries_executed
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def cleanup_old_events(self, retention_days: int = 30) -> int:
        """
        Delete events older than retention period.
        
        Args:
            retention_days: Number of days to retain events
            
        Returns:
            Number of events deleted
        """
        try:
            cutoff = (datetime.now() - timedelta(days=retention_days)).timestamp()
            
            with self._get_cursor() as cursor:
                cursor.execute(
                    "DELETE FROM events WHERE timestamp < ?",
                    (cutoff,)
                )
                deleted_count = cursor.rowcount
            
            logger.info(f"Cleaned up {deleted_count} old events (older than {retention_days} days)")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup events: {e}")
            return 0
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Event database closed")
