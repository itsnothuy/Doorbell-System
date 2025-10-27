#!/usr/bin/env python3
"""
Metrics Database - System Performance and Health Metrics Storage

Stores time-series metrics data for system monitoring, performance tracking,
and analytics. Supports multiple metric types with efficient querying.
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from src.storage.base_storage import BaseDatabase, DatabaseConfig, QueryResult

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be stored."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMING = "timing"
    RATE = "rate"


class MetricAggregation(Enum):
    """Aggregation methods for metrics."""
    NONE = auto()
    AVERAGE = auto()
    SUM = auto()
    MIN = auto()
    MAX = auto()
    COUNT = auto()
    HOURLY_AVERAGE = auto()
    DAILY_AVERAGE = auto()


@dataclass
class SystemMetric:
    """Represents a system metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


@dataclass
class MetricDataPoint:
    """A single metric data point with timestamp."""
    value: float
    timestamp: datetime
    tags: Optional[Dict[str, str]] = None


@dataclass
class TimeRange:
    """Time range for metric queries."""
    start: datetime
    end: datetime


@dataclass
class MetricStatistics:
    """Statistical summary of metrics."""
    count: int = 0
    sum: float = 0.0
    average: float = 0.0
    min: float = 0.0
    max: float = 0.0
    std_dev: float = 0.0


class MetricsDatabase(BaseDatabase):
    """Database for system metrics and performance data."""
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize metrics database.
        
        Args:
            config: Database configuration
        """
        super().__init__(config)
        self.metrics_table = "system_metrics"
        self.performance_table = "performance_metrics"
        self.aggregates_table = "metric_aggregates"
    
    def initialize(self) -> bool:
        """Initialize metrics database schema."""
        try:
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            
            # Configure database
            self._configure_database()
            
            # Create schema
            self._create_tables()
            
            logger.info("Metrics database initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics database: {e}")
            return False
    
    def _create_tables(self) -> None:
        """Create metrics database tables."""
        with self.get_cursor() as cursor:
            # System metrics table
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.metrics_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    metric_unit TEXT,
                    tags TEXT,
                    timestamp REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Performance metrics table
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.performance_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    success INTEGER DEFAULT 1,
                    error_message TEXT,
                    metadata TEXT,
                    timestamp REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Pre-computed aggregates table for performance
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.aggregates_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    aggregation_type TEXT NOT NULL,
                    aggregation_value REAL NOT NULL,
                    start_timestamp REAL NOT NULL,
                    end_timestamp REAL NOT NULL,
                    sample_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(metric_name, aggregation_type, start_timestamp)
                )
            ''')
            
            # Create indexes
            self._create_indexes(cursor)
    
    def _create_indexes(self, cursor: sqlite3.Cursor) -> None:
        """Create performance indexes."""
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_metrics_name ON {self.metrics_table}(metric_name)",
            f"CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON {self.metrics_table}(timestamp DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_metrics_name_time ON {self.metrics_table}(metric_name, timestamp DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_perf_component ON {self.performance_table}(component)",
            f"CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON {self.performance_table}(timestamp DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_aggregates_name ON {self.aggregates_table}(metric_name, start_timestamp)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
    
    def store_metric(self, metric: SystemMetric) -> bool:
        """
        Store system metric data point.
        
        Args:
            metric: SystemMetric to store
            
        Returns:
            True if successful
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute(f'''
                    INSERT INTO {self.metrics_table}
                    (metric_name, metric_value, metric_type, metric_unit, tags, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    metric.name,
                    metric.value,
                    metric.metric_type.value,
                    metric.unit,
                    json.dumps(metric.tags),
                    metric.timestamp
                ))
                
                self.inserts_executed += 1
                return True
                
        except Exception as e:
            logger.error(f"Failed to store metric {metric.name}: {e}")
            return False
    
    def store_metrics_batch(self, metrics: List[SystemMetric]) -> int:
        """
        Store multiple metrics in a batch for efficiency.
        
        Args:
            metrics: List of metrics to store
            
        Returns:
            Number of metrics successfully stored
        """
        if not metrics:
            return 0
        
        stored_count = 0
        try:
            with self.get_cursor() as cursor:
                for metric in metrics:
                    try:
                        cursor.execute(f'''
                            INSERT INTO {self.metrics_table}
                            (metric_name, metric_value, metric_type, metric_unit, tags, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            metric.name,
                            metric.value,
                            metric.metric_type.value,
                            metric.unit,
                            json.dumps(metric.tags),
                            metric.timestamp
                        ))
                        stored_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to store metric {metric.name} in batch: {e}")
                
                self.inserts_executed += stored_count
                
        except Exception as e:
            logger.error(f"Batch metric storage failed: {e}")
        
        return stored_count
    
    def store_performance_metric(self, component: str, operation: str,
                                 duration_ms: float, success: bool = True,
                                 error_message: Optional[str] = None,
                                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store performance metric for a component operation.
        
        Args:
            component: Component name
            operation: Operation name
            duration_ms: Duration in milliseconds
            success: Whether operation succeeded
            error_message: Optional error message
            metadata: Optional additional metadata
            
        Returns:
            True if successful
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute(f'''
                    INSERT INTO {self.performance_table}
                    (component, operation, duration_ms, success, error_message, metadata, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    component,
                    operation,
                    duration_ms,
                    1 if success else 0,
                    error_message,
                    json.dumps(metadata or {}),
                    time.time()
                ))
                
                self.inserts_executed += 1
                return True
                
        except Exception as e:
            logger.error(f"Failed to store performance metric: {e}")
            return False
    
    def get_metric_history(self, metric_name: str, time_range: TimeRange,
                          aggregation: MetricAggregation = MetricAggregation.NONE,
                          limit: Optional[int] = None) -> List[MetricDataPoint]:
        """
        Get metric history with optional aggregation.
        
        Args:
            metric_name: Name of metric
            time_range: Time range to query
            aggregation: Aggregation method
            limit: Optional limit on results
            
        Returns:
            List of metric data points
        """
        try:
            with self.get_cursor() as cursor:
                start_ts = time_range.start.timestamp()
                end_ts = time_range.end.timestamp()
                
                if aggregation == MetricAggregation.NONE:
                    # Raw data points
                    query = f'''
                        SELECT metric_value, timestamp, tags
                        FROM {self.metrics_table}
                        WHERE metric_name = ? AND timestamp BETWEEN ? AND ?
                        ORDER BY timestamp DESC
                    '''
                    params = [metric_name, start_ts, end_ts]
                    
                    if limit:
                        query += " LIMIT ?"
                        params.append(limit)
                    
                    cursor.execute(query, params)
                    
                elif aggregation == MetricAggregation.HOURLY_AVERAGE:
                    # Hourly averages
                    cursor.execute(f'''
                        SELECT AVG(metric_value) as avg_value,
                               CAST(timestamp / 3600 AS INTEGER) * 3600 as hour_timestamp,
                               NULL as tags
                        FROM {self.metrics_table}
                        WHERE metric_name = ? AND timestamp BETWEEN ? AND ?
                        GROUP BY hour_timestamp
                        ORDER BY hour_timestamp DESC
                    ''' + (f" LIMIT {limit}" if limit else ""), 
                    (metric_name, start_ts, end_ts))
                    
                elif aggregation == MetricAggregation.DAILY_AVERAGE:
                    # Daily averages
                    cursor.execute(f'''
                        SELECT AVG(metric_value) as avg_value,
                               CAST(timestamp / 86400 AS INTEGER) * 86400 as day_timestamp,
                               NULL as tags
                        FROM {self.metrics_table}
                        WHERE metric_name = ? AND timestamp BETWEEN ? AND ?
                        GROUP BY day_timestamp
                        ORDER BY day_timestamp DESC
                    ''' + (f" LIMIT {limit}" if limit else ""),
                    (metric_name, start_ts, end_ts))
                
                else:
                    # Other aggregations (average, sum, min, max)
                    agg_func = aggregation.name
                    cursor.execute(f'''
                        SELECT {agg_func}(metric_value) as agg_value,
                               MAX(timestamp) as max_timestamp,
                               NULL as tags
                        FROM {self.metrics_table}
                        WHERE metric_name = ? AND timestamp BETWEEN ? AND ?
                    ''', (metric_name, start_ts, end_ts))
                
                data_points = []
                for row in cursor.fetchall():
                    if aggregation == MetricAggregation.NONE:
                        value, timestamp_val, tags_json = row
                        tags = json.loads(tags_json) if tags_json else None
                    else:
                        value, timestamp_val, tags_json = row
                        tags = None
                    
                    data_points.append(MetricDataPoint(
                        value=value,
                        timestamp=datetime.fromtimestamp(timestamp_val),
                        tags=tags
                    ))
                
                self.queries_executed += 1
                return data_points
                
        except Exception as e:
            logger.error(f"Failed to get metric history for {metric_name}: {e}")
            return []
    
    def get_metric_statistics(self, metric_name: str, time_range: TimeRange) -> MetricStatistics:
        """
        Get statistical summary for a metric.
        
        Args:
            metric_name: Name of metric
            time_range: Time range to analyze
            
        Returns:
            Metric statistics
        """
        try:
            with self.get_cursor() as cursor:
                start_ts = time_range.start.timestamp()
                end_ts = time_range.end.timestamp()
                
                cursor.execute(f'''
                    SELECT 
                        COUNT(*) as count,
                        SUM(metric_value) as sum,
                        AVG(metric_value) as average,
                        MIN(metric_value) as min,
                        MAX(metric_value) as max
                    FROM {self.metrics_table}
                    WHERE metric_name = ? AND timestamp BETWEEN ? AND ?
                ''', (metric_name, start_ts, end_ts))
                
                row = cursor.fetchone()
                if row and row['count'] > 0:
                    # Calculate standard deviation
                    avg = row['average']
                    cursor.execute(f'''
                        SELECT SUM((metric_value - ?) * (metric_value - ?)) / COUNT(*) as variance
                        FROM {self.metrics_table}
                        WHERE metric_name = ? AND timestamp BETWEEN ? AND ?
                    ''', (avg, avg, metric_name, start_ts, end_ts))
                    
                    variance_row = cursor.fetchone()
                    std_dev = (variance_row['variance'] ** 0.5) if variance_row['variance'] else 0.0
                    
                    self.queries_executed += 2
                    
                    return MetricStatistics(
                        count=row['count'],
                        sum=row['sum'] or 0.0,
                        average=row['average'] or 0.0,
                        min=row['min'] or 0.0,
                        max=row['max'] or 0.0,
                        std_dev=std_dev
                    )
                
                self.queries_executed += 1
                return MetricStatistics()
                
        except Exception as e:
            logger.error(f"Failed to get metric statistics for {metric_name}: {e}")
            return MetricStatistics()
    
    def get_performance_summary(self, component: str, 
                               time_range: Optional[TimeRange] = None) -> Dict[str, Any]:
        """
        Get performance summary for a component.
        
        Args:
            component: Component name
            time_range: Optional time range (defaults to last hour)
            
        Returns:
            Performance summary dictionary
        """
        if time_range is None:
            time_range = TimeRange(
                start=datetime.now() - timedelta(hours=1),
                end=datetime.now()
            )
        
        try:
            with self.get_cursor() as cursor:
                start_ts = time_range.start.timestamp()
                end_ts = time_range.end.timestamp()
                
                # Get operation statistics
                cursor.execute(f'''
                    SELECT 
                        operation,
                        COUNT(*) as operation_count,
                        AVG(duration_ms) as avg_duration,
                        MIN(duration_ms) as min_duration,
                        MAX(duration_ms) as max_duration,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                        SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as error_count
                    FROM {self.performance_table}
                    WHERE component = ? AND timestamp BETWEEN ? AND ?
                    GROUP BY operation
                ''', (component, start_ts, end_ts))
                
                operations = {}
                for row in cursor.fetchall():
                    operations[row['operation']] = {
                        'count': row['operation_count'],
                        'avg_duration_ms': row['avg_duration'],
                        'min_duration_ms': row['min_duration'],
                        'max_duration_ms': row['max_duration'],
                        'success_count': row['success_count'],
                        'error_count': row['error_count'],
                        'success_rate': row['success_count'] / row['operation_count']
                    }
                
                self.queries_executed += 1
                
                return {
                    'component': component,
                    'time_range': {
                        'start': time_range.start.isoformat(),
                        'end': time_range.end.isoformat()
                    },
                    'operations': operations
                }
                
        except Exception as e:
            logger.error(f"Failed to get performance summary for {component}: {e}")
            return {}
    
    def cleanup_old_metrics(self, retention_days: int) -> int:
        """
        Clean up metrics older than retention period.
        
        Args:
            retention_days: Number of days to retain
            
        Returns:
            Number of metrics deleted
        """
        try:
            cutoff_timestamp = (datetime.now() - timedelta(days=retention_days)).timestamp()
            
            with self.get_cursor() as cursor:
                # Delete old metrics
                cursor.execute(f'''
                    DELETE FROM {self.metrics_table}
                    WHERE timestamp < ?
                ''', (cutoff_timestamp,))
                
                metrics_deleted = cursor.rowcount
                
                # Delete old performance data
                cursor.execute(f'''
                    DELETE FROM {self.performance_table}
                    WHERE timestamp < ?
                ''', (cutoff_timestamp,))
                
                perf_deleted = cursor.rowcount
                
                # Delete old aggregates
                cursor.execute(f'''
                    DELETE FROM {self.aggregates_table}
                    WHERE end_timestamp < ?
                ''', (cutoff_timestamp,))
                
                agg_deleted = cursor.rowcount
                
                total_deleted = metrics_deleted + perf_deleted + agg_deleted
                
                self.deletes_executed += 1
                logger.info(f"Cleaned up {total_deleted} old metric records")
                
                return total_deleted
                
        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")
            return 0
