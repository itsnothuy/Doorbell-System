#!/usr/bin/env python3
"""
Storage Layer Demo - Showcase Storage System Capabilities

Demonstrates the key features of the comprehensive storage layer including:
- Storage manager initialization
- Metrics storage and querying
- Configuration management with versioning
- Database migrations
- Backup and restore operations
- Query builder usage
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from src.storage import (
    StorageManager,
    StorageConfig,
    SystemMetric,
    MetricType,
    MigrationManager,
    BackupManager,
    QueryBuilder,
    Operator,
    SortOrder
)
from src.storage.metrics_database import TimeRange


def demo_storage_manager():
    """Demonstrate storage manager capabilities."""
    print("\n" + "="*60)
    print("STORAGE MANAGER DEMO")
    print("="*60)
    
    # Create temporary directory for demo
    temp_dir = tempfile.mkdtemp(prefix="storage_demo_")
    print(f"✓ Created demo directory: {temp_dir}")
    
    try:
        # Initialize storage manager
        config = StorageConfig(
            data_directory=temp_dir,
            event_db_path=f"{temp_dir}/events.db",
            face_db_path=f"{temp_dir}/faces.db",
            metrics_db_path=f"{temp_dir}/metrics.db",
            config_db_path=f"{temp_dir}/config.db",
            event_db_wal_mode=False,  # Simpler for demo
            metrics_db_wal_mode=False,
            config_db_wal_mode=False
        )
        
        print("\n1. Initializing Storage Manager...")
        with StorageManager(config) as manager:
            print("   ✓ Storage manager initialized")
            print(f"   ✓ Event database: {Path(config.event_db_path).exists()}")
            print(f"   ✓ Face database: {Path(config.face_db_path).exists()}")
            print(f"   ✓ Metrics database: {Path(config.metrics_db_path).exists()}")
            print(f"   ✓ Config database: {Path(config.config_db_path).exists()}")
            
            # Health check
            print("\n2. Running Health Check...")
            health = manager.health_check()
            print(f"   ✓ System healthy: {health['healthy']}")
            print(f"   ✓ Databases: {len(health['databases'])}")
            
            # Store some metrics
            print("\n3. Storing System Metrics...")
            for i in range(5):
                metric = SystemMetric(
                    name="cpu_usage",
                    value=float(50 + i * 10),
                    metric_type=MetricType.GAUGE,
                    unit="percent"
                )
                manager.metrics_db.store_metric(metric)
                time.sleep(0.01)
            print("   ✓ Stored 5 CPU usage metrics")
            
            # Query metrics
            print("\n4. Querying Metrics...")
            time_range = TimeRange(
                start=datetime.now() - timedelta(minutes=1),
                end=datetime.now() + timedelta(minutes=1)
            )
            history = manager.metrics_db.get_metric_history("cpu_usage", time_range)
            print(f"   ✓ Retrieved {len(history)} metric data points")
            if history:
                print(f"   ✓ Latest value: {history[0].value}%")
            
            # Store configuration
            print("\n5. Managing Configuration...")
            config_value = {
                "retention_days": 90,
                "backup_enabled": True,
                "compression": "gzip"
            }
            manager.config_db.set_config(
                "system_settings",
                config_value,
                description="Main system configuration"
            )
            print("   ✓ Configuration stored")
            
            # Retrieve configuration
            retrieved = manager.config_db.get_config("system_settings")
            print(f"   ✓ Retrieved config: retention_days={retrieved['retention_days']}")
            
            # Update configuration (creates new version)
            config_value["retention_days"] = 180
            manager.config_db.set_config(
                "system_settings",
                config_value,
                change_reason="Increased retention for compliance"
            )
            print("   ✓ Configuration updated (version 2)")
            
            # Show version history
            history = manager.config_db.get_version_history("system_settings")
            print(f"   ✓ Version history: {len(history)} versions")
            
            # Get storage metrics
            print("\n6. Storage Metrics...")
            metrics = manager.get_metrics()
            print(f"   ✓ Total queries: {metrics.total_queries}")
            print(f"   ✓ Total inserts: {metrics.total_inserts}")
            print(f"   ✓ Databases: {list(metrics.database_sizes.keys())}")
            
            total_size = sum(metrics.database_sizes.values())
            print(f"   ✓ Total storage: {total_size / 1024:.1f} KB")
        
        print("\n✓ Demo completed successfully!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n✓ Cleaned up demo directory")


def demo_migrations():
    """Demonstrate migration manager."""
    print("\n" + "="*60)
    print("MIGRATION MANAGER DEMO")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp(prefix="migration_demo_")
    db_path = f"{temp_dir}/test.db"
    migrations_dir = f"{temp_dir}/migrations"
    Path(migrations_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        print("\n1. Creating Sample Migration...")
        
        # Create a simple migration file
        migration_sql = """-- version: 001
-- name: create_users_table
-- description: Create initial users table

-- up
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL,
    email TEXT NOT NULL
);

-- down
DROP TABLE users;
"""
        migration_file = Path(migrations_dir) / "001_create_users_table.sql"
        migration_file.write_text(migration_sql)
        print(f"   ✓ Created migration: {migration_file.name}")
        
        print("\n2. Running Migrations...")
        with MigrationManager(db_path, migrations_dir) as manager:
            status = manager.get_migration_status()
            print(f"   ✓ Applied: {status['total_applied']}")
            print(f"   ✓ Pending: {status['total_pending']}")
            
            if status['total_pending'] > 0:
                result = manager.run_migrations()
                if result.success:
                    print(f"   ✓ Applied {len(result.migrations_applied)} migration(s)")
                else:
                    print(f"   ✗ Migration failed: {result.error}")
        
        print("\n✓ Migration demo completed!")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_backups():
    """Demonstrate backup manager."""
    print("\n" + "="*60)
    print("BACKUP MANAGER DEMO")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp(prefix="backup_demo_")
    backup_dir = f"{temp_dir}/backups"
    
    try:
        # Create a sample database
        db_path = f"{temp_dir}/sample.db"
        with open(db_path, 'w') as f:
            f.write("Sample database content for backup demo")
        
        print("\n1. Creating Backup...")
        manager = BackupManager(backup_dir, compress=True)
        
        result = manager.backup_database(db_path, backup_name="sample_backup")
        
        if result.success:
            print(f"   ✓ Backup created: {result.backup_id}")
            print(f"   ✓ Original size: {result.size_bytes} bytes")
            print(f"   ✓ Backup path: {Path(result.backup_path).name}")
        
        print("\n2. Listing Backups...")
        backups = manager.list_backups()
        print(f"   ✓ Total backups: {len(backups)}")
        
        if backups:
            backup = backups[0]
            print(f"   ✓ Latest backup: {backup.backup_id}")
            print(f"   ✓ Compression ratio: {backup.compression_ratio * 100:.1f}%")
        
        print("\n3. Backup Statistics...")
        stats = manager.get_backup_statistics()
        print(f"   ✓ Total backups: {stats['total_backups']}")
        print(f"   ✓ Total size: {stats['total_size_bytes']} bytes")
        
        print("\n✓ Backup demo completed!")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_query_builder():
    """Demonstrate query builder."""
    print("\n" + "="*60)
    print("QUERY BUILDER DEMO")
    print("="*60)
    
    print("\n1. Simple SELECT Query...")
    builder = QueryBuilder(table="users")
    query, params = builder.select("id", "username", "email").build_select()
    print(f"   SQL: {query}")
    print(f"   Params: {params}")
    
    print("\n2. Filtered Query...")
    builder = (QueryBuilder(table="events")
              .select("event_id", "event_type", "timestamp")
              .filter_eq("status", "completed")
              .filter_gte("timestamp", 1234567890))
    query, params = builder.build_select()
    print(f"   SQL: {query}")
    print(f"   Params: {params}")
    
    print("\n3. Complex Query with Sorting and Pagination...")
    builder = (QueryBuilder(table="users")
              .filter_in("role", ["admin", "moderator"])
              .filter_like("email", "%@example.com")
              .order_by_desc("created_at")
              .paginate(limit=10, offset=0))
    query, params = builder.build_select()
    print(f"   SQL: {query}")
    print(f"   Params: {params}")
    
    print("\n4. COUNT Query...")
    builder = QueryBuilder(table="events").filter_eq("event_type", "MOTION_DETECTED")
    query, params = builder.build_count()
    print(f"   SQL: {query}")
    print(f"   Params: {params}")
    
    print("\n✓ Query builder demo completed!")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("STORAGE LAYER COMPREHENSIVE DEMO")
    print("="*60)
    print("\nDemonstrating complete storage system capabilities...")
    
    demo_storage_manager()
    demo_migrations()
    demo_backups()
    demo_query_builder()
    
    print("\n" + "="*60)
    print("ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nThe storage layer provides:")
    print("  ✓ Centralized storage management")
    print("  ✓ Time-series metrics with aggregation")
    print("  ✓ Versioned configuration with rollback")
    print("  ✓ Database migrations with safety")
    print("  ✓ Automated backups with compression")
    print("  ✓ Type-safe query building")
    print("  ✓ Comprehensive health monitoring")
    print("\nReady for production use!")


if __name__ == '__main__':
    main()
