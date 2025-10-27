-- version: 001
-- name: initial_storage_schema
-- description: Create initial storage layer schema for events, faces, metrics, and config

-- up

-- Events table (core event storage)
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
    event_data TEXT,
    enriched_data TEXT,
    processing_stage TEXT,
    enrichment_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_event_id) REFERENCES events(event_id)
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_source ON events(source);
CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at DESC);

-- Metrics table (system performance metrics)
CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_type TEXT NOT NULL,
    metric_unit TEXT,
    tags TEXT,
    timestamp REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_metrics_name ON system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_name_time ON system_metrics(metric_name, timestamp DESC);

-- Configuration table (versioned config storage)
CREATE TABLE IF NOT EXISTS configurations (
    config_key TEXT PRIMARY KEY,
    config_value TEXT NOT NULL,
    config_type TEXT DEFAULT 'general',
    description TEXT,
    is_sensitive INTEGER DEFAULT 0,
    version INTEGER DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    updated_by TEXT
);

CREATE INDEX IF NOT EXISTS idx_config_type ON configurations(config_type);
CREATE INDEX IF NOT EXISTS idx_config_updated ON configurations(updated_at DESC);

-- down

-- Drop indexes first
DROP INDEX IF EXISTS idx_events_timestamp;
DROP INDEX IF EXISTS idx_events_event_type;
DROP INDEX IF EXISTS idx_events_source;
DROP INDEX IF EXISTS idx_events_created_at;
DROP INDEX IF EXISTS idx_metrics_name;
DROP INDEX IF EXISTS idx_metrics_timestamp;
DROP INDEX IF EXISTS idx_metrics_name_time;
DROP INDEX IF EXISTS idx_config_type;
DROP INDEX IF EXISTS idx_config_updated;

-- Drop tables
DROP TABLE IF EXISTS events;
DROP TABLE IF EXISTS system_metrics;
DROP TABLE IF EXISTS configurations;
