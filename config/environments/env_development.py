#!/usr/bin/env python3
"""
Development Environment Configuration

This configuration is optimized for local development with verbose logging
and debugging features enabled.
"""

# Logging configuration
LOG_LEVEL = "DEBUG"
LOG_TO_CONSOLE = True

# Performance settings - use more resources for faster development
WORKER_COUNT = 4
FACE_DETECTION_WORKERS = 4
FACE_RECOGNITION_WORKERS = 3

# Development features
ENABLE_DEBUGGING = True
ENABLE_PROFILING = True
ENABLE_HOT_RELOAD = True

# Monitoring
HEALTH_CHECK_INTERVAL = 10.0
METRICS_COLLECTION = True

# Notifications - disabled or console-only for development
TELEGRAM_ENABLED = False
NOTIFICATION_CONSOLE_OUTPUT = True

# Storage
KEEP_ALL_CAPTURES = True
DATABASE_BACKUP_ENABLED = False

# Hardware mocks
USE_MOCK_CAMERA = True
USE_MOCK_GPIO = True

# Web interface
WEB_DEBUG_MODE = True
WEB_PORT = 5000
WEB_RELOAD = True
