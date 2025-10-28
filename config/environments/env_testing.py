#!/usr/bin/env python3
"""
Testing Environment Configuration

This configuration is optimized for automated testing with minimal resources
and fast startup times.
"""

# Logging configuration
LOG_LEVEL = "WARNING"
LOG_TO_CONSOLE = False

# Performance settings - minimal for fast tests
WORKER_COUNT = 1
FACE_DETECTION_WORKERS = 1
FACE_RECOGNITION_WORKERS = 1

# Testing features
ENABLE_DEBUGGING = False
ENABLE_PROFILING = False
ENABLE_HOT_RELOAD = False

# Monitoring - disabled for speed
HEALTH_CHECK_INTERVAL = 60.0
METRICS_COLLECTION = False

# Notifications - disabled for testing
TELEGRAM_ENABLED = False
NOTIFICATION_CONSOLE_OUTPUT = False

# Storage
KEEP_CAPTURES_DAYS = 1
DATABASE_BACKUP_ENABLED = False
USE_IN_MEMORY_DB = True

# Hardware - always use mocks
USE_MOCK_CAMERA = True
USE_MOCK_GPIO = True
USE_MOCK_DETECTOR = True

# Web interface - disabled
WEB_ENABLED = False
WEB_PORT = 5001

# Motion detection - disabled for predictable tests
MOTION_DETECTION_ENABLED = False

# Fast timeouts
OPERATION_TIMEOUT = 5.0
QUEUE_TIMEOUT = 1.0
