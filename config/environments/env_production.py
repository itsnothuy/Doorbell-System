#!/usr/bin/env python3
"""
Production Environment Configuration

This configuration is optimized for production deployment with reliability
and security as top priorities.
"""

# Logging configuration
LOG_LEVEL = "INFO"
LOG_TO_CONSOLE = False
LOG_TO_FILE = True

# Performance settings - conservative for stability
WORKER_COUNT = 2
FACE_DETECTION_WORKERS = 2
FACE_RECOGNITION_WORKERS = 2

# Production features
ENABLE_DEBUGGING = False
ENABLE_PROFILING = False
ENABLE_HOT_RELOAD = False  # Can be enabled if needed

# Monitoring - critical for production
HEALTH_CHECK_INTERVAL = 30.0
METRICS_COLLECTION = True
ALERT_ON_ERRORS = True

# Notifications
TELEGRAM_ENABLED = True
RATE_LIMITING_ENABLED = True
MAX_NOTIFICATIONS_PER_HOUR = 100

# Storage
KEEP_CAPTURES_DAYS = 7
DATABASE_BACKUP_ENABLED = True
BACKUP_INTERVAL_HOURS = 24

# Hardware - real devices
USE_MOCK_CAMERA = False
USE_MOCK_GPIO = False

# Web interface
WEB_DEBUG_MODE = False
WEB_PORT = 8000
WEB_RELOAD = False

# Security
ENABLE_AUTHENTICATION = True
ENABLE_HTTPS = True
SESSION_TIMEOUT = 3600

# Motion detection - enabled for resource efficiency
MOTION_DETECTION_ENABLED = True
