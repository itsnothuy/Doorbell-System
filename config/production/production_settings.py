#!/usr/bin/env python3
"""
Production Settings

Configuration settings specific to production deployments.
"""

import os
from pathlib import Path
from typing import Any, Dict


class ProductionSettings:
    """Production environment configuration settings."""

    def __init__(self):
        """Initialize production settings."""
        # Environment
        self.ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
        self.DEBUG = os.getenv("DEBUG", "False").lower() == "true"

        # Application
        self.APP_NAME = os.getenv("APP_NAME", "doorbell-security-system")
        self.APP_VERSION = os.getenv("APP_VERSION", "1.0.0")

        # Paths
        self.PROJECT_ROOT = Path(__file__).parent.parent.parent
        self.DATA_DIR = Path(os.getenv("DATA_DIR", self.PROJECT_ROOT / "data"))
        self.LOG_DIR = Path(os.getenv("LOG_DIR", self.DATA_DIR / "logs"))
        self.BACKUP_DIR = Path(os.getenv("BACKUP_DIR", self.DATA_DIR / "backups"))

        # Create directories
        for directory in [self.DATA_DIR, self.LOG_DIR, self.BACKUP_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

        # Security
        self.SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
        self.ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "*").split(",")
        self.CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

        # Performance
        self.WORKER_PROCESSES = int(os.getenv("WORKER_PROCESSES", "4"))
        self.WORKER_THREADS = int(os.getenv("WORKER_THREADS", "2"))
        self.MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "1000"))

        # Monitoring
        self.MONITORING_ENABLED = os.getenv("MONITORING_ENABLED", "True").lower() == "true"
        self.PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "8000"))
        self.METRICS_COLLECTION_INTERVAL = int(os.getenv("METRICS_COLLECTION_INTERVAL", "30"))

        # Logging
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FORMAT = os.getenv("LOG_FORMAT", "json")
        self.LOG_ROTATION_SIZE = int(os.getenv("LOG_ROTATION_SIZE", "10485760"))  # 10MB
        self.LOG_RETENTION_DAYS = int(os.getenv("LOG_RETENTION_DAYS", "30"))

        # Deployment
        self.DEPLOYMENT_STRATEGY = os.getenv("DEPLOYMENT_STRATEGY", "blue-green")
        self.HEALTH_CHECK_TIMEOUT = int(os.getenv("HEALTH_CHECK_TIMEOUT", "30"))
        self.ROLLBACK_ENABLED = os.getenv("ROLLBACK_ENABLED", "True").lower() == "true"

        # Scaling
        self.AUTO_SCALING_ENABLED = os.getenv("AUTO_SCALING_ENABLED", "False").lower() == "true"
        self.MIN_WORKERS = int(os.getenv("MIN_WORKERS", "2"))
        self.MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))
        self.SCALE_UP_THRESHOLD = float(os.getenv("SCALE_UP_THRESHOLD", "0.8"))
        self.SCALE_DOWN_THRESHOLD = float(os.getenv("SCALE_DOWN_THRESHOLD", "0.3"))

        # Backup
        self.BACKUP_ENABLED = os.getenv("BACKUP_ENABLED", "True").lower() == "true"
        self.BACKUP_SCHEDULE = os.getenv("BACKUP_SCHEDULE", "0 2 * * *")  # 2 AM daily
        self.BACKUP_RETENTION_DAYS = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }

    def validate(self) -> bool:
        """
        Validate production settings.

        Returns:
            True if settings are valid
        """
        # Check required settings
        if self.SECRET_KEY == "change-me-in-production":
            raise ValueError("SECRET_KEY must be changed in production")

        if self.ENVIRONMENT == "production" and self.DEBUG:
            raise ValueError("DEBUG must be False in production")

        return True


# Singleton instance
_production_settings = None


def get_production_settings() -> ProductionSettings:
    """
    Get production settings singleton.

    Returns:
        ProductionSettings instance
    """
    global _production_settings
    if _production_settings is None:
        _production_settings = ProductionSettings()
    return _production_settings
