#!/usr/bin/env python3
"""
Migration package initialization.
"""

from config.migration.migration_config import MigrationConfig
from config.migration.legacy_mapping import LegacyMapping

__all__ = ['MigrationConfig', 'LegacyMapping']
