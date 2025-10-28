#!/usr/bin/env python3
"""
Integration Layer - Pipeline Orchestrator Integration

This package provides integration utilities for migrating from the legacy
DoorbellSecuritySystem to the new pipeline architecture.
"""

from .orchestrator_manager import OrchestratorManager, SystemState, SystemHealth
from .legacy_adapter import LegacyAdapter

__all__ = [
    'OrchestratorManager',
    'SystemState',
    'SystemHealth',
    'LegacyAdapter'
]
