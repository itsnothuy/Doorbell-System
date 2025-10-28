#!/usr/bin/env python3
"""
Migration Utilities - Legacy to Pipeline Migration Tools

Utilities for migrating from legacy DoorbellSecuritySystem to pipeline architecture.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class MigrationUtils:
    """Utilities for system migration."""
    
    @staticmethod
    def validate_migration_compatibility() -> Dict[str, Any]:
        """
        Validate that the system is ready for migration.
        
        Returns:
            Dict containing validation results and any issues found
        """
        results = {
            'compatible': True,
            'issues': [],
            'warnings': []
        }
        
        # Check for required directories
        required_dirs = [
            'data/captures',
            'data/known_faces',
            'data/blacklist_faces',
            'data/logs'
        ]
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            if not path.exists():
                results['warnings'].append(f"Directory {dir_path} does not exist (will be created)")
        
        # Check for configuration files
        config_files = [
            'config/settings.py',
            'config/pipeline_config.py'
        ]
        
        for config_file in config_files:
            path = Path(config_file)
            if not path.exists():
                results['issues'].append(f"Required configuration file missing: {config_file}")
                results['compatible'] = False
        
        return results
    
    @staticmethod
    def migrate_legacy_data() -> Dict[str, Any]:
        """
        Migrate data from legacy system to pipeline structure.
        
        Returns:
            Dict containing migration results
        """
        results = {
            'success': True,
            'migrated_items': [],
            'errors': []
        }
        
        try:
            # Ensure required directories exist
            directories = [
                'data/captures',
                'data/cropped_faces',
                'data/known_faces',
                'data/blacklist_faces',
                'data/logs'
            ]
            
            for dir_path in directories:
                path = Path(dir_path)
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
                    results['migrated_items'].append(f"Created directory: {dir_path}")
                    logger.info(f"Created directory: {dir_path}")
            
            # No actual data migration needed as both systems use same data structure
            logger.info("Legacy data structure is compatible with pipeline")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            results['success'] = False
            results['errors'].append(str(e))
        
        return results
    
    @staticmethod
    def create_backup(backup_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Create a backup of the current system state.
        
        Args:
            backup_path: Path to store backup (defaults to data/backups)
            
        Returns:
            Dict containing backup results
        """
        import shutil
        from datetime import datetime
        
        if backup_path is None:
            backup_path = Path('data/backups')
        
        backup_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results = {
            'success': True,
            'backup_path': None,
            'errors': []
        }
        
        try:
            # Create timestamped backup directory
            backup_dir = backup_path / f"backup_{timestamp}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup data directories
            data_dirs = ['captures', 'known_faces', 'blacklist_faces', 'logs']
            for dir_name in data_dirs:
                src = Path('data') / dir_name
                if src.exists():
                    dst = backup_dir / dir_name
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    logger.info(f"Backed up {src} to {dst}")
            
            # Backup database if it exists
            db_file = Path('data/events.db')
            if db_file.exists():
                shutil.copy2(db_file, backup_dir / 'events.db')
                logger.info(f"Backed up database to {backup_dir}")
            
            results['backup_path'] = str(backup_dir)
            logger.info(f"Backup created successfully at {backup_dir}")
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            results['success'] = False
            results['errors'].append(str(e))
        
        return results
    
    @staticmethod
    def verify_pipeline_health() -> Dict[str, Any]:
        """
        Verify that the pipeline is healthy and ready.
        
        Returns:
            Dict containing health check results
        """
        results = {
            'healthy': True,
            'checks': {},
            'issues': []
        }
        
        # Check if pipeline modules can be imported
        try:
            from src.pipeline.orchestrator import PipelineOrchestrator
            results['checks']['orchestrator_import'] = True
        except Exception as e:
            results['checks']['orchestrator_import'] = False
            results['healthy'] = False
            results['issues'].append(f"Cannot import orchestrator: {e}")
        
        try:
            from src.integration.orchestrator_manager import OrchestratorManager
            results['checks']['manager_import'] = True
        except Exception as e:
            results['checks']['manager_import'] = False
            results['healthy'] = False
            results['issues'].append(f"Cannot import manager: {e}")
        
        try:
            from config.orchestrator_config import OrchestratorConfig
            results['checks']['config_import'] = True
        except Exception as e:
            results['checks']['config_import'] = False
            results['healthy'] = False
            results['issues'].append(f"Cannot import config: {e}")
        
        return results
