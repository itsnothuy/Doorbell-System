#!/usr/bin/env python3
"""
Migrate to Pipeline - Complete System Migration Script

This script migrates the doorbell security system from legacy architecture
to the new pipeline orchestrator architecture.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.integration.migration_manager import MigrationManager
from config.logging_config import setup_logging


def main():
    """Execute complete system migration."""
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("Doorbell Security System - Pipeline Migration")
    logger.info("=" * 70)
    
    # Create migration manager
    migration_config = {
        'backup_dir': 'data/migration_backup',
        'log_file': 'data/logs/migration.log',
        'auto_rollback': True,
        'compress_backup': False
    }
    
    manager = MigrationManager(migration_config)
    
    # Display migration information
    logger.info("\nMigration Plan:")
    logger.info("  1. Preparation - Validate system and check dependencies")
    logger.info("  2. Backup - Create comprehensive backup of current system")
    logger.info("  3. Configuration Migration - Migrate settings to pipeline format")
    logger.info("  4. Data Migration - Migrate face databases and events")
    logger.info("  5. System Integration - Integrate pipeline orchestrator")
    logger.info("  6. Validation - Validate complete migration")
    logger.info("  7. Cleanup - Remove temporary migration artifacts")
    logger.info("")
    
    # Confirm migration
    try:
        response = input("Proceed with migration? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            logger.info("Migration cancelled by user")
            return 0
    except KeyboardInterrupt:
        logger.info("\nMigration cancelled by user")
        return 0
    
    # Run migration
    logger.info("\nStarting migration...")
    success = manager.run_migration()
    
    # Display results
    status = manager.get_migration_status()
    
    logger.info("\n" + "=" * 70)
    if success:
        logger.info("✅ MIGRATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Stage: {status.stage.value}")
        logger.info(f"Progress: {status.progress * 100:.1f}%")
        
        if status.warnings:
            logger.info(f"\nWarnings ({len(status.warnings)}):")
            for warning in status.warnings:
                logger.info(f"  - {warning}")
        
        logger.info("\nNext Steps:")
        logger.info("  1. Review migration log: data/logs/migration.log")
        logger.info("  2. Test the system: python src/main.py")
        logger.info("  3. Test web interface: python app.py")
        logger.info("  4. Run validation: python scripts/validate_migration.py")
        
        return 0
    else:
        logger.error("❌ MIGRATION FAILED")
        logger.error("=" * 70)
        logger.error(f"Stage: {status.stage.value}")
        logger.error(f"Progress: {status.progress * 100:.1f}%")
        
        if status.errors:
            logger.error(f"\nErrors ({len(status.errors)}):")
            for error in status.errors:
                logger.error(f"  - {error}")
        
        logger.error("\nTroubleshooting:")
        logger.error("  1. Check migration log: data/logs/migration.log")
        logger.error("  2. Verify dependencies: pip install -r requirements.txt")
        logger.error("  3. Check disk space: df -h")
        logger.error("  4. Rollback if needed: python scripts/rollback_migration.py")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
