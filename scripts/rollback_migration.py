#!/usr/bin/env python3
"""
Rollback Migration - Migration Rollback Script

Rollback the system to the state before migration to pipeline architecture.
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
    """Rollback migration to pipeline architecture."""
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("Doorbell Security System - Migration Rollback")
    logger.info("=" * 70)
    
    # Check if backup exists
    backup_dir = Path("data/migration_backup")
    if not backup_dir.exists():
        logger.error("No backup found!")
        logger.error("Cannot rollback without a backup.")
        return 1
    
    backup_path_file = backup_dir / "current_backup.txt"
    if not backup_path_file.exists():
        logger.error("Backup metadata not found!")
        logger.error("Cannot determine backup location.")
        return 1
    
    with open(backup_path_file, "r") as f:
        backup_path = f.read().strip()
    
    logger.info(f"\nBackup found: {backup_path}")
    logger.info("\n⚠️  WARNING: This will restore your system to the state before migration.")
    logger.info("Any changes made after migration will be lost.")
    logger.info("")
    
    # Confirm rollback
    try:
        response = input("Proceed with rollback? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            logger.info("Rollback cancelled by user")
            return 0
    except KeyboardInterrupt:
        logger.info("\nRollback cancelled by user")
        return 0
    
    # Create migration manager
    migration_config = {
        'backup_dir': 'data/migration_backup',
        'log_file': 'data/logs/rollback.log'
    }
    
    manager = MigrationManager(migration_config)
    
    # Run rollback
    logger.info("\nStarting rollback...")
    success = manager.rollback_migration()
    
    # Display results
    logger.info("\n" + "=" * 70)
    if success:
        logger.info("✅ ROLLBACK COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info("\nYour system has been restored to the pre-migration state.")
        logger.info("\nNext Steps:")
        logger.info("  1. Restart the system: python src/doorbell_security.py")
        logger.info("  2. Verify functionality")
        logger.info("  3. Review rollback log: data/logs/rollback.log")
        
        return 0
    else:
        logger.error("❌ ROLLBACK FAILED")
        logger.error("=" * 70)
        
        status = manager.get_migration_status()
        if status.errors:
            logger.error(f"\nErrors ({len(status.errors)}):")
            for error in status.errors:
                logger.error(f"  - {error}")
        
        logger.error("\nTroubleshooting:")
        logger.error("  1. Check rollback log: data/logs/rollback.log")
        logger.error("  2. Manually restore from backup: " + backup_path)
        logger.error("  3. Contact support if issues persist")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
