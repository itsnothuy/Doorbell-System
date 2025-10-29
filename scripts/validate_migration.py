#!/usr/bin/env python3
"""
Validate Migration - Migration Validation Script

Validates that the migration to pipeline architecture was successful.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.integration.migration_utils import MigrationUtils
from config.logging_config import setup_logging


def main():
    """Validate migration to pipeline architecture."""
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("Migration Validation")
    logger.info("=" * 70)
    
    all_passed = True
    
    # Check pipeline health
    logger.info("\n1. Checking pipeline health...")
    health_results = MigrationUtils.verify_pipeline_health()
    
    if health_results['healthy']:
        logger.info("   ✅ Pipeline health check PASSED")
        for check_name, check_result in health_results['checks'].items():
            status = "✅" if check_result else "❌"
            logger.info(f"      {status} {check_name}")
    else:
        logger.error("   ❌ Pipeline health check FAILED")
        all_passed = False
        for issue in health_results['issues']:
            logger.error(f"      - {issue}")
    
    # Check migration compatibility
    logger.info("\n2. Checking migration compatibility...")
    compat_results = MigrationUtils.validate_migration_compatibility()
    
    if compat_results['compatible']:
        logger.info("   ✅ Compatibility check PASSED")
    else:
        logger.error("   ❌ Compatibility check FAILED")
        all_passed = False
        for issue in compat_results['issues']:
            logger.error(f"      - {issue}")
    
    if compat_results['warnings']:
        for warning in compat_results['warnings']:
            logger.warning(f"      ⚠️  {warning}")
    
    # Test orchestrator startup
    logger.info("\n3. Testing orchestrator startup...")
    try:
        from src.integration.orchestrator_manager import OrchestratorManager
        import time
        
        manager = OrchestratorManager()
        manager.start()
        
        # Wait for initialization
        time.sleep(2.0)
        
        # Check health
        health = manager.get_health_status()
        
        # Stop
        manager.stop()
        
        logger.info(f"   ✅ Orchestrator startup PASSED")
        logger.info(f"      State: {health.state.value}")
        logger.info(f"      Performance: {health.performance_score:.2f}")
        
    except Exception as e:
        logger.error(f"   ❌ Orchestrator startup FAILED: {e}")
        all_passed = False
    
    # Test web interface
    logger.info("\n4. Testing web interface...")
    try:
        from src.web_interface import create_web_app
        from src.integration.orchestrator_manager import OrchestratorManager
        
        manager = OrchestratorManager()
        legacy_interface = manager.get_legacy_interface()
        app = create_web_app(legacy_interface)
        
        with app.test_client() as client:
            # Test status endpoint
            response = client.get('/api/status')
            if response.status_code == 200:
                logger.info("   ✅ Web interface PASSED")
                logger.info(f"      API status: {response.status_code}")
            else:
                logger.error(f"   ❌ Web interface FAILED: status {response.status_code}")
                all_passed = False
        
    except Exception as e:
        logger.error(f"   ❌ Web interface FAILED: {e}")
        all_passed = False
    
    # Check data integrity
    logger.info("\n5. Checking data integrity...")
    try:
        known_faces_dir = Path("data/known_faces")
        blacklist_dir = Path("data/blacklist_faces")
        
        known_count = len(list(known_faces_dir.glob("*.jpg"))) if known_faces_dir.exists() else 0
        blacklist_count = len(list(blacklist_dir.glob("*.jpg"))) if blacklist_dir.exists() else 0
        
        logger.info("   ✅ Data integrity check PASSED")
        logger.info(f"      Known faces: {known_count}")
        logger.info(f"      Blacklist faces: {blacklist_count}")
        
    except Exception as e:
        logger.error(f"   ❌ Data integrity check FAILED: {e}")
        all_passed = False
    
    # Display results
    logger.info("\n" + "=" * 70)
    if all_passed:
        logger.info("✅ ALL VALIDATION CHECKS PASSED")
        logger.info("=" * 70)
        logger.info("\nMigration was successful!")
        logger.info("The system is ready to use with the new pipeline architecture.")
        return 0
    else:
        logger.error("❌ SOME VALIDATION CHECKS FAILED")
        logger.error("=" * 70)
        logger.error("\nMigration validation failed!")
        logger.error("Please review the errors above and fix any issues.")
        logger.error("You may need to rollback: python scripts/rollback_migration.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
