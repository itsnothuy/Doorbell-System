#!/usr/bin/env python3
"""
Production Deployment - Production Deployment Script

Deploy the pipeline architecture to production with validation and monitoring.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.integration.deployment_manager import DeploymentManager
from config.logging_config import setup_logging


def main():
    """Execute production deployment."""
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("Doorbell Security System - Production Deployment")
    logger.info("=" * 70)
    
    logger.info("\nDeployment Plan:")
    logger.info("  1. Pre-deployment checks - System resources, dependencies, configuration")
    logger.info("  2. Deployment - Start pipeline orchestrator")
    logger.info("  3. Post-deployment validation - Web interface, API endpoints")
    logger.info("  4. Monitoring - Continuous health monitoring")
    logger.info("")
    
    # Confirm deployment
    try:
        response = input("Proceed with production deployment? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            logger.info("Deployment cancelled by user")
            return 0
    except KeyboardInterrupt:
        logger.info("\nDeployment cancelled by user")
        return 0
    
    # Create deployment manager
    deployment_config = {
        'health_check_interval': 30.0,
        'auto_recovery': True
    }
    
    manager = DeploymentManager(deployment_config)
    
    # Run deployment
    logger.info("\nStarting production deployment...")
    success = manager.deploy()
    
    # Display results
    status = manager.get_deployment_status()
    
    logger.info("\n" + "=" * 70)
    if success:
        logger.info("✅ PRODUCTION DEPLOYMENT COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Stage: {status.stage.value}")
        logger.info(f"Health Score: {status.health_score:.2f}")
        
        if status.warnings:
            logger.info(f"\nWarnings ({len(status.warnings)}):")
            for warning in status.warnings:
                logger.info(f"  - {warning}")
        
        logger.info("\nNext Steps:")
        logger.info("  1. Monitor system health: Check logs in data/logs/")
        logger.info("  2. Test functionality: python scripts/validate_migration.py")
        logger.info("  3. Access web interface: http://localhost:5000")
        logger.info("  4. Monitor performance metrics")
        
        return 0
    else:
        logger.error("❌ PRODUCTION DEPLOYMENT FAILED")
        logger.error("=" * 70)
        logger.error(f"Stage: {status.stage.value}")
        
        if status.errors:
            logger.error(f"\nErrors ({len(status.errors)}):")
            for error in status.errors:
                logger.error(f"  - {error}")
        
        logger.error("\nTroubleshooting:")
        logger.error("  1. Check deployment logs")
        logger.error("  2. Verify all pre-deployment checks passed")
        logger.error("  3. Review system resources")
        logger.error("  4. Contact support if issues persist")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
