#!/usr/bin/env python3
"""
Main Entry Point - Pipeline Architecture

New main entry point using the PipelineOrchestrator instead of legacy system.
"""

import sys
import signal
import logging
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.integration.orchestrator_manager import OrchestratorManager, SystemState
from config.logging_config import setup_logging

logger = logging.getLogger(__name__)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    if hasattr(signal_handler, 'manager'):
        signal_handler.manager.stop()
    sys.exit(0)


def main(config: Optional[dict] = None) -> int:
    """
    Main entry point for the doorbell security system.
    
    Args:
        config: Optional configuration override
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Setup logging
        setup_logging(level=logging.INFO)
        logger.info("=" * 70)
        logger.info("Starting Doorbell Security System with Pipeline Architecture")
        logger.info("=" * 70)
        
        # Create and start orchestrator manager
        manager = OrchestratorManager(config)
        signal_handler.manager = manager  # Store for signal handler
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start the system
        manager.start()
        
        # Monitor and keep running
        try:
            import time
            last_status_time = 0
            
            while manager.state in [SystemState.STARTING, SystemState.RUNNING]:
                health = manager.get_health_status()
                
                if health.state == SystemState.ERROR:
                    logger.error(f"System error detected: {health.last_error}")
                    break
                
                # Log periodic status
                current_time = health.uptime
                if current_time - last_status_time > 300:  # Every 5 minutes
                    logger.info(
                        f"System Status - Uptime: {health.uptime:.1f}s, "
                        f"Performance: {health.performance_score:.2f}, "
                        f"Errors: {health.error_count}"
                    )
                    last_status_time = current_time
                
                # Brief sleep
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        
        # Cleanup
        manager.stop()
        logger.info("System shutdown complete")
        return 0
        
    except Exception as e:
        logger.error(f"System startup failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
