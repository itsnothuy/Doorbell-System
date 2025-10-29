#!/usr/bin/env python3
"""
Blue-Green Deployment Manager

Zero-downtime deployment system with automated rollback capabilities.
"""

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    """Deployment environment types."""

    BLUE = "blue"
    GREEN = "green"


class DeploymentStatus(Enum):
    """Deployment status types."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    ACTIVE = "active"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeploymentInfo:
    """Information about a deployment."""

    environment: DeploymentEnvironment
    version: str
    timestamp: datetime
    status: DeploymentStatus
    health_check_passed: bool = False
    validation_passed: bool = False
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "environment": self.environment.value,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "health_check_passed": self.health_check_passed,
            "validation_passed": self.validation_passed,
            "metadata": self.metadata or {},
        }


class BlueGreenDeployer:
    """
    Blue-Green Deployment Manager.

    Implements zero-downtime deployment with:
    - Parallel environment management
    - Health checks and validation
    - Automated traffic switching
    - Instant rollback capability
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize blue-green deployer.

        Args:
            config: Deployment configuration
        """
        self.config = config
        self.deployment_dir = Path(config.get("deployment_dir", "/opt/doorbell-system"))
        self.blue_dir = self.deployment_dir / "blue"
        self.green_dir = self.deployment_dir / "green"
        self.current_link = self.deployment_dir / "current"
        self.history_file = self.deployment_dir / "deployment_history.json"

        # Create directories
        self.blue_dir.mkdir(parents=True, exist_ok=True)
        self.green_dir.mkdir(parents=True, exist_ok=True)

        # Deployment state
        self.active_environment = self._get_active_environment()
        self.deployment_history: List[Dict[str, Any]] = self._load_deployment_history()

        logger.info(f"Blue-Green Deployer initialized. Active: {self.active_environment.value}")

    def _get_active_environment(self) -> DeploymentEnvironment:
        """Get currently active environment."""
        if self.current_link.exists():
            target = self.current_link.resolve()
            if target == self.blue_dir:
                return DeploymentEnvironment.BLUE
            elif target == self.green_dir:
                return DeploymentEnvironment.GREEN

        # Default to blue if not set
        return DeploymentEnvironment.BLUE

    def _get_inactive_environment(self) -> DeploymentEnvironment:
        """Get currently inactive environment."""
        return (
            DeploymentEnvironment.GREEN
            if self.active_environment == DeploymentEnvironment.BLUE
            else DeploymentEnvironment.BLUE
        )

    def _get_environment_dir(self, environment: DeploymentEnvironment) -> Path:
        """Get directory for environment."""
        return self.blue_dir if environment == DeploymentEnvironment.BLUE else self.green_dir

    def _load_deployment_history(self) -> List[Dict[str, Any]]:
        """Load deployment history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load deployment history: {e}")
        return []

    def _save_deployment_history(self) -> None:
        """Save deployment history to file."""
        try:
            with open(self.history_file, "w") as f:
                json.dump(self.deployment_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save deployment history: {e}")

    def deploy(self, version: str, source_path: Path, dry_run: bool = False) -> bool:
        """
        Deploy new version using blue-green strategy.

        Args:
            version: Version identifier
            source_path: Path to deployment package
            dry_run: If True, simulate deployment without actual changes

        Returns:
            True if deployment successful
        """
        logger.info(f"Starting blue-green deployment for version {version}")

        if dry_run:
            logger.info("DRY RUN MODE - No actual changes will be made")

        try:
            # Determine target environment
            target_env = self._get_inactive_environment()
            target_dir = self._get_environment_dir(target_env)

            # Create deployment info
            deployment = DeploymentInfo(
                environment=target_env,
                version=version,
                timestamp=datetime.now(),
                status=DeploymentStatus.PENDING,
                metadata={"source_path": str(source_path), "dry_run": dry_run},
            )

            # Step 1: Deploy to inactive environment
            logger.info(f"Deploying to {target_env.value} environment")
            if not dry_run:
                self._deploy_to_environment(source_path, target_dir)
            deployment.status = DeploymentStatus.IN_PROGRESS

            # Step 2: Health check
            logger.info("Performing health checks")
            if not dry_run:
                health_ok = self._health_check(target_dir)
                deployment.health_check_passed = health_ok
                if not health_ok:
                    logger.error("Health check failed")
                    deployment.status = DeploymentStatus.FAILED
                    self._record_deployment(deployment)
                    return False
            else:
                deployment.health_check_passed = True

            # Step 3: Validation
            logger.info("Validating deployment")
            deployment.status = DeploymentStatus.VALIDATING
            if not dry_run:
                validation_ok = self._validate_deployment(target_dir)
                deployment.validation_passed = validation_ok
                if not validation_ok:
                    logger.error("Deployment validation failed")
                    deployment.status = DeploymentStatus.FAILED
                    self._record_deployment(deployment)
                    return False
            else:
                deployment.validation_passed = True

            # Step 4: Switch traffic
            logger.info(f"Switching traffic to {target_env.value}")
            if not dry_run:
                self._switch_environment(target_env)
                self.active_environment = target_env

            deployment.status = DeploymentStatus.ACTIVE

            # Record successful deployment
            self._record_deployment(deployment)

            logger.info(f"Deployment {version} completed successfully")
            return True

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            deployment.status = DeploymentStatus.FAILED
            self._record_deployment(deployment)
            return False

    def _deploy_to_environment(self, source_path: Path, target_dir: Path) -> None:
        """
        Deploy code to target environment.

        Args:
            source_path: Source deployment package
            target_dir: Target directory
        """
        logger.info(f"Copying deployment files from {source_path} to {target_dir}")

        # Backup existing deployment
        if target_dir.exists():
            backup_dir = target_dir.parent / f"{target_dir.name}_backup_{int(time.time())}"
            shutil.move(str(target_dir), str(backup_dir))
            logger.info(f"Backed up existing deployment to {backup_dir}")

        # Copy new deployment
        if source_path.is_dir():
            shutil.copytree(source_path, target_dir)
        else:
            # Assume it's an archive
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.unpack_archive(str(source_path), str(target_dir))

        logger.info("Deployment files copied successfully")

    def _health_check(self, environment_dir: Path) -> bool:
        """
        Perform health check on deployed environment.

        Args:
            environment_dir: Directory of deployed environment

        Returns:
            True if health check passes
        """
        try:
            # Check if required files exist
            required_files = ["app.py", "src", "config"]
            for required_file in required_files:
                file_path = environment_dir / required_file
                if not file_path.exists():
                    logger.error(f"Required file/directory not found: {required_file}")
                    return False

            # Run application health check script if available
            health_check_script = environment_dir / "scripts" / "health_check.sh"
            if health_check_script.exists():
                result = subprocess.run(
                    [str(health_check_script)],
                    cwd=environment_dir,
                    capture_output=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    logger.error(f"Health check script failed: {result.stderr.decode()}")
                    return False

            logger.info("Health check passed")
            return True

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def _validate_deployment(self, environment_dir: Path) -> bool:
        """
        Validate deployment before switching traffic.

        Args:
            environment_dir: Directory of deployed environment

        Returns:
            True if validation passes
        """
        try:
            # Run validation tests
            validation_script = environment_dir / "scripts" / "validate_deployment.sh"
            if validation_script.exists():
                result = subprocess.run(
                    [str(validation_script)],
                    cwd=environment_dir,
                    capture_output=True,
                    timeout=60,
                )
                if result.returncode != 0:
                    logger.error(f"Validation script failed: {result.stderr.decode()}")
                    return False

            logger.info("Deployment validation passed")
            return True

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def _switch_environment(self, target_env: DeploymentEnvironment) -> None:
        """
        Switch active environment by updating symlink.

        Args:
            target_env: Target environment to switch to
        """
        target_dir = self._get_environment_dir(target_env)

        # Remove existing symlink
        if self.current_link.exists() or self.current_link.is_symlink():
            self.current_link.unlink()

        # Create new symlink
        self.current_link.symlink_to(target_dir)

        logger.info(f"Switched to {target_env.value} environment")

    def _record_deployment(self, deployment: DeploymentInfo) -> None:
        """Record deployment in history."""
        self.deployment_history.append(deployment.to_dict())
        self._save_deployment_history()

    def rollback(self) -> bool:
        """
        Rollback to previous deployment.

        Returns:
            True if rollback successful
        """
        logger.info("Initiating rollback")

        try:
            # Find last successful deployment in the other environment
            other_env = self._get_inactive_environment()

            logger.info(f"Rolling back to {other_env.value} environment")

            # Switch back to other environment
            self._switch_environment(other_env)
            self.active_environment = other_env

            # Record rollback
            rollback_info = DeploymentInfo(
                environment=other_env,
                version="rollback",
                timestamp=datetime.now(),
                status=DeploymentStatus.ROLLED_BACK,
            )
            self._record_deployment(rollback_info)

            logger.info("Rollback completed successfully")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        return {
            "active_environment": self.active_environment.value,
            "blue_exists": self.blue_dir.exists(),
            "green_exists": self.green_dir.exists(),
            "last_deployment": self.deployment_history[-1] if self.deployment_history else None,
            "deployment_count": len(self.deployment_history),
        }

    def get_deployment_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get deployment history.

        Args:
            limit: Maximum number of deployments to return

        Returns:
            List of deployment records
        """
        return self.deployment_history[-limit:]
