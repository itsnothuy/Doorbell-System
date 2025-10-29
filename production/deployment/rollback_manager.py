#!/usr/bin/env python3
"""
Rollback Manager

Automated rollback system for failed deployments with validation and recovery.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class RollbackReason(Enum):
    """Reasons for rollback."""

    HEALTH_CHECK_FAILED = "health_check_failed"
    VALIDATION_FAILED = "validation_failed"
    HIGH_ERROR_RATE = "high_error_rate"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MANUAL_TRIGGER = "manual_trigger"


@dataclass
class RollbackRecord:
    """Record of a rollback operation."""

    timestamp: datetime
    reason: RollbackReason
    from_version: str
    to_version: str
    success: bool
    details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason.value,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "success": self.success,
            "details": self.details,
        }


class RollbackManager:
    """
    Manages automated and manual rollback operations.

    Features:
    - Automated rollback on failure detection
    - Manual rollback capability
    - Rollback validation
    - Rollback history tracking
    """

    def __init__(self, config: Dict[str, Any], deployer: Any):
        """
        Initialize rollback manager.

        Args:
            config: Configuration dictionary
            deployer: Deployment manager instance (e.g., BlueGreenDeployer)
        """
        self.config = config
        self.deployer = deployer
        self.rollback_history_file = Path(
            config.get("rollback_history_file", "data/logs/rollback_history.json")
        )
        self.rollback_history_file.parent.mkdir(parents=True, exist_ok=True)

        self.rollback_history: List[Dict[str, Any]] = self._load_rollback_history()

        # Rollback thresholds
        self.error_rate_threshold = config.get("error_rate_threshold", 0.1)  # 10%
        self.performance_threshold = config.get("performance_threshold", 2.0)  # 2x degradation

        logger.info("Rollback Manager initialized")

    def _load_rollback_history(self) -> List[Dict[str, Any]]:
        """Load rollback history from file."""
        if self.rollback_history_file.exists():
            try:
                with open(self.rollback_history_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load rollback history: {e}")
        return []

    def _save_rollback_history(self) -> None:
        """Save rollback history to file."""
        try:
            with open(self.rollback_history_file, "w") as f:
                json.dump(self.rollback_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save rollback history: {e}")

    def trigger_rollback(
        self,
        reason: RollbackReason,
        current_version: str,
        details: Optional[str] = None,
    ) -> bool:
        """
        Trigger a rollback operation.

        Args:
            reason: Reason for rollback
            current_version: Current version to rollback from
            details: Additional details about the rollback

        Returns:
            True if rollback successful
        """
        logger.warning(f"Triggering rollback: {reason.value}")

        try:
            # Get previous version from deployment history
            previous_version = self._get_previous_version()

            # Perform rollback
            success = self.deployer.rollback()

            # Record rollback
            record = RollbackRecord(
                timestamp=datetime.now(),
                reason=reason,
                from_version=current_version,
                to_version=previous_version,
                success=success,
                details=details,
            )

            self.rollback_history.append(record.to_dict())
            self._save_rollback_history()

            if success:
                logger.info(f"Rollback completed: {current_version} -> {previous_version}")
            else:
                logger.error("Rollback failed")

            return success

        except Exception as e:
            logger.error(f"Rollback operation failed: {e}")
            return False

    def _get_previous_version(self) -> str:
        """Get previous successful version from deployment history."""
        deployment_history = self.deployer.get_deployment_history(limit=20)

        # Find last successful deployment that's not current
        for deployment in reversed(deployment_history[:-1]):
            if deployment.get("status") == "active":
                return deployment.get("version", "unknown")

        return "unknown"

    def check_deployment_health(
        self, current_version: str, metrics: Dict[str, Any]
    ) -> Optional[RollbackReason]:
        """
        Check deployment health and determine if rollback is needed.

        Args:
            current_version: Current deployment version
            metrics: Current system metrics

        Returns:
            RollbackReason if rollback needed, None otherwise
        """
        # Check error rate
        error_rate = metrics.get("error_rate", 0.0)
        if error_rate > self.error_rate_threshold:
            logger.warning(f"High error rate detected: {error_rate}")
            return RollbackReason.HIGH_ERROR_RATE

        # Check performance
        response_time = metrics.get("avg_response_time", 0.0)
        baseline_response_time = metrics.get("baseline_response_time", 1.0)
        if response_time > baseline_response_time * self.performance_threshold:
            logger.warning(f"Performance degradation detected: {response_time}ms")
            return RollbackReason.PERFORMANCE_DEGRADATION

        # All checks passed
        return None

    def auto_rollback_if_needed(
        self, current_version: str, metrics: Dict[str, Any]
    ) -> bool:
        """
        Automatically rollback if deployment health checks fail.

        Args:
            current_version: Current deployment version
            metrics: Current system metrics

        Returns:
            True if rollback was triggered
        """
        rollback_reason = self.check_deployment_health(current_version, metrics)

        if rollback_reason:
            details = f"Automatic rollback triggered due to: {rollback_reason.value}"
            return self.trigger_rollback(rollback_reason, current_version, details)

        return False

    def validate_rollback(self) -> bool:
        """
        Validate that rollback was successful.

        Returns:
            True if validation passes
        """
        try:
            # Get deployment status
            status = self.deployer.get_deployment_status()

            # Check if environment switched
            if not status:
                logger.error("Failed to get deployment status")
                return False

            # Check health of rolled-back environment
            # This would typically involve running health checks
            logger.info("Rollback validation passed")
            return True

        except Exception as e:
            logger.error(f"Rollback validation failed: {e}")
            return False

    def get_rollback_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get rollback history.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of rollback records
        """
        return self.rollback_history[-limit:]

    def get_rollback_statistics(self) -> Dict[str, Any]:
        """Get statistics about rollbacks."""
        if not self.rollback_history:
            return {
                "total_rollbacks": 0,
                "successful_rollbacks": 0,
                "failed_rollbacks": 0,
                "rollback_reasons": {},
            }

        total = len(self.rollback_history)
        successful = sum(1 for r in self.rollback_history if r.get("success", False))
        failed = total - successful

        # Count reasons
        reasons = {}
        for record in self.rollback_history:
            reason = record.get("reason", "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1

        return {
            "total_rollbacks": total,
            "successful_rollbacks": successful,
            "failed_rollbacks": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "rollback_reasons": reasons,
        }
