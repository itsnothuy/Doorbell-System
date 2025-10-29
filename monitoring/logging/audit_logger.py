#!/usr/bin/env python3
"""
Audit Logger

Security and compliance audit logging with comprehensive event tracking.
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class AuditEventType(Enum):
    """Types of audit events."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SECURITY_EVENT = "security_event"
    SYSTEM_CHANGE = "system_change"
    FACE_RECOGNITION = "face_recognition"
    NOTIFICATION_SENT = "notification_sent"


class AuditLogger:
    """
    Audit logger for security and compliance tracking.

    Features:
    - Tamper-evident logging with hashes
    - Structured audit events
    - Compliance-ready format
    - Retention management
    """

    def __init__(self, audit_log_path: Optional[Path] = None):
        """
        Initialize audit logger.

        Args:
            audit_log_path: Path to audit log file
        """
        self.audit_log_path = audit_log_path or Path("data/logs/audit.log")
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)

        # Create file handler for audit logs
        if not self.logger.handlers:
            handler = logging.FileHandler(self.audit_log_path)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

        self.previous_hash = self._get_last_hash()

    def _get_last_hash(self) -> str:
        """Get hash of last audit entry."""
        try:
            if self.audit_log_path.exists():
                with open(self.audit_log_path, "r") as f:
                    lines = f.readlines()
                    if lines:
                        last_entry = json.loads(lines[-1])
                        return last_entry.get("hash", "0")
        except Exception:
            pass
        return "0"

    def _calculate_hash(self, event_data: Dict[str, Any]) -> str:
        """Calculate hash for tamper-evident logging."""
        hash_input = json.dumps(event_data, sort_keys=True) + self.previous_hash
        return hashlib.sha256(hash_input.encode()).hexdigest()

    def log_event(
        self,
        event_type: AuditEventType,
        actor: str,
        action: str,
        resource: str,
        result: str,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an audit event.

        Args:
            event_type: Type of audit event
            actor: Who performed the action
            action: What action was performed
            resource: What resource was affected
            result: Result of the action (success/failure)
            additional_data: Additional event data
        """
        event_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type.value,
            "actor": actor,
            "action": action,
            "resource": resource,
            "result": result,
        }

        if additional_data:
            event_data["data"] = additional_data

        # Calculate hash for tamper-evident logging
        event_data["hash"] = self._calculate_hash(event_data)

        # Log the event
        self.logger.info(json.dumps(event_data))

        # Update previous hash
        self.previous_hash = event_data["hash"]

    def log_authentication(
        self, user: str, success: bool, method: str = "password", **kwargs: Any
    ) -> None:
        """Log authentication event."""
        self.log_event(
            AuditEventType.AUTHENTICATION,
            actor=user,
            action="authenticate",
            resource="system",
            result="success" if success else "failure",
            additional_data={"method": method, **kwargs},
        )

    def log_authorization(self, user: str, resource: str, granted: bool, **kwargs: Any) -> None:
        """Log authorization event."""
        self.log_event(
            AuditEventType.AUTHORIZATION,
            actor=user,
            action="authorize",
            resource=resource,
            result="granted" if granted else "denied",
            additional_data=kwargs,
        )

    def log_data_access(self, user: str, resource: str, operation: str, **kwargs: Any) -> None:
        """Log data access event."""
        self.log_event(
            AuditEventType.DATA_ACCESS,
            actor=user,
            action=operation,
            resource=resource,
            result="success",
            additional_data=kwargs,
        )

    def log_data_modification(
        self, user: str, resource: str, operation: str, changes: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Log data modification event."""
        self.log_event(
            AuditEventType.DATA_MODIFICATION,
            actor=user,
            action=operation,
            resource=resource,
            result="success",
            additional_data={"changes": changes, **kwargs},
        )

    def log_security_event(
        self, event_type: str, severity: str, description: str, **kwargs: Any
    ) -> None:
        """Log security event."""
        self.log_event(
            AuditEventType.SECURITY_EVENT,
            actor="system",
            action=event_type,
            resource="security",
            result=severity,
            additional_data={"description": description, **kwargs},
        )

    def log_system_change(
        self, component: str, change_type: str, description: str, **kwargs: Any
    ) -> None:
        """Log system configuration change."""
        self.log_event(
            AuditEventType.SYSTEM_CHANGE,
            actor="system",
            action=change_type,
            resource=component,
            result="applied",
            additional_data={"description": description, **kwargs},
        )

    def log_face_recognition(
        self,
        result_type: str,
        confidence: float,
        person_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log face recognition event."""
        self.log_event(
            AuditEventType.FACE_RECOGNITION,
            actor="system",
            action="recognize_face",
            resource="face_database",
            result=result_type,
            additional_data={
                "confidence": confidence,
                "person_id": person_id,
                **kwargs,
            },
        )

    def log_notification(
        self, channel: str, recipient: str, notification_type: str, success: bool, **kwargs: Any
    ) -> None:
        """Log notification event."""
        self.log_event(
            AuditEventType.NOTIFICATION_SENT,
            actor="system",
            action="send_notification",
            resource=f"{channel}:{recipient}",
            result="success" if success else "failure",
            additional_data={"type": notification_type, **kwargs},
        )

    def verify_integrity(self, start_index: int = 0, end_index: Optional[int] = None) -> bool:
        """
        Verify integrity of audit log entries.

        Args:
            start_index: Starting index to verify
            end_index: Ending index to verify (None for all)

        Returns:
            True if integrity check passes
        """
        try:
            with open(self.audit_log_path, "r") as f:
                lines = f.readlines()[start_index:end_index]

            previous_hash = "0"
            for i, line in enumerate(lines):
                entry = json.loads(line)
                expected_hash = entry.get("hash")

                # Recalculate hash
                entry_copy = entry.copy()
                del entry_copy["hash"]
                calculated_hash = self._calculate_hash(entry_copy)

                if calculated_hash != expected_hash:
                    logging.error(
                        f"Audit log integrity check failed at index {start_index + i}"
                    )
                    return False

                previous_hash = expected_hash

            return True

        except Exception as e:
            logging.error(f"Failed to verify audit log integrity: {e}")
            return False
