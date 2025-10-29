#!/usr/bin/env python3
"""
Communication Module

High-performance message bus and error handling infrastructure.
"""

from src.communication.message_bus import (
    MessageBus,
    Message,
    MessagePriority,
    Subscription,
    RequestReplyBus,
    get_message_bus,
    init_message_bus,
)

from src.communication.error_handling import (
    ErrorLogger,
    ErrorRecoveryManager,
    ErrorReport,
    ErrorCategory,
    ErrorSeverity,
    CircuitBreaker,
    DeadLetterQueue,
)

__all__ = [
    # Message Bus
    'MessageBus',
    'Message',
    'MessagePriority',
    'Subscription',
    'RequestReplyBus',
    'get_message_bus',
    'init_message_bus',
    # Error Handling
    'ErrorLogger',
    'ErrorRecoveryManager',
    'ErrorReport',
    'ErrorCategory',
    'ErrorSeverity',
    'CircuitBreaker',
    'DeadLetterQueue',
]
