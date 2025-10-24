#!/usr/bin/env python3
"""
Base Pipeline Worker

Abstract base class for all pipeline workers implementing the worker pattern
from the Frigate-inspired architecture.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from src.communication.message_bus import MessageBus, Message

logger = logging.getLogger(__name__)


class PipelineWorker(ABC):
    """Base class for pipeline workers."""
    
    def __init__(self, message_bus: MessageBus, config: Dict[str, Any]):
        self.message_bus = message_bus
        self.config = config
        self.worker_id = f"{self.__class__.__name__}_{int(time.time())}"
        
        # Worker state
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Performance metrics
        self.processed_count = 0
        self.error_count = 0
        self.start_time: Optional[float] = None
        
        # Subscribe to input events
        self._setup_subscriptions()
        
        logger.info(f"Initialized {self.worker_id}")
    
    @abstractmethod
    def _setup_subscriptions(self) -> None:
        """Setup message bus subscriptions. Override in subclasses."""
        pass
    
    def start(self) -> None:
        """Start the worker."""
        if self.running:
            logger.warning(f"{self.worker_id} already running")
            return
        
        self.running = True
        self.start_time = time.time()
        
        try:
            self._initialize_worker()
            self._worker_loop()
        except Exception as e:
            logger.error(f"{self.worker_id} failed: {e}")
            raise
        finally:
            self._cleanup_worker()
    
    def stop(self) -> None:
        """Stop the worker gracefully."""
        logger.info(f"Stopping {self.worker_id}...")
        self.running = False
        self.shutdown_event.set()
    
    def _initialize_worker(self) -> None:
        """Initialize worker-specific resources. Override in subclasses."""
        pass
    
    def _worker_loop(self) -> None:
        """Main worker processing loop."""
        while self.running and not self.shutdown_event.is_set():
            try:
                # Process events or perform work
                self._process_iteration()
                
                # Brief sleep to prevent busy waiting
                time.sleep(0.01)
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"{self.worker_id} processing error: {e}")
                time.sleep(0.1)  # Longer sleep on error
    
    def _process_iteration(self) -> None:
        """Override this method for worker-specific processing."""
        pass
    
    def _cleanup_worker(self) -> None:
        """Cleanup worker resources. Override in subclasses."""
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get worker performance metrics."""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        return {
            'worker_id': self.worker_id,
            'running': self.running,
            'uptime_seconds': uptime,
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(1, self.processed_count) if self.processed_count > 0 else 0,
            'processing_rate': self.processed_count / max(1, uptime) if uptime > 0 else 0
        }
    
    def handle_shutdown(self, message: Message) -> None:
        """Handle graceful shutdown signal."""
        logger.info(f"{self.worker_id} received shutdown signal")
        self.stop()
