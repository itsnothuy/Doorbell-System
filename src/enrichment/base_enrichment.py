#!/usr/bin/env python3
"""
Base Enrichment - Abstract Base Class for Event Enrichment Processors

Defines the interface and common functionality for all enrichment processors
in the event processing pipeline.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class EnrichmentStatus(Enum):
    """Status of enrichment processing."""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class EnrichmentResult:
    """Result of enrichment processing."""
    success: bool
    enriched_data: Dict[str, Any]
    processing_time: float
    processor_name: str
    error_message: Optional[str] = None
    requires_retry: bool = False
    status: EnrichmentStatus = EnrichmentStatus.SUCCESS
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'success': self.success,
            'enriched_data': self.enriched_data,
            'processing_time': self.processing_time,
            'processor_name': self.processor_name,
            'error_message': self.error_message,
            'requires_retry': self.requires_retry,
            'status': self.status.value,
            'metadata': self.metadata
        }


class BaseEnrichment(ABC):
    """Abstract base class for event enrichment processors."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize enrichment processor.
        
        Args:
            config: Processor-specific configuration dictionary
        """
        self.config = config
        self.name = self.__class__.__name__
        self.priority = config.get('priority', 5)
        self.enabled = config.get('enabled', True)
        self.timeout = config.get('timeout', 5.0)
        
        # Performance metrics
        self.processed_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_processing_time = 0.0
        self.last_error: Optional[Exception] = None
        
        logger.info(f"Initialized enrichment processor: {self.name} (priority={self.priority})")
    
    @abstractmethod
    def can_process(self, event: Any) -> bool:
        """
        Check if this enrichment can process the given event.
        
        Args:
            event: Event to check
            
        Returns:
            True if this processor can handle the event, False otherwise
        """
        pass
    
    @abstractmethod
    def enrich(self, event: Any) -> EnrichmentResult:
        """
        Enrich the event with additional data.
        
        Args:
            event: Event to enrich
            
        Returns:
            EnrichmentResult with enrichment data and status
        """
        pass
    
    def process_event(self, event: Any) -> EnrichmentResult:
        """
        Process event with error handling and metrics tracking.
        
        Args:
            event: Event to process
            
        Returns:
            EnrichmentResult with processing outcome
        """
        start_time = time.time()
        
        try:
            # Check if processor is enabled
            if not self.enabled:
                return EnrichmentResult(
                    success=True,
                    enriched_data={},
                    processing_time=0.0,
                    processor_name=self.name,
                    status=EnrichmentStatus.SKIPPED,
                    metadata={'reason': 'processor_disabled'}
                )
            
            # Check if processor can handle this event
            if not self.can_process(event):
                return EnrichmentResult(
                    success=True,
                    enriched_data={},
                    processing_time=time.time() - start_time,
                    processor_name=self.name,
                    status=EnrichmentStatus.SKIPPED,
                    metadata={'reason': 'event_not_supported'}
                )
            
            # Process the event
            result = self.enrich(event)
            
            # Update metrics
            self.processed_count += 1
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            if result.success:
                self.success_count += 1
            else:
                self.failure_count += 1
                self.last_error = Exception(result.error_message or "Unknown error")
            
            # Update result with actual processing time
            result.processing_time = processing_time
            
            logger.debug(f"{self.name} processed event in {processing_time*1000:.2f}ms: {result.status.value}")
            
            return result
            
        except Exception as e:
            self.processed_count += 1
            self.failure_count += 1
            self.last_error = e
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            logger.error(f"{self.name} enrichment failed: {e}", exc_info=True)
            
            return EnrichmentResult(
                success=False,
                enriched_data={},
                processing_time=processing_time,
                processor_name=self.name,
                error_message=str(e),
                requires_retry=self._should_retry(e),
                status=EnrichmentStatus.FAILED,
                metadata={'exception_type': type(e).__name__}
            )
    
    def get_dependencies(self) -> List[str]:
        """
        Get list of enrichment processor dependencies.
        
        This processor will only run after all dependencies have completed.
        
        Returns:
            List of processor names that must run before this processor
        """
        return []
    
    def _should_retry(self, error: Exception) -> bool:
        """
        Determine if processing should be retried after an error.
        
        Args:
            error: Exception that occurred
            
        Returns:
            True if retry is recommended, False otherwise
        """
        # Retry on temporary failures like network issues or timeouts
        retryable_errors = (TimeoutError, ConnectionError, OSError)
        return isinstance(error, retryable_errors)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get enrichment processor performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        avg_time = (self.total_processing_time / self.processed_count 
                   if self.processed_count > 0 else 0.0)
        success_rate = (self.success_count / self.processed_count 
                       if self.processed_count > 0 else 0.0)
        
        return {
            'processor_name': self.name,
            'enabled': self.enabled,
            'priority': self.priority,
            'processed_count': self.processed_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': success_rate,
            'avg_processing_time': avg_time,
            'total_processing_time': self.total_processing_time,
            'last_error': str(self.last_error) if self.last_error else None
        }
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.processed_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_processing_time = 0.0
        self.last_error = None
        
        logger.debug(f"Reset metrics for {self.name}")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', priority={self.priority}, enabled={self.enabled})"
