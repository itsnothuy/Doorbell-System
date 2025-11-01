#!/usr/bin/env python3
"""
Event Processor - Central Event Processing System

Manages the complete lifecycle of security events from detection through enrichment,
persistence, and notification delivery. Coordinates all event-driven activities
in the Frigate-inspired architecture.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from src.pipeline.base_worker import PipelineWorker
from src.communication.message_bus import MessageBus, Message
from src.communication.events import (
    PipelineEvent,
    FaceRecognitionEvent,
    EventType,
    EventPriority
)
from src.storage.event_database import EventDatabase
from src.enrichment.base_enrichment import BaseEnrichment, EnrichmentResult
from src.enrichment.enrichment_orchestrator import EnrichmentOrchestrator
from src.enrichment.metadata_enrichment import MetadataEnrichment
from src.enrichment.web_events import WebEventsEnrichment, WebEventStreamer
from config.event_config import get_event_config

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Event processing stages."""
    RECEIVED = "received"
    VALIDATING = "validating"
    ENRICHING = "enriching"
    PERSISTING = "persisting"
    ROUTING = "routing"
    STREAMING = "streaming"
    COMPLETED = "completed"
    FAILED = "failed"


class PersistenceStatus(Enum):
    """Event persistence status."""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProcessedEvent:
    """Fully processed event with all enrichments."""
    original_event: PipelineEvent
    enriched_data: Dict[str, Any]
    processing_metadata: Dict[str, Any]
    persistence_status: PersistenceStatus
    enrichment_results: Dict[str, EnrichmentResult] = field(default_factory=dict)
    notification_targets: List[str] = field(default_factory=list)


@dataclass
class EventState:
    """Event state tracking."""
    event_id: str
    current_stage: ProcessingStage
    start_time: float
    last_update: float
    retry_count: int = 0
    errors: List[str] = field(default_factory=list)


class EventProcessor(PipelineWorker):
    """Central event processing system for security events."""
    
    def _get_config_value(self, config, key, default=None):
        """Helper method to handle both dict and config objects."""
        if hasattr(config, key):
            return getattr(config, key, default)
        elif hasattr(config, 'get'):
            return config.get(key, default)
        else:
            return default
    
    def __init__(self, message_bus: MessageBus, config: Optional[Dict[str, Any]] = None):
        """
        Initialize event processor.
        
        Args:
            message_bus: Message bus for event communication
            config: Event processing configuration
        """
        # Get configuration
        if config is None:
            config = get_event_config()
        
        self.event_config = config
        
        # Use helper method for config access
        db_config = self._get_config_value(config, 'database_config', {})
        web_config = self._get_config_value(config, 'web_streaming_config', {})
        enrichment_config = self._get_config_value(config, 'enrichment_config', {})
        
        # Initialize database
        db_path = self._get_config_value(db_config, 'path', 'data/events.db')
        self.event_database = EventDatabase(
            db_path=db_path,
            config=db_config if isinstance(db_config, dict) else {}
        )
        
        # Initialize web event streamer
        self.web_streamer = WebEventStreamer(web_config)
        
        # Initialize enrichment processors
        self.enrichment_processors = self._initialize_enrichment_processors()
        
        # Initialize enrichment orchestrator
        self.enrichment_orchestrator = EnrichmentOrchestrator(
            self.enrichment_processors,
            enrichment_config
        )
        
        # Event state tracking
        self.active_events: Dict[str, EventState] = {}
        
        # Performance metrics
        self.events_processed = 0
        self.events_failed = 0
        self.enrichment_success_rate = 0.0
        self.avg_processing_time = 0.0
        self.total_processing_time = 0.0
        self.persistence_success_count = 0
        self.persistence_failure_count = 0
        
        # Configuration
        performance_config = self._get_config_value(config, 'performance_config', {})
        self.max_concurrent_events = self._get_config_value(performance_config, 'max_concurrent_events', 100)
        self.event_timeout = self._get_config_value(performance_config, 'event_timeout', 60.0)
        
        # Call parent constructor
        base_config = self._get_config_value(config, 'base_config', {})
        super().__init__(message_bus, base_config)
        
        logger.info(f"Initialized {self.worker_id} with {len(self.enrichment_processors)} enrichment processors")
    
    def _initialize_enrichment_processors(self) -> List[BaseEnrichment]:
        """Initialize enrichment processors from configuration."""
        processors = []
        enrichment_config = self._get_config_value(self.event_config, 'enrichment_config', {})
        enabled_processors = self._get_config_value(enrichment_config, 'enabled_processors', [])
        
        # Metadata enrichment (priority 1)
        if 'metadata_enrichment' in enabled_processors:
            processors.append(MetadataEnrichment({
                'priority': 1,
                'enabled': True,
                'include_system_info': True,
                'include_timestamps': True,
                'include_processing_context': True
            }))
        
        # Web events streaming (priority 8)
        if 'web_events' in enabled_processors:
            web_config = self._get_config_value(self.event_config, 'web_streaming_config', {})
            processors.append(WebEventsEnrichment(
                {
                    'priority': 8,
                    'enabled': self._get_config_value(web_config, 'enabled', True),
                    'streaming': web_config,
                    'stream_all_events': True
                },
                streamer=self.web_streamer
            ))
        
        logger.info(f"Initialized {len(processors)} enrichment processors")
        return processors
    
    def _setup_subscriptions(self):
        """Setup message bus subscriptions."""
        # Subscribe to face recognition events
        self.message_bus.subscribe('faces_recognized', self.handle_recognition_event, self.worker_id)
        
        # Subscribe to other event types
        self.message_bus.subscribe('motion_detected', self.handle_motion_event, self.worker_id)
        self.message_bus.subscribe('doorbell_pressed', self.handle_doorbell_event, self.worker_id)
        
        # Subscribe to system events
        self.message_bus.subscribe('system_event', self.handle_system_event, self.worker_id)
        self.message_bus.subscribe('system_shutdown', self.handle_shutdown, self.worker_id)
        
        logger.debug(f"{self.worker_id} subscriptions configured")
    
    def _initialize_worker(self):
        """Initialize event processor."""
        try:
            # Initialize database
            self.event_database.initialize()
            
            logger.info(f"{self.worker_id} initialized successfully")
            
        except Exception as e:
            logger.error(f"{self.worker_id} initialization failed: {e}")
            raise
    
    def handle_recognition_event(self, message: Message):
        """Handle face recognition event."""
        recognition_event: FaceRecognitionEvent = message.data
        
        try:
            logger.debug(f"Processing recognition event: {recognition_event.event_id}")
            
            # Process event through pipeline
            processed_event = self._process_event_pipeline(recognition_event)
            
            # Update metrics
            self.events_processed += 1
            self._update_processing_metrics(processed_event)
            
            # Publish completion event
            self._publish_completion_event(processed_event)
            
        except Exception as e:
            self.events_failed += 1
            self.error_count += 1
            logger.error(f"Recognition event processing failed: {e}", exc_info=True)
    
    def handle_motion_event(self, message: Message):
        """Handle motion detection event."""
        motion_event = message.data
        
        try:
            # Process as generic pipeline event
            processed_event = self._process_event_pipeline(motion_event)
            self.events_processed += 1
            
        except Exception as e:
            self.events_failed += 1
            logger.error(f"Motion event processing failed: {e}")
    
    def handle_doorbell_event(self, message: Message):
        """Handle doorbell press event."""
        doorbell_event = message.data
        
        try:
            # Process as generic pipeline event
            processed_event = self._process_event_pipeline(doorbell_event)
            self.events_processed += 1
            
        except Exception as e:
            self.events_failed += 1
            logger.error(f"Doorbell event processing failed: {e}")
    
    def handle_system_event(self, message: Message):
        """Handle system-level event."""
        system_event = message.data
        
        try:
            # Process as generic pipeline event
            processed_event = self._process_event_pipeline(system_event)
            self.events_processed += 1
            
        except Exception as e:
            logger.error(f"System event processing failed: {e}")
    
    def _process_event_pipeline(self, event: PipelineEvent) -> ProcessedEvent:
        """
        Complete event processing pipeline.
        
        Args:
            event: Event to process
            
        Returns:
            ProcessedEvent with complete processing results
        """
        start_time = time.time()
        event_id = event.event_id
        
        # Create event state
        event_state = EventState(
            event_id=event_id,
            current_stage=ProcessingStage.RECEIVED,
            start_time=start_time,
            last_update=start_time
        )
        self.active_events[event_id] = event_state
        
        try:
            # 1. Validate event
            event_state.current_stage = ProcessingStage.VALIDATING
            if not self._validate_event(event):
                raise ValueError("Event validation failed")
            
            # 2. Apply enrichment processors
            event_state.current_stage = ProcessingStage.ENRICHING
            enrichment_results = self.enrichment_orchestrator.process_event(event)
            
            # 3. Persist event
            event_state.current_stage = ProcessingStage.PERSISTING
            persistence_status = self._persist_event(event, enrichment_results)
            
            # 4. Update processing metadata
            event_state.current_stage = ProcessingStage.COMPLETED
            processing_metadata = {
                'start_time': start_time,
                'end_time': time.time(),
                'processing_duration': time.time() - start_time,
                'enrichment_count': len(enrichment_results),
                'stage': ProcessingStage.COMPLETED.value
            }
            
            # Create processed event
            processed_event = ProcessedEvent(
                original_event=event,
                enriched_data=self._merge_enrichment_data(enrichment_results),
                processing_metadata=processing_metadata,
                persistence_status=persistence_status,
                enrichment_results=enrichment_results
            )
            
            self.processed_count += 1
            logger.debug(f"Event {event_id} processed successfully in {processing_metadata['processing_duration']*1000:.2f}ms")
            
            return processed_event
            
        except Exception as e:
            event_state.current_stage = ProcessingStage.FAILED
            event_state.errors.append(str(e))
            
            logger.error(f"Event processing failed for {event_id}: {e}")
            
            # Create error processed event
            return ProcessedEvent(
                original_event=event,
                enriched_data={'error': str(e)},
                processing_metadata={
                    'start_time': start_time,
                    'end_time': time.time(),
                    'processing_duration': time.time() - start_time,
                    'stage': ProcessingStage.FAILED.value,
                    'errors': event_state.errors
                },
                persistence_status=PersistenceStatus.FAILED
            )
        finally:
            # Clean up event state
            if event_id in self.active_events:
                del self.active_events[event_id]
    
    def _validate_event(self, event: PipelineEvent) -> bool:
        """
        Validate event structure and required fields.
        
        Args:
            event: Event to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            if not hasattr(event, 'event_id') or not event.event_id:
                logger.warning("Event missing event_id")
                return False
            
            if not hasattr(event, 'event_type'):
                logger.warning("Event missing event_type")
                return False
            
            if not hasattr(event, 'timestamp'):
                logger.warning("Event missing timestamp")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Event validation error: {e}")
            return False
    
    def _persist_event(self, event: PipelineEvent, 
                      enrichment_results: Dict[str, Any]) -> PersistenceStatus:
        """
        Persist event to database with enrichment data.
        
        Args:
            event: Event to persist
            enrichment_results: Enrichment processing results
            
        Returns:
            PersistenceStatus indicating success or failure
        """
        try:
            success = self.event_database.store_event(event, enrichment_results)
            
            if success:
                self.persistence_success_count += 1
                return PersistenceStatus.SUCCESS
            else:
                self.persistence_failure_count += 1
                return PersistenceStatus.FAILED
                
        except Exception as e:
            logger.error(f"Event persistence failed: {e}")
            self.persistence_failure_count += 1
            return PersistenceStatus.FAILED
    
    def _merge_enrichment_data(self, enrichment_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge enrichment data from all processors.
        
        Args:
            enrichment_results: Dictionary of enrichment results
            
        Returns:
            Merged enrichment data dictionary
        """
        merged_data = {}
        
        for processor_name, result in enrichment_results.items():
            if hasattr(result, 'enriched_data'):
                enriched_data = result.enriched_data
            elif isinstance(result, dict):
                enriched_data = result.get('enriched_data', {})
            else:
                continue
            
            merged_data[processor_name] = enriched_data
        
        return merged_data
    
    def _update_processing_metrics(self, processed_event: ProcessedEvent):
        """Update processing metrics from completed event."""
        metadata = processed_event.processing_metadata
        processing_time = metadata.get('processing_duration', 0.0)
        
        self.total_processing_time += processing_time
        
        # Update average processing time
        self.avg_processing_time = (
            self.total_processing_time / max(1, self.events_processed)
        )
        
        # Update enrichment success rate
        enrichment_count = metadata.get('enrichment_count', 0)
        if enrichment_count > 0:
            successful_enrichments = sum(
                1 for r in processed_event.enrichment_results.values()
                if (hasattr(r, 'success') and r.success) or 
                   (isinstance(r, dict) and r.get('success', False))
            )
            self.enrichment_success_rate = successful_enrichments / enrichment_count
    
    def _publish_completion_event(self, processed_event: ProcessedEvent):
        """Publish event processing completion."""
        try:
            completion_data = {
                'event_id': processed_event.original_event.event_id,
                'processing_metadata': processed_event.processing_metadata,
                'persistence_status': processed_event.persistence_status.value,
                'enrichment_count': len(processed_event.enrichment_results)
            }
            
            self.message_bus.publish('event_processed', completion_data, source=self.worker_id)
            
        except Exception as e:
            logger.error(f"Failed to publish completion event: {e}")
    
    def _cleanup_worker(self):
        """Cleanup event processor resources."""
        try:
            # Close database
            if self.event_database:
                self.event_database.close()
            
            logger.info(f"{self.worker_id} cleanup completed")
            
        except Exception as e:
            logger.error(f"{self.worker_id} cleanup failed: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event processor performance metrics."""
        base_metrics = super().get_metrics()
        
        event_metrics = {
            'events_processed': self.events_processed,
            'events_failed': self.events_failed,
            'success_rate': (self.events_processed / max(1, self.events_processed + self.events_failed)),
            'avg_processing_time': self.avg_processing_time,
            'total_processing_time': self.total_processing_time,
            'enrichment_success_rate': self.enrichment_success_rate,
            'persistence_success_count': self.persistence_success_count,
            'persistence_failure_count': self.persistence_failure_count,
            'persistence_success_rate': (
                self.persistence_success_count / 
                max(1, self.persistence_success_count + self.persistence_failure_count)
            ),
            'active_events': len(self.active_events),
            'max_concurrent_events': self.max_concurrent_events,
            'database_stats': self.event_database.get_statistics(days=1),
            'enrichment_metrics': self.enrichment_orchestrator.get_metrics(),
            'web_streaming_stats': self.web_streamer.get_stats()
        }
        
        return {**base_metrics, **event_metrics}
