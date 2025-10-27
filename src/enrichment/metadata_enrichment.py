#!/usr/bin/env python3
"""
Metadata Enrichment - Add Contextual Metadata to Events

Enriches events with additional metadata such as timestamps, environment info,
processing context, and event relationships.
"""

import time
import logging
import platform
from typing import Dict, Any
from datetime import datetime

from src.enrichment.base_enrichment import BaseEnrichment, EnrichmentResult, EnrichmentStatus
from src.communication.events import PipelineEvent, FaceRecognitionEvent

logger = logging.getLogger(__name__)


class MetadataEnrichment(BaseEnrichment):
    """Enriches events with contextual metadata."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metadata enrichment processor.
        
        Args:
            config: Configuration dictionary
        """
        # Set default priority
        if 'priority' not in config:
            config['priority'] = 1  # Run early in the pipeline
        
        super().__init__(config)
        
        # Configuration
        self.include_system_info = config.get('include_system_info', True)
        self.include_timestamps = config.get('include_timestamps', True)
        self.include_processing_context = config.get('include_processing_context', True)
        
        # Cache system information
        self._system_info = self._collect_system_info() if self.include_system_info else {}
        
        logger.info(f"Initialized {self.name} enrichment processor")
    
    def can_process(self, event: Any) -> bool:
        """Check if this enrichment can process the given event."""
        # Can process any PipelineEvent
        return isinstance(event, PipelineEvent)
    
    def enrich(self, event: PipelineEvent) -> EnrichmentResult:
        """
        Enrich event with metadata.
        
        Args:
            event: Event to enrich
            
        Returns:
            EnrichmentResult with metadata enrichment
        """
        start_time = time.time()
        
        try:
            enriched_data = {}
            
            # Add timestamp information
            if self.include_timestamps:
                enriched_data['timestamps'] = self._add_timestamp_info(event)
            
            # Add system information
            if self.include_system_info and self._system_info:
                enriched_data['system_info'] = self._system_info.copy()
            
            # Add processing context
            if self.include_processing_context:
                enriched_data['processing_context'] = self._add_processing_context(event)
            
            # Add event type specific metadata
            type_specific_metadata = self._add_type_specific_metadata(event)
            if type_specific_metadata:
                enriched_data['type_specific'] = type_specific_metadata
            
            # Add event relationships
            enriched_data['relationships'] = self._add_event_relationships(event)
            
            processing_time = time.time() - start_time
            
            return EnrichmentResult(
                success=True,
                enriched_data=enriched_data,
                processing_time=processing_time,
                processor_name=self.name,
                status=EnrichmentStatus.SUCCESS,
                metadata={
                    'enrichment_count': len(enriched_data),
                    'has_timestamps': 'timestamps' in enriched_data,
                    'has_system_info': 'system_info' in enriched_data
                }
            )
            
        except Exception as e:
            logger.error(f"Metadata enrichment failed: {e}", exc_info=True)
            return EnrichmentResult(
                success=False,
                enriched_data={},
                processing_time=time.time() - start_time,
                processor_name=self.name,
                error_message=str(e),
                status=EnrichmentStatus.FAILED
            )
    
    def _add_timestamp_info(self, event: PipelineEvent) -> Dict[str, Any]:
        """Add detailed timestamp information."""
        now = datetime.now()
        event_time = datetime.fromtimestamp(event.timestamp)
        
        return {
            'enrichment_timestamp': now.isoformat(),
            'enrichment_unix_time': time.time(),
            'event_timestamp': event_time.isoformat(),
            'event_unix_time': event.timestamp,
            'age_seconds': (now.timestamp() - event.timestamp),
            'iso_8601': now.isoformat(),
            'timezone': time.tzname[0]
        }
    
    def _add_processing_context(self, event: PipelineEvent) -> Dict[str, Any]:
        """Add processing context information."""
        context = {
            'processor_name': self.name,
            'enrichment_version': '1.0',
            'event_id': event.event_id,
            'event_type': event.event_type.name if hasattr(event, 'event_type') else 'unknown',
            'source': event.source if hasattr(event, 'source') else 'unknown',
            'priority': event.priority.name if hasattr(event, 'priority') else 'unknown'
        }
        
        # Add correlation tracking
        if hasattr(event, 'correlation_id') and event.correlation_id:
            context['correlation_id'] = event.correlation_id
        
        if hasattr(event, 'parent_event_id') and event.parent_event_id:
            context['parent_event_id'] = event.parent_event_id
        
        return context
    
    def _add_type_specific_metadata(self, event: PipelineEvent) -> Dict[str, Any]:
        """Add event type specific metadata."""
        metadata = {}
        
        # Face recognition event specific
        if isinstance(event, FaceRecognitionEvent):
            metadata['recognition_summary'] = {
                'total_recognitions': len(event.recognitions) if hasattr(event, 'recognitions') else 0,
                'known_count': getattr(event, 'known_count', 0),
                'unknown_count': getattr(event, 'unknown_count', 0),
                'blacklisted_count': getattr(event, 'blacklisted_count', 0),
                'recognition_time': getattr(event, 'recognition_time', 0.0)
            }
        
        return metadata
    
    def _add_event_relationships(self, event: PipelineEvent) -> Dict[str, Any]:
        """Add event relationship information."""
        relationships = {
            'has_parent': hasattr(event, 'parent_event_id') and event.parent_event_id is not None,
            'parent_id': getattr(event, 'parent_event_id', None),
            'correlation_id': getattr(event, 'correlation_id', None)
        }
        
        return relationships
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information (cached)."""
        try:
            return {
                'platform': platform.system(),
                'platform_version': platform.version(),
                'architecture': platform.machine(),
                'python_version': platform.python_version(),
                'hostname': platform.node()
            }
        except Exception as e:
            logger.warning(f"Failed to collect system info: {e}")
            return {}
    
    def get_dependencies(self) -> list:
        """No dependencies - runs first."""
        return []
