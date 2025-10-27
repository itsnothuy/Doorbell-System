#!/usr/bin/env python3
"""
Enrichment Orchestrator - Coordinates Multiple Enrichment Processors

Manages the execution of multiple enrichment processors in dependency order,
handles retries, and coordinates enrichment results.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict, deque

from src.enrichment.base_enrichment import (
    BaseEnrichment,
    EnrichmentResult,
    EnrichmentStatus
)

logger = logging.getLogger(__name__)


class EnrichmentOrchestrator:
    """Orchestrate multiple enrichment processors with dependency management."""
    
    def __init__(self, processors: List[BaseEnrichment], config: Dict[str, Any]):
        """
        Initialize enrichment orchestrator.
        
        Args:
            processors: List of enrichment processors
            config: Orchestrator configuration
        """
        self.processors = processors
        self.config = config
        
        # Configuration
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        self.timeout_per_processor = config.get('timeout_per_processor', 5.0)
        self.max_enrichment_time = config.get('max_enrichment_time', 10.0)
        self.retry_failed_enrichments = config.get('retry_failed_enrichments', True)
        
        # Sort processors by priority (lower priority values run first)
        self.processors = sorted(processors, key=lambda p: p.priority)
        
        # Build dependency graph
        self.dependency_graph = self._build_dependency_graph()
        self.processing_order = self._compute_processing_order()
        
        # Performance tracking
        self.events_processed = 0
        self.total_enrichment_time = 0.0
        self.enrichment_failures = 0
        self.retry_counts = defaultdict(int)
        
        logger.info(f"Initialized enrichment orchestrator with {len(processors)} processors")
        logger.debug(f"Processing order: {[p.name for p in self.processing_order]}")
    
    def process_event(self, event: Any) -> Dict[str, EnrichmentResult]:
        """
        Process event through all applicable enrichment processors.
        
        Args:
            event: Event to enrich
            
        Returns:
            Dictionary mapping processor name to EnrichmentResult
        """
        start_time = time.time()
        results = {}
        enriched_event = event
        retry_queue = deque()
        
        try:
            # Process in dependency order
            for processor in self.processing_order:
                if not processor.enabled:
                    continue
                
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > self.max_enrichment_time:
                    logger.warning(f"Enrichment timeout reached ({elapsed:.2f}s), skipping remaining processors")
                    break
                
                # Process with this enrichment processor
                result = self._process_with_processor(processor, enriched_event)
                results[processor.name] = result
                
                # If successful, apply enrichment to event
                if result.success and result.enriched_data:
                    enriched_event = self._apply_enrichment(enriched_event, result)
                elif result.requires_retry and self.retry_failed_enrichments:
                    # Queue for retry
                    retry_queue.append((processor, enriched_event, 0))
            
            # Process retry queue
            if retry_queue:
                self._process_retries(retry_queue, results)
            
            # Update metrics
            self.events_processed += 1
            processing_time = time.time() - start_time
            self.total_enrichment_time += processing_time
            
            # Count failures
            failure_count = sum(1 for r in results.values() if not r.success)
            self.enrichment_failures += failure_count
            
            logger.debug(f"Event enrichment completed in {processing_time*1000:.2f}ms with {len(results)} processors")
            
        except Exception as e:
            logger.error(f"Enrichment orchestration failed: {e}", exc_info=True)
        
        return results
    
    def _process_with_processor(self, processor: BaseEnrichment, event: Any) -> EnrichmentResult:
        """
        Process event with a single enrichment processor.
        
        Args:
            processor: Enrichment processor to use
            event: Event to process
            
        Returns:
            EnrichmentResult from processing
        """
        try:
            # Check dependencies
            if not self._dependencies_satisfied(processor, event):
                return EnrichmentResult(
                    success=False,
                    enriched_data={},
                    processing_time=0.0,
                    processor_name=processor.name,
                    error_message="Dependencies not satisfied",
                    status=EnrichmentStatus.SKIPPED,
                    metadata={'reason': 'missing_dependencies'}
                )
            
            # Process the event
            result = processor.process_event(event)
            return result
            
        except Exception as e:
            logger.error(f"Processor {processor.name} failed: {e}")
            return EnrichmentResult(
                success=False,
                enriched_data={},
                processing_time=0.0,
                processor_name=processor.name,
                error_message=str(e),
                status=EnrichmentStatus.FAILED
            )
    
    def _process_retries(self, retry_queue: deque, results: Dict[str, EnrichmentResult]) -> None:
        """
        Process retry queue for failed enrichments.
        
        Args:
            retry_queue: Queue of (processor, event, retry_count) tuples
            results: Results dictionary to update
        """
        while retry_queue:
            processor, event, retry_count = retry_queue.popleft()
            
            if retry_count >= self.max_retries:
                logger.warning(f"Max retries ({self.max_retries}) reached for {processor.name}")
                continue
            
            # Wait before retry
            if retry_count > 0:
                time.sleep(self.retry_delay)
            
            # Retry processing
            logger.debug(f"Retrying {processor.name} (attempt {retry_count + 1}/{self.max_retries})")
            result = self._process_with_processor(processor, event)
            
            # Update result
            results[processor.name] = result
            self.retry_counts[processor.name] += 1
            
            # Queue for another retry if needed
            if result.requires_retry and retry_count + 1 < self.max_retries:
                retry_queue.append((processor, event, retry_count + 1))
    
    def _apply_enrichment(self, event: Any, result: EnrichmentResult) -> Any:
        """
        Apply enrichment data to event.
        
        Args:
            event: Original event
            result: Enrichment result to apply
            
        Returns:
            Event with enrichment applied
        """
        try:
            # If event has enriched_data attribute, update it
            if hasattr(event, 'data') and isinstance(event.data, dict):
                # Add enrichment data to event data
                enrichment_key = f"enrichment_{result.processor_name}"
                event.data[enrichment_key] = result.enriched_data
                
                # Track enrichment processors
                if hasattr(event, 'enrichments'):
                    if isinstance(event.enrichments, list):
                        event.enrichments.append(result.processor_name)
                else:
                    # Add enrichments tracking
                    event.enrichments = [result.processor_name]
            
            return event
            
        except Exception as e:
            logger.error(f"Failed to apply enrichment from {result.processor_name}: {e}")
            return event
    
    def _dependencies_satisfied(self, processor: BaseEnrichment, event: Any) -> bool:
        """
        Check if processor dependencies are satisfied.
        
        Args:
            processor: Processor to check
            event: Event being processed
            
        Returns:
            True if dependencies are satisfied, False otherwise
        """
        dependencies = processor.get_dependencies()
        
        if not dependencies:
            return True
        
        # Check if event has enrichments tracking
        if not hasattr(event, 'enrichments'):
            return len(dependencies) == 0
        
        event_enrichments = getattr(event, 'enrichments', [])
        
        # All dependencies must be in the enrichments list
        for dep in dependencies:
            if dep not in event_enrichments:
                logger.debug(f"Dependency {dep} not satisfied for {processor.name}")
                return False
        
        return True
    
    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """
        Build dependency graph for processors.
        
        Returns:
            Dictionary mapping processor name to set of dependencies
        """
        graph = {}
        
        for processor in self.processors:
            dependencies = processor.get_dependencies()
            graph[processor.name] = set(dependencies)
        
        logger.debug(f"Built dependency graph: {graph}")
        return graph
    
    def _compute_processing_order(self) -> List[BaseEnrichment]:
        """
        Compute processing order using topological sort with priority.
        
        Returns:
            List of processors in processing order
        """
        # Start with priority-sorted list
        ordered = []
        remaining = set(p.name for p in self.processors)
        processor_map = {p.name: p for p in self.processors}
        
        # Topological sort with priority
        while remaining:
            # Find processors with no unresolved dependencies
            ready = []
            for name in remaining:
                dependencies = self.dependency_graph.get(name, set())
                if dependencies.issubset(set(p.name for p in ordered)):
                    ready.append(name)
            
            if not ready:
                # Circular dependency or missing dependency
                logger.warning(f"Circular or missing dependencies detected. Remaining: {remaining}")
                # Add remaining processors in priority order
                for processor in self.processors:
                    if processor.name in remaining:
                        ordered.append(processor)
                        remaining.remove(processor.name)
                break
            
            # Sort ready processors by priority
            ready.sort(key=lambda name: processor_map[name].priority)
            
            # Add first ready processor
            next_name = ready[0]
            ordered.append(processor_map[next_name])
            remaining.remove(next_name)
        
        return ordered
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get orchestrator performance metrics.
        
        Returns:
            Dictionary with metrics
        """
        avg_enrichment_time = (self.total_enrichment_time / self.events_processed 
                              if self.events_processed > 0 else 0.0)
        
        processor_metrics = {
            p.name: p.get_metrics() for p in self.processors
        }
        
        return {
            'events_processed': self.events_processed,
            'total_enrichment_time': self.total_enrichment_time,
            'avg_enrichment_time': avg_enrichment_time,
            'enrichment_failures': self.enrichment_failures,
            'retry_counts': dict(self.retry_counts),
            'processor_count': len(self.processors),
            'enabled_processor_count': sum(1 for p in self.processors if p.enabled),
            'processor_metrics': processor_metrics
        }
    
    def get_processor(self, name: str) -> Optional[BaseEnrichment]:
        """
        Get processor by name.
        
        Args:
            name: Processor name
            
        Returns:
            Processor instance or None if not found
        """
        for processor in self.processors:
            if processor.name == name:
                return processor
        return None
    
    def enable_processor(self, name: str) -> bool:
        """
        Enable a specific processor.
        
        Args:
            name: Processor name
            
        Returns:
            True if processor was found and enabled
        """
        processor = self.get_processor(name)
        if processor:
            processor.enabled = True
            logger.info(f"Enabled processor: {name}")
            return True
        return False
    
    def disable_processor(self, name: str) -> bool:
        """
        Disable a specific processor.
        
        Args:
            name: Processor name
            
        Returns:
            True if processor was found and disabled
        """
        processor = self.get_processor(name)
        if processor:
            processor.enabled = False
            logger.info(f"Disabled processor: {name}")
            return True
        return False
