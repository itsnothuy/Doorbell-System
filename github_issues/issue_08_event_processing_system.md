# Issue #8: Event Processing System Implementation

## 📋 **Overview**

Implement the comprehensive event processing system that manages the complete lifecycle of security events from detection through enrichment, persistence, and notification delivery. This system serves as the central coordinator for all event-driven activities in the Frigate-inspired architecture.

## 🎯 **Objectives**

### **Primary Goals**
1. **Event Lifecycle Management**: Complete event state machine from creation to resolution
2. **Enrichment Coordination**: Orchestrate multiple enrichment processors
3. **Database Persistence**: Reliable event storage with proper indexing
4. **Notification Routing**: Intelligent routing to internal notification systems
5. **Performance Monitoring**: Track event processing metrics and performance

### **Success Criteria**
- Event processing latency <100ms for 95% of events
- Zero event loss with proper persistence guarantees
- Scalable enrichment pipeline supporting multiple processors
- Real-time event streaming to web interface
- Comprehensive audit trail for all events

## 🏗️ **Architecture Requirements**

### **Pipeline Position**
```
Face Recognition Worker → Event Processing System → Internal Notifications
                                ↓
                        Database Persistence + Web Streaming
```

### **Processing Flow**
1. **Event Ingestion**: Receive events from recognition and other sources
2. **Event Validation**: Validate event structure and required fields
3. **Enrichment Pipeline**: Apply all configured enrichment processors
4. **State Management**: Track event state transitions
5. **Persistence**: Store events with proper indexing
6. **Notification Routing**: Route to appropriate notification channels
7. **Real-time Streaming**: Broadcast events to web interface

## 📁 **Implementation Specifications**

### **File Structure**
```
src/pipeline/event_processor.py         # Main event processing worker
src/enrichment/                         # Enrichment processors
    __init__.py
    base_enrichment.py                   # Abstract enrichment interface
    alert_manager.py                     # Alert processing and routing
    notification_handler.py              # Internal notification system
    web_events.py                        # Web interface event streaming
    metadata_enrichment.py               # Event metadata enhancement
config/event_config.py                  # Event processing configuration
tests/test_event_processor.py           # Comprehensive test suite
tests/test_enrichment_processors.py     # Enrichment processor tests
```

### **Core Component: `EventProcessor`**
```python
class EventProcessor(PipelineWorker):
    """Central event processing system for security events."""
    
    def __init__(self, message_bus: MessageBus, config: EventConfig):
        super().__init__(message_bus, config.base_config)
        self.event_config = config
        
        # Event storage
        self.event_database = EventDatabase(config.database_config)
        
        # Enrichment processors
        self.enrichment_processors = self._initialize_enrichment_processors()
        
        # Event state tracking
        self.active_events = {}  # event_id -> EventState
        self.event_queue = PriorityQueue()
        
        # Performance metrics
        self.events_processed = 0
        self.enrichment_success_rate = 0.0
        self.avg_processing_time = 0.0
        
        # Real-time streaming
        self.web_event_streamer = WebEventStreamer(config.web_config)
        
    def process_event(self, event: SecurityEvent) -> ProcessedEvent:
        """Process security event through complete pipeline."""
        
    def enrich_event(self, event: SecurityEvent) -> SecurityEvent:
        """Apply all enrichment processors to event."""
        
    def persist_event(self, event: ProcessedEvent) -> bool:
        """Persist event to database with error handling."""
        
    def route_notifications(self, event: ProcessedEvent) -> List[NotificationResult]:
        """Route event to appropriate notification channels."""
        
    def stream_to_web(self, event: ProcessedEvent) -> None:
        """Stream event to web interface in real-time."""
```

### **Event Data Structures**
```python
@dataclass
class SecurityEvent:
    """Base security event structure."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    source: str
    confidence: float
    
    # Core event data
    data: Dict[str, Any]
    
    # Processing metadata
    processing_stage: str
    enrichments: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Event relationships
    parent_event_id: Optional[str] = None
    child_event_ids: List[str] = field(default_factory=list)

@dataclass
class FaceRecognitionEvent(SecurityEvent):
    """Face recognition specific event."""
    # Face data
    face_encoding: Optional[np.ndarray] = None
    face_image: Optional[np.ndarray] = None
    face_location: Optional[Tuple[int, int, int, int]] = None
    
    # Recognition results
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    recognition_confidence: Optional[float] = None
    is_known_person: bool = False
    is_blacklisted: bool = False
    
    # Context data
    motion_data: Optional[Any] = None
    environmental_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessedEvent:
    """Fully processed event with all enrichments."""
    original_event: SecurityEvent
    enriched_data: Dict[str, Any]
    processing_metadata: ProcessingMetadata
    notification_targets: List[NotificationTarget]
    persistence_status: PersistenceStatus
    
@dataclass
class ProcessingMetadata:
    """Event processing metadata."""
    start_time: datetime
    end_time: datetime
    processing_duration: float
    enrichment_results: Dict[str, EnrichmentResult]
    error_count: int
    retry_count: int
    priority_level: int

@dataclass
class EventState:
    """Event state tracking."""
    event_id: str
    current_stage: ProcessingStage
    start_time: datetime
    last_update: datetime
    retry_count: int
    errors: List[ProcessingError]
    enrichment_status: Dict[str, EnrichmentStatus]
```

### **Enrichment Framework**
```python
class BaseEnrichment(ABC):
    """Abstract base class for event enrichment processors."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.priority = config.get('priority', 5)
        self.enabled = config.get('enabled', True)
        
    @abstractmethod
    def can_process(self, event: SecurityEvent) -> bool:
        """Check if this enrichment can process the given event."""
        
    @abstractmethod
    def enrich(self, event: SecurityEvent) -> EnrichmentResult:
        """Enrich the event with additional data."""
        
    def get_dependencies(self) -> List[str]:
        """Get list of enrichment dependencies."""
        return []
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get enrichment performance metrics."""
        return {}

@dataclass
class EnrichmentResult:
    """Result of enrichment processing."""
    success: bool
    enriched_data: Dict[str, Any]
    processing_time: float
    error_message: Optional[str] = None
    requires_retry: bool = False
    
class EnrichmentOrchestrator:
    """Orchestrate multiple enrichment processors."""
    
    def __init__(self, processors: List[BaseEnrichment]):
        self.processors = sorted(processors, key=lambda p: p.priority)
        self.dependency_graph = self._build_dependency_graph()
        
    def process_event(self, event: SecurityEvent) -> Dict[str, EnrichmentResult]:
        """Process event through all applicable enrichment processors."""
        results = {}
        enriched_event = event
        
        # Process in dependency order
        for processor in self._get_processing_order():
            if processor.enabled and processor.can_process(enriched_event):
                try:
                    result = processor.enrich(enriched_event)
                    results[processor.name] = result
                    
                    if result.success:
                        # Apply enrichment to event
                        enriched_event = self._apply_enrichment(enriched_event, result)
                    
                except Exception as e:
                    results[processor.name] = EnrichmentResult(
                        success=False,
                        enriched_data={},
                        processing_time=0.0,
                        error_message=str(e)
                    )
        
        return results
```

## 🔧 **Implementation Details**

### **1. Event Processing Pipeline**
```python
def _process_event_pipeline(self, event: SecurityEvent) -> ProcessedEvent:
    """Complete event processing pipeline."""
    start_time = time.time()
    processing_metadata = ProcessingMetadata(
        start_time=datetime.now(),
        end_time=None,
        processing_duration=0.0,
        enrichment_results={},
        error_count=0,
        retry_count=0,
        priority_level=self._calculate_event_priority(event)
    )
    
    try:
        # 1. Validate event
        self._validate_event(event)
        
        # 2. Apply enrichment processors
        enrichment_results = self.enrichment_orchestrator.process_event(event)
        processing_metadata.enrichment_results = enrichment_results
        
        # 3. Calculate notification targets
        notification_targets = self._determine_notification_targets(event, enrichment_results)
        
        # 4. Prepare final processed event
        enriched_data = self._merge_enrichment_data(enrichment_results)
        
        # 5. Update processing metadata
        processing_metadata.end_time = datetime.now()
        processing_metadata.processing_duration = time.time() - start_time
        
        processed_event = ProcessedEvent(
            original_event=event,
            enriched_data=enriched_data,
            processing_metadata=processing_metadata,
            notification_targets=notification_targets,
            persistence_status=PersistenceStatus.PENDING
        )
        
        return processed_event
        
    except Exception as e:
        processing_metadata.error_count += 1
        processing_metadata.end_time = datetime.now()
        processing_metadata.processing_duration = time.time() - start_time
        
        logger.error(f"Event processing failed for {event.event_id}: {e}")
        
        # Create error processed event
        return ProcessedEvent(
            original_event=event,
            enriched_data={'error': str(e)},
            processing_metadata=processing_metadata,
            notification_targets=[],
            persistence_status=PersistenceStatus.FAILED
        )

def _handle_face_recognition_event(self, message: Message) -> None:
    """Handle face recognition events specifically."""
    try:
        face_event = FaceRecognitionEvent.from_message(message)
        
        # Add face-specific validation
        if not self._validate_face_event(face_event):
            logger.warning(f"Invalid face event: {face_event.event_id}")
            return
        
        # Process through pipeline
        processed_event = self._process_event_pipeline(face_event)
        
        # Handle face-specific persistence
        self._persist_face_event(processed_event)
        
        # Route notifications
        self._route_face_notifications(processed_event)
        
        # Stream to web interface
        self._stream_face_event(processed_event)
        
        # Update metrics
        self.events_processed += 1
        self._update_processing_metrics(processed_event)
        
    except Exception as e:
        logger.error(f"Face recognition event handling failed: {e}")
```

### **2. Database Persistence**
```python
class EventDatabase:
    """Database interface for event persistence."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection_pool = self._create_connection_pool()
        
    def store_event(self, event: ProcessedEvent) -> bool:
        """Store processed event with full metadata."""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # Insert main event record
                event_record = {
                    'event_id': event.original_event.event_id,
                    'event_type': event.original_event.event_type.value,
                    'timestamp': event.original_event.timestamp,
                    'source': event.original_event.source,
                    'confidence': event.original_event.confidence,
                    'data': json.dumps(event.original_event.data),
                    'enriched_data': json.dumps(event.enriched_data),
                    'processing_metadata': json.dumps(asdict(event.processing_metadata)),
                    'created_at': datetime.now()
                }
                
                cursor.execute('''
                    INSERT INTO security_events 
                    (event_id, event_type, timestamp, source, confidence, data, 
                     enriched_data, processing_metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', tuple(event_record.values()))
                
                # Store face-specific data if applicable
                if isinstance(event.original_event, FaceRecognitionEvent):
                    self._store_face_event_data(cursor, event)
                
                # Store enrichment results
                self._store_enrichment_results(cursor, event)
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Event storage failed: {e}")
            return False
    
    def query_events(self, filters: EventQueryFilters) -> List[SecurityEvent]:
        """Query events with complex filtering."""
        # Implementation for event querying
        pass
    
    def get_event_statistics(self, time_range: TimeRange) -> EventStatistics:
        """Get event statistics for time range."""
        # Implementation for statistics
        pass
```

### **3. Real-time Web Streaming**
```python
class WebEventStreamer:
    """Stream events to web interface in real-time."""
    
    def __init__(self, config: WebConfig):
        self.config = config
        self.active_connections = set()
        self.event_queue = Queue()
        
    def stream_event(self, event: ProcessedEvent) -> None:
        """Stream event to all active web connections."""
        try:
            # Prepare web-safe event data
            web_event = self._prepare_web_event(event)
            
            # Add to streaming queue
            self.event_queue.put(web_event)
            
            # Notify all active connections
            self._broadcast_to_connections(web_event)
            
        except Exception as e:
            logger.error(f"Web event streaming failed: {e}")
    
    def _prepare_web_event(self, event: ProcessedEvent) -> Dict[str, Any]:
        """Prepare event data for web consumption."""
        return {
            'event_id': event.original_event.event_id,
            'event_type': event.original_event.event_type.value,
            'timestamp': event.original_event.timestamp.isoformat(),
            'confidence': event.original_event.confidence,
            'enriched_data': self._sanitize_for_web(event.enriched_data),
            'processing_time': event.processing_metadata.processing_duration,
            'notification_count': len(event.notification_targets)
        }
```

### **4. Notification Routing**
```python
class NotificationRouter:
    """Route events to appropriate notification channels."""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        self.notification_handlers = self._initialize_handlers()
        
    def route_event(self, event: ProcessedEvent) -> List[NotificationResult]:
        """Route event to all applicable notification channels."""
        results = []
        
        for target in event.notification_targets:
            try:
                handler = self.notification_handlers.get(target.channel)
                if handler and handler.is_enabled():
                    result = handler.send_notification(event, target)
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Notification routing failed for {target.channel}: {e}")
                results.append(NotificationResult(
                    success=False,
                    channel=target.channel,
                    error_message=str(e)
                ))
        
        return results
    
    def _determine_notification_targets(self, event: SecurityEvent, 
                                      enrichment_results: Dict[str, EnrichmentResult]) -> List[NotificationTarget]:
        """Determine which notification channels should receive this event."""
        targets = []
        
        # Priority-based routing
        if event.event_type == EventType.UNKNOWN_PERSON_DETECTED:
            targets.append(NotificationTarget(
                channel='internal_alert',
                priority=NotificationPriority.HIGH,
                template='unknown_person_alert'
            ))
        
        if event.event_type == EventType.BLACKLISTED_PERSON_DETECTED:
            targets.append(NotificationTarget(
                channel='internal_alert',
                priority=NotificationPriority.CRITICAL,
                template='blacklisted_person_alert'
            ))
            targets.append(NotificationTarget(
                channel='web_notification',
                priority=NotificationPriority.CRITICAL,
                template='security_breach_alert'
            ))
        
        # Add web streaming for all events
        targets.append(NotificationTarget(
            channel='web_stream',
            priority=NotificationPriority.LOW,
            template='event_stream'
        ))
        
        return targets
```

## 🧪 **Testing Requirements**

### **Unit Tests**
```python
class TestEventProcessor:
    """Comprehensive test suite for event processor."""
    
    def test_event_processing_pipeline(self):
        """Test complete event processing flow."""
        
    def test_enrichment_orchestration(self):
        """Test enrichment processor coordination."""
        
    def test_database_persistence(self):
        """Test event storage and retrieval."""
        
    def test_notification_routing(self):
        """Test notification channel routing."""
        
    def test_web_streaming(self):
        """Test real-time web event streaming."""
        
    def test_error_handling(self):
        """Test error handling and recovery."""
        
    def test_performance_metrics(self):
        """Test performance tracking and metrics."""
        
    def test_concurrent_processing(self):
        """Test concurrent event processing."""

class TestEnrichmentProcessors:
    """Test suite for enrichment processors."""
    
    def test_alert_manager(self):
        """Test alert processing and routing."""
        
    def test_notification_handler(self):
        """Test internal notification system."""
        
    def test_web_events(self):
        """Test web interface event streaming."""
        
    def test_metadata_enrichment(self):
        """Test event metadata enhancement."""
```

### **Integration Tests**
```python
def test_end_to_end_event_processing():
    """Test complete event flow from recognition to notification."""
    
def test_event_processor_performance():
    """Test event processing performance under load."""
    
def test_database_integration():
    """Test database operations and data integrity."""
    
def test_notification_integration():
    """Test integration with notification systems."""
```

## 📊 **Performance Targets**

### **Processing Performance**
- **Event Latency**: <100ms for 95% of events
- **Throughput**: >50 events/second sustained
- **Memory Usage**: <200MB for 10,000 active events
- **Database Response**: <50ms for event storage

### **Scalability**
- **Concurrent Events**: Handle 100+ simultaneous events
- **Enrichment Processors**: Support 10+ enrichment stages
- **Notification Channels**: Route to multiple channels simultaneously
- **Web Streaming**: Support 50+ concurrent web connections

### **Reliability**
- **Event Loss**: Zero tolerance for event loss
- **Database Durability**: ACID compliance for critical events
- **Error Recovery**: Automatic retry with exponential backoff
- **Health Monitoring**: Real-time system health tracking

## 🔧 **Configuration Example**

### **event_config.py**
```python
# Event Processing Configuration
EVENT_CONFIG = {
    "base_config": {
        "worker_count": 3,
        "queue_size": 1000,
        "timeout": 30.0
    },
    
    # Database configuration
    "database_config": {
        "type": "sqlite",
        "path": "data/events.db",
        "connection_pool_size": 10,
        "batch_insert_size": 100
    },
    
    # Enrichment configuration
    "enrichment_config": {
        "enabled_processors": [
            "alert_manager",
            "notification_handler", 
            "web_events",
            "metadata_enrichment"
        ],
        "max_enrichment_time": 5.0,
        "retry_failed_enrichments": True
    },
    
    # Notification configuration
    "notification_config": {
        "internal_alerts": {
            "enabled": True,
            "priority_threshold": "MEDIUM"
        },
        "web_notifications": {
            "enabled": True,
            "real_time_streaming": True,
            "max_connections": 50
        }
    },
    
    # Performance configuration
    "performance_config": {
        "max_concurrent_events": 100,
        "event_timeout": 60.0,
        "metrics_collection": True,
        "health_check_interval": 30.0
    }
}
```

## 📈 **Monitoring and Metrics**

### **Key Metrics to Track**
- Event processing latency and throughput
- Enrichment processor success rates
- Database performance and storage usage
- Notification delivery success rates
- Web streaming connection health

### **Health Checks**
- Event processing pipeline status
- Database connectivity and performance
- Enrichment processor availability
- Notification channel health
- Memory usage and resource consumption

## 🎯 **Definition of Done**

### **Functional Requirements**
- [ ] Complete event processing pipeline from ingestion to delivery
- [ ] Enrichment orchestration with dependency management
- [ ] Database persistence with proper indexing and querying
- [ ] Real-time web streaming of events
- [ ] Internal notification routing and delivery
- [ ] Comprehensive error handling and recovery

### **Non-Functional Requirements**
- [ ] Event processing latency meets performance targets (<100ms)
- [ ] Zero event loss with proper persistence guarantees
- [ ] Scalable to handle high event volumes
- [ ] Memory usage bounded and efficient
- [ ] Comprehensive monitoring and metrics collection

### **Documentation Requirements**
- [ ] Code documentation with clear docstrings
- [ ] Event processing flow documentation
- [ ] Enrichment processor development guide
- [ ] Configuration reference and examples
- [ ] Performance tuning recommendations

---

## 🔗 **Dependencies**

### **Previous Issues**
- **Issue #4**: Frame Capture Worker (event source)
- **Issue #5**: Motion Detection Worker (event source)
- **Issue #6**: Face Detection Worker Pool (event source)
- **Issue #7**: Face Recognition Engine (primary event source)

### **Next Issues**
- **Issue #9**: Hardware Abstraction Layer (platform integration)
- **Issue #10**: Storage Layer (database implementation)
- **Issue #11**: Internal Notification System (notification delivery)

### **External Dependencies**
- SQLite/Database system for persistence
- Message bus infrastructure
- Configuration management system
- Web framework for real-time streaming

---

## 🤖 **For Coding Agents: Auto-Close Setup**

### **Branch Naming Convention**
When implementing this issue, create your branch using one of these patterns:
- `issue-8/event-processing-system`
- `8-event-processing-system` 
- `issue-8/implement-event-processing`

### **PR Creation**
The GitHub Action will automatically append `Closes #8` to your PR description when you follow the branch naming convention above. This ensures the issue closes automatically when your PR is merged to the default branch.

### **Manual Alternative**
If you prefer manual control, include one of these in your PR description:
```
Closes #8
Fixes #8
Resolves #8
```

---

**This issue implements the central event processing system that coordinates all event-driven activities in the Frigate-inspired security architecture, providing reliable event handling, enrichment, persistence, and delivery.**