# ADR-006: Communication Infrastructure and Message Bus

**Date:** 2025-10-28  
**Status:** Accepted  
**Related:** ADR-005 (Pipeline Architecture)

## Context

The pipeline architecture requires high-performance inter-process communication between workers to achieve real-time performance. The system needs to handle:

1. **High-frequency events**: Camera frames, motion detection, face detection results
2. **Low-latency communication**: Sub-millisecond message passing for real-time processing
3. **Reliable message delivery**: Ensure critical events (doorbell triggers, face recognition results) are not lost
4. **Event correlation**: Track related events across the entire pipeline
5. **Backpressure handling**: Prevent system overload when processing falls behind
6. **Observability**: Monitor message flow for debugging and performance optimization

Traditional message queue solutions (Redis, RabbitMQ) introduce external dependencies and network overhead that are unsuitable for edge device deployment and real-time processing requirements.

## Decision

We will implement a **ZeroMQ-inspired in-process message bus** optimized for high-performance local communication:

### Core Message Bus Design

1. **In-Process Message Bus (`MessageBus`)**
   - Thread-safe message passing using Python queues
   - Topic-based publish-subscribe pattern
   - Connection pooling and lifecycle management
   - Built-in serialization and deserialization
   - No external dependencies or network overhead

2. **Event System (`PipelineEvent`)**
   - Structured event format with metadata
   - Event correlation and tracing support
   - Immutable event objects for thread safety
   - Event enrichment and transformation pipeline
   - Comprehensive event logging and debugging

3. **Queue Management (`QueueManager`)**
   - Priority queue support for critical events
   - Backpressure detection and handling
   - Queue size monitoring and alerting
   - Automatic queue cleanup and maintenance
   - Dead letter queue for failed events

4. **Message Routing and Patterns**
   - Topic-based routing with wildcard support
   - Request-response pattern for synchronous operations
   - Fan-out pattern for parallel processing
   - Event streaming for real-time monitoring
   - Circuit breaker pattern for fault tolerance

### Communication Patterns

#### Publisher-Subscriber Pattern
```python
# Publisher (Frame Capture Worker)
message_bus.publish('camera.frames', frame_event)

# Subscriber (Face Detection Worker)  
message_bus.subscribe('camera.frames', self.handle_frame)
```

#### Request-Response Pattern
```python
# Request face recognition
result = message_bus.request('face.recognize', recognition_request, timeout=5.0)
```

#### Event Streaming Pattern
```python
# Stream real-time events to web interface
for event in message_bus.stream('*.notifications'):
    web_interface.send_event(event)
```

### Performance Optimizations

1. **Lock-Free Queues**: Use queue.Queue for thread-safe operations
2. **Message Pooling**: Reuse message objects to reduce GC pressure
3. **Batching**: Group related messages for efficiency
4. **Compression**: Optional compression for large payloads
5. **Zero-Copy**: Minimize data copying in hot paths

## Alternatives Considered

### 1. Redis Pub/Sub
**Rejected** because:
- External dependency management complexity
- Network serialization overhead unacceptable for real-time processing
- Redis server resource usage on edge devices
- Single point of failure for local processing

### 2. RabbitMQ/AMQP
**Rejected** because:
- Heavy resource usage inappropriate for Raspberry Pi
- Complex setup and maintenance
- Network overhead for local communication
- Over-engineered for single-node deployment

### 3. Apache Kafka
**Rejected** because:
- Massive resource requirements
- Designed for distributed systems, not single-node
- Complex operational overhead
- Overkill for the problem domain

### 4. ZeroMQ (actual)
**Considered but rejected** because:
- Additional native dependency compilation complexity
- Learning curve for contributors
- Binding and configuration complexity
- Python implementation provides sufficient performance

### 5. Python multiprocessing.Queue
**Rejected** because:
- Limited to parent-child process communication
- No pub-sub or routing capabilities
- Poor performance for high-frequency messages
- Limited observability and monitoring

### 6. Direct method calls (synchronous)
**Rejected** because:
- Tight coupling between components
- No fault isolation
- Blocking operations harm performance
- Difficult to test and debug

## Consequences

### Positive Consequences

1. **Performance Benefits**
   - Zero external dependencies for communication
   - Sub-millisecond message passing within process
   - No network serialization overhead
   - Lock-free queue operations for hot paths

2. **Reliability and Fault Tolerance**
   - No external service dependencies to fail
   - Built-in backpressure handling
   - Dead letter queues for failed messages
   - Circuit breaker patterns for fault isolation

3. **Observability and Debugging**
   - Built-in message tracing and correlation
   - Real-time queue monitoring and metrics
   - Event flow visualization capabilities
   - Comprehensive logging and debugging support

4. **Simplicity and Maintainability**
   - Pure Python implementation
   - No additional infrastructure to manage
   - Easy to understand and debug
   - Minimal operational overhead

5. **Development Productivity**
   - Easy to test with mock message bus
   - Clear separation of concerns
   - Type-safe event definitions
   - Rich debugging and development tools

### Negative Consequences

1. **Custom Implementation Complexity**
   - Need to implement and maintain custom message bus
   - Potential bugs in custom communication layer
   - Performance tuning and optimization required
   - Testing complexity for edge cases

2. **Scalability Limitations**
   - Limited to single-node deployment
   - No built-in persistence or replay capabilities
   - Memory usage scales with message volume
   - No automatic load balancing across nodes

3. **Feature Limitations**
   - No guaranteed delivery semantics
   - Limited routing complexity compared to enterprise solutions
   - No built-in message transformation pipelines
   - Manual implementation of advanced patterns

4. **Learning Curve**
   - Contributors need to understand custom message bus
   - Different patterns from standard message queues
   - Custom debugging and monitoring tools
   - Documentation and examples needed

### Risk Mitigation Strategies

1. **Comprehensive Testing**
   - Unit tests for all message bus components
   - Integration tests for communication patterns
   - Load testing under high message volume
   - Fault injection testing for edge cases

2. **Performance Monitoring**
   - Real-time queue depth monitoring
   - Message throughput and latency metrics
   - Memory usage tracking
   - Performance regression testing

3. **Graceful Degradation**
   - Fallback mechanisms for communication failures
   - Queue overflow protection
   - Automatic cleanup of stale connections
   - Circuit breaker implementation

4. **Documentation and Examples**
   - Comprehensive API documentation
   - Usage patterns and best practices
   - Troubleshooting guides
   - Performance tuning guidelines

## Implementation Details

### Message Bus Core Components

```python
class MessageBus:
    """High-performance in-process message bus."""
    
    def publish(self, topic: str, event: PipelineEvent) -> None:
        """Publish event to topic."""
    
    def subscribe(self, topic: str, handler: Callable) -> str:
        """Subscribe to topic with handler."""
    
    def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from topic."""
    
    def request(self, topic: str, request: Any, timeout: float) -> Any:
        """Send request and wait for response."""
    
    def stream(self, topic_pattern: str) -> Iterator[PipelineEvent]:
        """Stream events matching topic pattern."""
```

### Event System Design

```python
@dataclass
class PipelineEvent:
    """Immutable pipeline event."""
    event_id: str
    event_type: EventType
    timestamp: float
    data: Dict[str, Any]
    source: str
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
```

### Queue Management

```python
class QueueManager:
    """Manages message queues with monitoring and backpressure."""
    
    def create_queue(self, name: str, max_size: int, priority: bool = False) -> Queue
    def get_queue_metrics(self, name: str) -> QueueMetrics
    def handle_backpressure(self, queue_name: str) -> None
    def cleanup_stale_queues(self) -> None
```

## Performance Characteristics

### Target Performance Metrics
- **Message Latency**: < 1ms for local delivery
- **Throughput**: > 10,000 messages/second
- **Memory Usage**: < 50MB for message bus overhead
- **Queue Depth**: Support for 1000+ queued messages per topic

### Benchmarking Strategy
- Continuous performance regression testing
- Load testing with realistic message patterns
- Memory usage profiling and optimization
- Latency distribution analysis

## References

- **ZeroMQ Design Patterns**: Inspiration for communication patterns
- **Frigate NVR**: Message passing architecture reference
- **ADR-005**: Pipeline architecture requiring communication infrastructure
- **Issue #1**: Communication infrastructure implementation
- **Issue #2**: Event system design and implementation

## Success Metrics

- **Latency**: < 1ms average message delivery time
- **Throughput**: Handle 10,000+ messages/second sustained
- **Reliability**: 99.99% message delivery success rate
- **Resource Usage**: < 5% CPU overhead for message bus operations
- **Development Velocity**: 50% faster feature development with clean communication patterns