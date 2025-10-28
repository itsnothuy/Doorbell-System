# ADR-005: Pipeline Architecture and Orchestration

**Date:** 2025-10-28  
**Status:** Accepted  
**Supersedes:** Portions of ADR-001 (legacy system architecture)

## Context

The original system used a monolithic `DoorbellSecuritySystem` class that directly managed all components (camera, GPIO, face recognition, notifications) in a tightly coupled manner. As the system has grown in complexity and we've identified performance bottlenecks, we need to transform the architecture to support:

1. **Scalability**: Parallel processing of face detection and recognition
2. **Reliability**: Fault isolation and recovery mechanisms  
3. **Performance**: Optimized pipeline processing inspired by Frigate NVR
4. **Maintainability**: Clear separation of concerns and testability
5. **Extensibility**: Easy addition of new processing stages and detectors

The system needs to handle real-time video processing, concurrent doorbell events, and maintain sub-second response times while supporting multiple face detection backends and notification channels.

## Decision

We will implement a **Pipeline Architecture with Orchestration** inspired by Frigate NVR's design, transforming from the legacy monolithic system to a modular pipeline-based approach:

### Core Architecture Components

1. **Pipeline Orchestrator (`PipelineOrchestrator`)**
   - Central coordinator managing the entire pipeline lifecycle
   - Worker pool management with health monitoring
   - Event-driven workflow coordination
   - Performance monitoring and metrics collection
   - Graceful startup, shutdown, and error recovery

2. **Pipeline Workers**
   - `FrameCaptureWorker`: Handles camera input and GPIO events
   - `MotionDetectorWorker`: Optional motion detection preprocessing  
   - `FaceDetectorWorker`: Face detection using pluggable backends
   - `FaceRecognizerWorker`: Face recognition and matching
   - `EventProcessorWorker`: Event enrichment and notification routing

3. **Message Bus Communication**
   - ZeroMQ-inspired high-performance message passing
   - Event-driven communication between workers
   - Queue management with backpressure handling
   - Message serialization and routing

4. **Legacy Compatibility Layer**
   - `LegacyAdapter` providing backward compatibility
   - Gradual migration path from legacy to pipeline
   - Existing API contract preservation

### Pipeline Flow Design

```
GPIO Trigger → Frame Capture → Motion Detection → Face Detection → Face Recognition → Event Processing → Notifications
      ↓              ↓               ↓               ↓                ↓                  ↓              ↓
   Event Bus    Ring Buffer     Motion Events    Face Events    Recognition Events  Enriched Events  Delivery
```

### Orchestrator Management Pattern

- **Worker Lifecycle**: Start, monitor, restart failed workers
- **Health Monitoring**: Continuous health checks and performance metrics
- **Resource Management**: Dynamic worker scaling based on load
- **Event Coordination**: Cross-worker event correlation and processing
- **Error Handling**: Graceful degradation and recovery mechanisms

## Alternatives Considered

### 1. Microservices Architecture
**Rejected** because:
- Increased deployment complexity for edge devices
- Network latency between services unacceptable for real-time processing
- Resource overhead too high for Raspberry Pi deployments
- Operational complexity outweighs benefits for this use case

### 2. Actor Model (e.g., using Ray or Celery)
**Rejected** because:
- Additional dependencies and complexity
- Overkill for the problem domain
- Memory overhead concerns on edge devices
- Learning curve for contributors

### 3. Pure Multithreading (Enhanced Legacy)
**Rejected** because:
- Python GIL limitations for CPU-intensive face recognition
- Difficult to achieve proper fault isolation
- Hard to implement dynamic scaling
- Limited testability and maintainability

### 4. Message Queue Systems (Redis, RabbitMQ)
**Rejected** because:
- External dependency management complexity
- Resource usage overhead on edge devices
- Network dependency for local processing
- Over-engineering for the use case

## Consequences

### Positive Consequences

1. **Performance Improvements**
   - Parallel face detection and recognition processing
   - Reduced latency through pipeline parallelism
   - Better resource utilization across CPU cores
   - Optimized memory usage with ring buffers

2. **Scalability and Flexibility**
   - Easy addition of new processing stages
   - Dynamic worker scaling based on load
   - Support for multiple face detection backends
   - Horizontal scaling preparation

3. **Reliability and Fault Tolerance**
   - Worker fault isolation prevents system-wide failures
   - Automatic worker restart and recovery
   - Graceful degradation under high load
   - Health monitoring and alerting

4. **Maintainability and Testing**
   - Clear separation of concerns between workers
   - Individual worker unit testing
   - Integration testing of pipeline flows
   - Easier debugging and profiling

5. **Backward Compatibility**
   - Existing web interface and APIs continue to work
   - Gradual migration path for existing deployments
   - Legacy configuration compatibility
   - No breaking changes for end users

### Negative Consequences

1. **Implementation Complexity**
   - More complex system startup and coordination
   - Inter-process communication overhead
   - Message serialization costs
   - Additional debugging complexity

2. **Resource Overhead**
   - Multiple worker processes consume more memory
   - Message passing introduces CPU overhead
   - Orchestrator adds management overhead
   - Monitoring systems require additional resources

3. **Migration Challenges**
   - Existing deployments need migration strategy
   - Configuration format changes required
   - Data migration for face databases
   - Potential temporary instability during migration

4. **Learning Curve**
   - Contributors need to understand pipeline architecture
   - More complex development and debugging workflows
   - Additional testing requirements
   - New deployment considerations

### Risk Mitigation Strategies

1. **Gradual Migration**
   - Implement legacy adapter for backward compatibility
   - Phased rollout with fallback mechanisms
   - Comprehensive migration testing
   - Clear rollback procedures

2. **Performance Monitoring**
   - Real-time performance metrics collection
   - Automated performance regression testing
   - Resource usage monitoring and alerting
   - Benchmark comparisons with legacy system

3. **Reliability Engineering**
   - Circuit breaker patterns for worker failures
   - Health check endpoints for all workers
   - Graceful shutdown procedures
   - Automated recovery mechanisms

4. **Documentation and Training**
   - Comprehensive architecture documentation
   - Worker development guidelines
   - Troubleshooting guides
   - Migration procedures

## Implementation Strategy

### Phase 1: Core Infrastructure (Issues 1-3)
- Implement message bus and communication layer
- Create base worker framework and interfaces
- Establish orchestrator foundation
- Build configuration management

### Phase 2: Pipeline Workers (Issues 4-8)  
- Implement individual pipeline workers
- Integrate with legacy face recognition engine
- Add event processing and notification routing
- Performance testing and optimization

### Phase 3: Integration and Migration (Issues 12-14)
- Build orchestrator manager and legacy adapter
- Implement migration tools and procedures
- Update application entry points
- End-to-end integration testing

### Phase 4: Production Readiness (Issues 15-16)
- Comprehensive testing framework
- Production monitoring and observability
- Deployment automation and scaling
- Security hardening and compliance

## References

- **Frigate NVR Architecture**: Inspiration for pipeline design and worker patterns
- **ADR-001**: Original system architecture (partially superseded)
- **ADR-004**: Testing strategy (updated for pipeline testing)
- **Issue #12**: Pipeline Orchestrator Integration implementation
- **Issue #13**: Detector strategy implementations
- **Issue #14**: Main application integration and migration

## Success Metrics

- **Performance**: 30% improvement in processing latency over legacy system
- **Reliability**: 99.9% uptime with automatic failure recovery
- **Scalability**: Support for 10x load increase through worker scaling
- **Maintainability**: 50% reduction in time to implement new features
- **Migration**: Zero-downtime migration from legacy to pipeline architecture