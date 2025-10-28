# ADR-001: System Architecture - Frigate-Inspired Pipeline Design

**Title**: "ADR 001: System Architecture - Modular Monolith with Pipeline Processing"  
**Date**: 2025-01-09  
**Status**: **Accepted** ✅ | Foundation for Issues #1-16

## Context

The doorbell security system requires a sophisticated architecture capable of **real-time video processing**, **local privacy-first face recognition**, and **high-performance event handling** on edge devices. Taking inspiration from **Frigate NVR's proven architecture**, we need a design that balances performance, maintainability, and deployment simplicity.

Frigate is an open source NVR that processes video streams on-premises through a **modular monolith architecture**. Its pipeline consists of frame capture via FFmpeg/go2rtc, motion detection, object detection (YOLO/TensorRT/TFLite), object tracking, and post-processing steps such as face recognition and license plate recognition. All components run inside a single service, communicating via **shared memory and in-process publish/subscribe queues**. This modular monolith allows Frigate to be deployed on a single host without requiring additional infrastructure while ensuring all computation stays local unless explicitly configured otherwise.

### Core Requirements
- **Real-time processing**: <100ms end-to-end doorbell event processing
- **Privacy-first**: All biometric processing local, no cloud dependencies  
- **Edge deployment**: Efficient operation on Raspberry Pi 4 hardware
- **High availability**: 99.9% uptime with graceful failure handling
- **Scalability**: Support for multiple detection scenarios and concurrent processing
- **Extensibility**: Plugin architecture for new detectors and enrichments

### Technical Challenges
- **Resource constraints**: Limited CPU, memory, and storage on edge devices
- **Real-time requirements**: Video processing with strict latency constraints
- **Hardware diversity**: Support for CPU, GPU, and EdgeTPU acceleration
- **Reliability**: Robust operation in uncontrolled environment conditions
- **Performance optimization**: Maximize throughput while minimizing resource usage

## Decision

We choose to **adopt Frigate's modular monolith architecture** with **pipeline processing** for our doorbell security system. All core processing (GPIO event handling, frame capture, motion detection, face detection, face recognition, event enrichment) will remain in a **single service container** with internal modules communicating via **message queues and shared memory**.

### Architectural Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                FRIGATE-INSPIRED DOORBELL PIPELINE                  │
├─────────────────────────────────────────────────────────────────────┤
│  GPIO Event → Frame Capture → Motion Detection → Face Detection    │
│       ↓              ↓              ↓               ↓              │
│  Message Bus ←→ Priority Queues ←→ Worker Pools ←→ Result Pipeline  │
│       ↓              ↓              ↓               ↓              │
│  Face Recognition → Event Enrichment → Notifications → Storage     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Core Design Principles

**1. Modular Monolith Architecture**
```
Single Container Deployment:
├── Pipeline Orchestrator (Central Coordinator)
├── Communication Infrastructure (ZeroMQ-inspired Message Bus)
├── Worker Pool Management (Multi-threaded/Process)
├── Hardware Abstraction Layer (GPIO, Camera, Storage)
├── Strategy Pattern Detectors (CPU/GPU/EdgeTPU)
└── Event Enrichment Pipeline (Notifications, Logging, Alerts)
```

**Benefits:**
- **Performance and resource efficiency**: Running all components on the same host avoids network latency and simplifies zero-copy frame sharing. We leverage shared memory buffers to reduce CPU overhead when passing video frames between processes.
- **Deployment simplicity**: Users can deploy the system as a single container. Multiple services would require container orchestration, service discovery, and robust fault handling.
- **Privacy by design**: Keeping the pipeline local ensures that video feeds and face encodings never leave the host unless explicitly configured. A monolithic service has a smaller attack surface than networked services.
- **Ease of integration**: The doorbell system can be treated as a cohesive component with clear API boundaries.

**2. Event-Driven Pipeline Processing**
```python
class PipelineStage:
    """Base class for pipeline processing stages (inspired by Frigate)."""
    
    def process_event(self, event: PipelineEvent) -> PipelineEvent:
        """Process event and pass to next stage with zero-copy optimization."""
        
    def handle_error(self, error: Exception, event: PipelineEvent):
        """Handle processing errors with graceful degradation."""
        
    def get_health_status(self) -> HealthStatus:
        """Report stage health and performance metrics."""
```

**Pipeline Flow (Frigate-inspired):**
1. **GPIO Event Detection** → Doorbell button press triggers processing pipeline
2. **Frame Capture** → High-resolution camera capture with ring buffer management
3. **Motion Detection** → Background subtraction to validate genuine events
4. **Face Detection** → Multi-strategy detection (CPU/GPU/EdgeTPU) via detector factory
5. **Face Recognition** → Local face matching with encoding comparison
6. **Event Enrichment** → Metadata addition, notification routing, audit logging
7. **Storage & Delivery** → Event persistence and notification delivery

**3. Strategy Pattern for Hardware Acceleration**
```python
class DetectorFactory:
    """Factory for optimal detector selection (Frigate-inspired pattern)."""
    
    @staticmethod
    def create_optimal_detector() -> BaseDetector:
        """Auto-detect best available hardware and create detector."""
        
        # Priority: EdgeTPU → GPU → CPU → Mock (Frigate's approach)
        if EdgeTPUDetector.is_available():
            return EdgeTPUDetector(config.edgetpu)
        elif GPUDetector.is_available():
            return GPUDetector(config.gpu)
        else:
            return CPUDetector(config.cpu)
```

**4. High-Performance Communication Infrastructure**
```python
class MessageBus:
    """ZeroMQ-inspired message bus with Frigate optimization patterns."""
    
    # Features (inspired by Frigate's inter-process communication):
    - Topic-based pub/sub with wildcard routing
    - Priority queues for time-sensitive events (doorbell triggers)
    - Zero-copy optimization for frame data sharing
    - Backpressure handling and flow control
    - Message correlation and distributed tracing
    - Comprehensive performance monitoring and health checks
```

### Performance Architecture (Frigate-Inspired)

**Multi-Processing Worker Pools:**
```
Frame Capture Worker:     Single-threaded, ring buffer (Frigate pattern)
Motion Detection Pool:    2-4 workers, CPU-optimized background subtraction
Face Detection Pool:      1-8 workers, hardware-dependent (CPU/GPU/EdgeTPU)  
Face Recognition Pool:    2-4 workers, face_recognition/dlib optimization
Event Processing Pool:    1-2 workers, I/O and notification optimization
```

**Memory Management (Frigate Approach):**
- **Ring buffers**: Efficient frame storage with automatic cleanup
- **Shared memory**: Zero-copy frame data between processes
- **Object pooling**: Reuse expensive objects (face encodings, detectors)
- **Garbage collection tuning**: Optimized for real-time constraints

**Resource Optimization:**
- **Dynamic worker scaling**: Adjust workers based on load and hardware
- **Hardware detection**: Automatic optimization for available acceleration
- **Performance monitoring**: Real-time metrics collection and alerting
- **Resource limits**: Prevent resource exhaustion and OOM conditions

### Production Features

**Fault Tolerance and Recovery (Frigate-Inspired):**
```python
class PipelineOrchestrator:
    """Main system coordinator with Frigate-style fault tolerance."""
    
    def handle_worker_failure(self, worker_id: str, error: Exception):
        """Handle worker failures with automatic recovery."""
        # 1. Isolate failed worker
        # 2. Redistribute pending work to healthy workers
        # 3. Restart worker with exponential backoff
        # 4. Monitor recovery success and health restoration
        # 5. Alert operators if persistent failures detected
```

**Configuration Management:**
```python
@dataclass
class PipelineConfig:
    """Comprehensive pipeline configuration (Frigate-inspired)."""
    
    # Hardware optimization (following Frigate's patterns)
    hardware_acceleration: bool = True
    detector_type: str = "auto"  # auto, cpu, gpu, edgetpu
    worker_scaling: str = "auto"  # auto, fixed, dynamic
    
    # Performance tuning (Frigate optimization approach)
    frame_buffer_size: int = 30  # Ring buffer size for frame management
    processing_timeout: float = 10.0  # Worker timeout for stuck processes
    batch_processing: bool = True  # Batch optimization for efficiency
    
    # Quality settings (Frigate-style detection thresholds)
    face_detection_confidence: float = 0.7
    face_recognition_tolerance: float = 0.6
    motion_detection_sensitivity: float = 0.5
```

## Alternatives Considered

**Microservices Architecture**: Breaking the system into separate services (e.g., detector microservice, recognition microservice) would allow scaling individual components independently and potentially improve resilience. However, this would add significant complexity to deployment, require distributed message bus and persistence layer, and introduce network latency for high-throughput video frames. The current requirements do not justify this overhead.

**Hybrid Approach**: Some parts could be split off (e.g., face recognition as separate service) while keeping core detection and tracking together. This still requires coordination and service discovery. We postpone this until a clear need arises.

**Direct Integration**: Embedding all functionality directly into existing application code would eliminate process boundaries but would create tight coupling and make testing, scaling, and maintenance significantly more difficult.

## Consequences

### Positive Impacts ✅

**Performance and Resource Efficiency:**
- Running all components on the same host avoids network latency between services
- Zero-copy frame sharing reduces CPU overhead and memory pressure
- Optimized for edge device resource constraints (Raspberry Pi)
- Hardware acceleration support (EdgeTPU/GPU) provides 5-10x performance improvements

**Deployment and Operational Simplicity:**
- Single container deployment simplifies installation and management
- No need for container orchestration, service discovery, or distributed coordination
- Unified logging, monitoring, and debugging capabilities
- Simple backup and recovery procedures

**Privacy and Security:**
- All video processing and face recognition stays local by default
- Smaller attack surface compared to distributed microservices
- No network communication required for core functionality
- Easy to audit and verify privacy compliance

**Development and Maintenance:**
- Proven architecture pattern (Frigate) with established best practices
- Clear module boundaries enable independent development and testing
- Easier debugging with unified logging and tracing
- Simplified dependency management and version control

### Negative Impacts ⚠️

**Scalability Limitations:**
- Scaling beyond the capacity of a single machine requires running multiple instances rather than scaling individual components
- Vertical scaling is the primary approach; horizontal scaling requires additional architectural work
- Cannot independently scale computationally expensive components (face recognition)

**Complexity and Coordination:**
- All modules must coordinate within a single process space
- Shared resource management requires careful attention to prevent conflicts
- Debugging async communication patterns can be more complex than direct function calls
- Worker coordination and message serialization introduce CPU overhead

**Deployment Constraints:**
- Single point of failure for the entire processing pipeline
- Updates require restarting the entire system rather than individual components
- Resource-intensive components can impact the performance of other modules
- Hardware-specific optimizations must work within shared environment

### Mitigation Strategies

**Scalability Management:**
- Design with clear module boundaries for future microservice extraction if needed
- Implement horizontal scaling through pipeline orchestrator load balancing
- Provide configuration options for resource allocation between components
- Monitor system performance and plan for architectural evolution

**Complexity Reduction:**
- Comprehensive documentation following Frigate's patterns and best practices
- Rich debugging tools with pipeline visualization and event tracing
- Simplified configuration with intelligent defaults and validation
- Team training on pipeline patterns and async communication

**Operational Excellence:**
- Comprehensive test coverage including failure scenarios and edge cases
- Monitoring dashboards with intelligent alerting and health checks
- Clear operational runbooks and troubleshooting guides
- Automated deployment with infrastructure as code

## Implementation Status

This ADR forms the foundation for the entire system implementation across Issues #1-16:

### Foundation (Issues 1-3) ✅
- [x] Communication infrastructure and message bus implementation
- [x] Base detector framework and factory pattern
- [x] Pipeline configuration and orchestration foundation

### Core Pipeline (Issues 4-11) ✅  
- [x] Frame capture and motion detection workers
- [x] Face detection worker pool with strategy pattern
- [x] Face recognition engine with caching optimization
- [x] Event processing and enrichment pipeline
- [x] Hardware abstraction layer and storage integration

### Production Integration (Issues 12-16) 🔄
- [ ] Pipeline orchestrator integration and legacy migration
- [ ] Hardware-accelerated detector implementations
- [ ] Main application integration with backward compatibility
- [ ] Comprehensive testing framework and quality assurance
- [ ] Production deployment infrastructure and monitoring

## Related ADRs
- **ADR-002**: Face Recognition Implementation Strategy (building on Frigate's approach)
- **ADR-005**: Pipeline Architecture and Orchestration Details
- **ADR-006**: Communication Infrastructure and Message Bus Implementation
- **ADR-007**: Detector Strategy Pattern and Hardware Abstraction

## References
- Frigate NVR Architecture Documentation: [https://docs.frigate.video/](https://docs.frigate.video/)
- Frigate GitHub Repository: [https://github.com/blakeblackshear/frigate](https://github.com/blakeblackshear/frigate)
- Pipeline Pattern Implementation in Real-time Video Processing Systems
- Edge Computing Architecture Patterns for Computer Vision Applications

---

**This ADR should be revisited if operational experience shows that separating services would provide significant benefits or if scale requirements increase dramatically beyond single-host capabilities.**