
title: "ADR 001: System Architecture - Modular Monolith" date: 2025-10-23 status: accepted

Context
Frigate is an open source NVR that processes video streams on-premises. Its pipeline consists of frame capture via FFmpeg/go2rtc, motion detection, object detection (YOLO/TensorRT/TFLite), object tracking, and a series of post-processing steps such as face recognition and licence plate recognition?24†L155-L163?. All of these components run inside a single service, communicating via shared memory and in-process publish/subscribe queues?18†L197-L202?. The simplicity of this modular monolith allows Frigate to be deployed on a single host without requiring additional infrastructure. It also ensures that all computation stays local unless the user explicitly enables cloud services (e.g., for generative descriptions).
The current project aims to integrate Frigate into another application. A key question is whether to maintain the monolithic design or break the system into separate microservices (for example, separate processes for camera ingest, detection, tracking and enrichment) or use a hybrid approach. Each option has trade-offs in complexity, scalability and performance.
Decision
We choose to retain the modular monolith architecture for the initial integration. All core processing (frame capture, detection, tracking, enrichment) will remain in a single service container. Internal modules communicate via message queues and shared memory but are not split into independently deployable microservices. The reasons for this decision include:
* Performance and resource efficiency. Running all components on the same host avoids network latency between services and simplifies zero-copy frame sharing. Frigate leverages shared memory buffers to reduce CPU overhead when passing video frames between processes?18†L197-L202?.
* Deployment simplicity. Users can deploy the NVR as a single container. Introducing multiple services would require container orchestration, service discovery and robust fault handling which may complicate installation.
* Privacy by design. Keeping the pipeline local ensures that video feeds and embeddings never leave the host unless explicitly configured. A monolithic service has a smaller attack surface than a collection of networked services.
* Ease of integration. Since the target application integrates via APIs (MQTT/HTTP) and not by directly inserting microservices, a monolithic Frigate can be treated as an external component. Splitting it would impose additional complexity on the integration.
Alternatives considered
* Microservices. Breaking the NVR into separate services (e.g., a detector microservice, tracker microservice, etc.) would allow scaling individual components independently and potentially improve resilience. However, this would add significant complexity to deployment, require a distributed message bus and persistence layer, and introduce network latency for high-throughput video frames. The current requirements do not justify this overhead.
* Hybrid approach. Some parts could be split off (e.g., face recognition) while keeping core detection and tracking together. This still requires coordination and service discovery. We postpone this until a clear need arises.
Consequences
* The integration remains simple to deploy and operate. A single container or process provides the full NVR functionality.
* Scaling beyond the capacity of a single machine will require running multiple instances of the monolith rather than scaling individual components. Additional architectural work would be needed if future requirements demand distributed processing.
* New features must be designed to fit into the modular structure. For example, new enrichments should be implemented as additional real-time processors within the same process.
This ADR should be revisited if operational experience shows that separating services would provide significant benefits or if the scale requirements increase dramatically.

