
title: "ADR 002: Face Recognition Implementation" date: 2025-10-23 status: accepted

Context
Frigate 0.16 introduces built‑in face recognition that adds a sub‑label to tracked person objects. The implementation uses a two‑stage pipeline:
1. Face detection. A person object triggers the face recogniser. If the main detector model includes a face label (available with Frigate+ models), Frigate uses the provided bounding box. Otherwise, a lightweight detector based on OpenCV's YuNet (FaceDetectorYN) is run on the cropped person region【62†L147-L155】. YuNet is chosen for its speed and accuracy on edge devices.
2. Embedding and classification. The detected face is aligned using facial landmarks and then passed through an embedding model. Frigate supports two models: a small FaceNet variant for CPU‑only systems and a larger ArcFace model for systems with GPU or powerful CPU【62†L164-L172】. The embedding vector is compared to stored vectors in a local face library using cosine similarity. A confidence score determines whether the face matches a known identity or is classified as unknown【62†L173-L176】.
The face library is maintained locally, and users can train new identities via the API. The system explicitly does not use large language models or external services for face recognition; all computation occurs on the host. Generative AI, if enabled, is only used for descriptive text and does not influence recognition【62†L173-L176】.
Decision
We will adopt Frigate's face recognition design without modification for integration. The integration will expose configuration parameters that mirror Frigate's existing settings (e.g. enabled, model_size, recognition_threshold, unknown_score, min_faces) and allow users to manage the face library via API endpoints. The embedding and matching implementation will remain local.
Reasons for this decision:
* Performance and resource constraints. The YuNet + ArcFace/FaceNet pipeline is optimised for real‑time processing on edge hardware. It avoids the overhead of deep object detectors per face and does not require a GPU unless using the large model.
* Privacy. All embeddings and images remain on the device. No external calls are made during recognition, which is important for sensitive use cases.
* Proven accuracy. The combination of facial alignment and modern embedding models provides good accuracy. Confidence thresholds can be tuned by the end user. Additional smoothing such as requiring a minimum number of consecutive matches (min_faces) increases robustness【62†L164-L172】.
* Simplicity for users. Exposing the existing configuration avoids confusing users with new parameters. Training and managing faces via the API remains unchanged.
Consequences
* We inherit Frigate's dependency on OpenCV's DNN module and on downloading the YuNet, ArcFace and FaceNet models at runtime. The integration must ensure these models are cached or downloaded in advance to avoid delays on first use.
* The face library will reside in the integration's storage area. A migration strategy may be needed if integrating with existing face libraries.
* Additional face processing features (e.g. age or emotion estimation) would require a new ADR and should be implemented as separate processors rather than modifying the recognition pipeline.

