
title: "ADR 003: Generative AI Integration" date: 2025-10-23 status: accepted

Context
Starting with version 0.15, Frigate offers an optional generative AI feature that produces human‑readable descriptions of tracked objects. At the end of an object's lifecycle, Frigate collects a set of thumbnails and sends them to a large language model with image support (e.g. OpenAI's Vision models, Google's Gemini API or a local Ollama server) 【12†L138-L146】. The model returns a textual caption that describes what happened in the event (e.g. "A person wearing a red jacket delivered a package and walked away"). These descriptions are stored alongside the semantic index and can be queried via search【14†L283-L292】【14†L339-L347】.
The generative AI feature is fully optional and requires configuration of an API provider and credentials. It does not affect core functions such as detection or face recognition - it merely adds descriptive text.
Decision
We will enable generative AI descriptions as an optional enrichment in the integration. The default configuration will disable the feature. If a user provides API credentials and chooses a provider, the system will send end‑of‑event thumbnails and metadata to that provider and store the returned description.
Key aspects of this decision:
* Opt‑in only. Generative AI calls will never be made unless the user explicitly enables them and supplies the necessary credentials.
* Provider abstraction. The integration will implement a provider‑agnostic interface similar to Frigate's genai module. New providers (e.g. local models via Ollama) can be added by implementing this interface.
* Separation of concerns. Generative captions will not influence detection, tracking or recognition. They will only be stored as additional metadata. Users can choose to display them in the UI or ignore them entirely.
* Privacy safeguards. When external providers are used, we will clearly document that images are sent to external servers【12†L153-L161】. To avoid sending entire frames, we will send only the selected thumbnails (as configured). A local provider option will be supported for privacy‑conscious deployments.
Alternatives considered
* Mandatory captions. Making AI captions compulsory would improve search but would violate privacy expectations and increase cost.
* Local caption generation only. Restricting to a local model (e.g. via Ollama) would avoid external calls but significantly raise hardware requirements【12†L199-L207】. We prefer giving users a choice.
Consequences
* Additional configuration UI/API is required to manage provider selection, API keys and caption preferences (e.g. number of thumbnails, description length).
* The integration must handle API errors and timeouts gracefully. If caption generation fails, the event should still be saved without a description.
* When using cloud providers, there may be usage costs and privacy implications. These must be clearly documented for users.

