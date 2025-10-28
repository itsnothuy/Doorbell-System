# ADR-003: Generative AI Integration - Optional Event Description Enhancement

**Title**: "ADR 003: Generative AI Integration - Privacy-Aware Event Enrichment"  
**Date**: 2025-01-09  
**Status**: **Accepted** ✅ | Implemented in Issue #11, Enhanced in Issues #15-16

## Context

Following **Frigate NVR's approach to generative AI**, we need an **optional event enrichment system** that provides human-readable descriptions of doorbell events without compromising the system's **privacy-first architecture**. Starting with Frigate 0.15+, generative AI produces descriptive captions by sending event thumbnails to large language models with image support (e.g., OpenAI Vision, Google Gemini, or local Ollama servers).

The generative AI feature in Frigate is **fully optional** and requires explicit configuration of API providers and credentials. Importantly, it **does not affect core functions** such as detection or face recognition—it merely adds descriptive text to enhance searchability and user experience.

### Core Requirements
- **Privacy-first**: Generative AI must be **opt-in only** with clear privacy implications
- **Provider flexibility**: Support for both **cloud providers** and **local models**
- **Non-intrusive**: Should not impact core doorbell functionality or performance
- **Configurable privacy**: Users control what data (if any) is sent externally
- **Quality thumbnails**: Intelligent selection of representative event images
- **Error resilience**: System continues operating if AI generation fails

### Technical Challenges
- **Privacy balance**: Provide AI benefits while maintaining privacy guarantees
- **Provider abstraction**: Support multiple AI providers with unified interface
- **Cost management**: Control API costs and usage for cloud providers
- **Local deployment**: Support resource-intensive local models when required
- **Error handling**: Graceful degradation when AI services are unavailable

## Decision

We implement **optional generative AI descriptions** as an **event enrichment processor** following Frigate's architecture. The system provides **provider-agnostic AI integration** with **strong privacy safeguards**, **cost controls**, and **local processing options**.

### Architecture Design

#### 1. **Provider-Agnostic AI Interface**

```python
class GenerativeAIProvider(ABC):
    """Abstract interface for generative AI providers."""
    
    @abstractmethod
    def generate_description(self, images: List[np.ndarray], 
                           context: EventContext) -> AIDescription:
        """Generate human-readable description from event images."""
        pass
    
    @abstractmethod
    def validate_configuration(self) -> bool:
        """Validate provider configuration and credentials."""
        pass
    
    @abstractmethod
    def estimate_cost(self, image_count: int, context_length: int) -> float:
        """Estimate API cost for given input."""
        pass
    
    @abstractmethod
    def get_privacy_implications(self) -> List[str]:
        """Return privacy implications of using this provider."""
        pass
```

#### 2. **Cloud Provider Implementations**

```python
class OpenAIVisionProvider(GenerativeAIProvider):
    """OpenAI GPT-4 Vision provider for image descriptions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI provider with privacy controls."""
        self.api_key = config['api_key']
        self.model = config.get('model', 'gpt-4-vision-preview')
        self.max_tokens = config.get('max_tokens', 300)
        self.max_images = config.get('max_images', 3)
        
        # Privacy and cost controls
        self.resize_images = config.get('resize_images', True)
        self.max_image_size = config.get('max_image_size', (512, 512))
        self.strip_metadata = config.get('strip_metadata', True)
        
        self.client = OpenAI(api_key=self.api_key)
    
    def generate_description(self, images: List[np.ndarray], 
                           context: EventContext) -> AIDescription:
        """Generate description using OpenAI Vision API."""
        try:
            # Prepare images with privacy controls
            processed_images = self._prepare_images_for_api(images)
            
            # Build context-aware prompt
            prompt = self._build_context_prompt(context)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            *[{"type": "image_url", 
                               "image_url": {"url": img_url}} 
                              for img_url in processed_images]
                        ]
                    }
                ],
                max_tokens=self.max_tokens
            )
            
            description = response.choices[0].message.content
            
            return AIDescription(
                text=description,
                confidence=0.9,  # OpenAI doesn't provide confidence
                provider="openai_vision",
                cost=self._calculate_cost(response.usage),
                privacy_level="external_cloud"
            )
            
        except Exception as e:
            logger.error(f"OpenAI Vision API failed: {e}")
            return AIDescription(
                text="",
                confidence=0.0,
                provider="openai_vision",
                error=str(e),
                privacy_level="external_cloud"
            )
    
    def get_privacy_implications(self) -> List[str]:
        """Return privacy implications for OpenAI."""
        return [
            "Images are sent to OpenAI's external servers",
            "OpenAI may store images temporarily for processing",
            "Data processing occurs outside your premises",
            "Subject to OpenAI's privacy policy and terms of service",
            "Potential for data analysis by third parties"
        ]


class GeminiVisionProvider(GenerativeAIProvider):
    """Google Gemini Vision provider for image descriptions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Gemini provider."""
        self.api_key = config['api_key']
        self.model = config.get('model', 'gemini-pro-vision')
        
        # Configure Google AI client
        genai.configure(api_key=self.api_key)
        self.model_client = genai.GenerativeModel(self.model)
    
    def generate_description(self, images: List[np.ndarray], 
                           context: EventContext) -> AIDescription:
        """Generate description using Gemini Vision API."""
        # Implementation similar to OpenAI but with Gemini-specific API calls
        pass
```

#### 3. **Local AI Provider Implementation**

```python
class OllamaLocalProvider(GenerativeAIProvider):
    """Local Ollama provider for privacy-first AI descriptions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize local Ollama provider."""
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.model = config.get('model', 'llava:latest')
        self.timeout = config.get('timeout', 30.0)
        
        # Local processing benefits
        self.unlimited_images = True
        self.no_cost_per_request = True
        self.full_privacy = True
    
    def generate_description(self, images: List[np.ndarray], 
                           context: EventContext) -> AIDescription:
        """Generate description using local Ollama model."""
        try:
            # Prepare images for local processing
            processed_images = self._prepare_images_for_local(images)
            
            # Build prompt for vision model
            prompt = self._build_local_prompt(context)
            
            # Call local Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": processed_images,
                    "stream": False
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                description = result.get('response', '')
                
                return AIDescription(
                    text=description,
                    confidence=0.8,  # Local models may be less confident
                    provider="ollama_local",
                    cost=0.0,  # No API cost for local processing
                    privacy_level="fully_local"
                )
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Local Ollama generation failed: {e}")
            return AIDescription(
                text="",
                confidence=0.0,
                provider="ollama_local",
                error=str(e),
                privacy_level="fully_local"
            )
    
    def get_privacy_implications(self) -> List[str]:
        """Return privacy implications for local processing."""
        return [
            "All processing occurs locally on your device",
            "No data is sent to external servers",
            "Complete privacy and data control",
            "Requires significant local computational resources",
            "Model updates and maintenance are your responsibility"
        ]
```

#### 4. **Event Enrichment Integration**

```python
class GenerativeAIEnrichment(BaseEnrichment):
    """AI description enrichment processor."""
    
    def __init__(self, message_bus: MessageBus, config: Dict[str, Any]):
        """Initialize AI enrichment with provider configuration."""
        super().__init__(message_bus, config)
        
        # Configuration
        self.enabled = config.get('enabled', False)
        self.provider_name = config.get('provider', 'disabled')
        self.max_images = config.get('max_images', 3)
        self.min_event_duration = config.get('min_duration', 2.0)
        
        # Cost and usage controls
        self.daily_cost_limit = config.get('daily_cost_limit', 5.0)
        self.monthly_cost_limit = config.get('monthly_cost_limit', 50.0)
        self.current_daily_cost = 0.0
        self.current_monthly_cost = 0.0
        
        # Initialize provider
        self.provider = self._initialize_provider(config.get('provider_config', {}))
        
        # Privacy tracking
        self.privacy_consent_given = config.get('privacy_consent', False)
        
    def enrich_event(self, event: EventData) -> EventData:
        """Enrich event with AI-generated description."""
        # Check if AI enrichment should be applied
        if not self._should_enrich_event(event):
            return event
        
        try:
            # Select best images from event
            selected_images = self._select_representative_images(event)
            
            if not selected_images:
                logger.debug(f"No suitable images for AI description: {event.event_id}")
                return event
            
            # Check cost limits before API call
            estimated_cost = self.provider.estimate_cost(
                len(selected_images), 
                len(event.context or {})
            )
            
            if not self._check_cost_limits(estimated_cost):
                logger.warning("AI description skipped due to cost limits")
                return event
            
            # Generate AI description
            context = EventContext(
                event_type=event.event_type,
                timestamp=event.timestamp,
                location="doorbell entrance",
                previous_events=event.get('context', {}),
                face_detections=event.get('face_results', [])
            )
            
            ai_result = self.provider.generate_description(selected_images, context)
            
            # Update cost tracking
            if ai_result.cost > 0:
                self._update_cost_tracking(ai_result.cost)
            
            # Add AI description to event
            enriched_event = event.copy()
            enriched_event.enrichments = enriched_event.enrichments + ['ai_description']
            enriched_event.data.update({
                'ai_description': {
                    'text': ai_result.text,
                    'confidence': ai_result.confidence,
                    'provider': ai_result.provider,
                    'cost': ai_result.cost,
                    'privacy_level': ai_result.privacy_level,
                    'generation_time': time.time()
                }
            })
            
            logger.info(f"Generated AI description for event {event.event_id}: {ai_result.text[:100]}...")
            return enriched_event
            
        except Exception as e:
            logger.error(f"AI enrichment failed for event {event.event_id}: {e}")
            return event
    
    def _should_enrich_event(self, event: EventData) -> bool:
        """Determine if event should receive AI enrichment."""
        # Check basic enablement
        if not self.enabled or not self.provider:
            return False
        
        # Check privacy consent
        if not self.privacy_consent_given and self.provider.get_privacy_implications():
            return False
        
        # Check event type (only enrich doorbell events)
        if event.event_type != EventType.DOORBELL_EVENT_COMPLETE:
            return False
        
        # Check event duration (skip very short events)
        duration = event.data.get('duration', 0)
        if duration < self.min_event_duration:
            return False
        
        # Check if faces were detected (may influence description)
        has_faces = bool(event.data.get('face_results', []))
        
        return True
    
    def _select_representative_images(self, event: EventData) -> List[np.ndarray]:
        """Select best representative images from event."""
        # Get all captured frames
        frame_data = event.data.get('captured_frames', [])
        
        if not frame_data:
            return []
        
        # Quality-based selection criteria
        selected_frames = []
        
        # 1. Select frame with best face detection (if any)
        face_frames = [f for f in frame_data if f.get('face_count', 0) > 0]
        if face_frames:
            best_face_frame = max(face_frames, key=lambda f: f.get('face_confidence', 0))
            selected_frames.append(best_face_frame['image'])
        
        # 2. Select frame with most motion/activity
        motion_frames = sorted(frame_data, key=lambda f: f.get('motion_score', 0), reverse=True)
        if motion_frames and len(selected_frames) < self.max_images:
            selected_frames.append(motion_frames[0]['image'])
        
        # 3. Select temporal diversity (beginning, middle, end)
        if len(frame_data) >= 3 and len(selected_frames) < self.max_images:
            temporal_indices = [0, len(frame_data) // 2, len(frame_data) - 1]
            for idx in temporal_indices:
                if len(selected_frames) < self.max_images:
                    frame = frame_data[idx]['image']
                    if not self._is_similar_to_selected(frame, selected_frames):
                        selected_frames.append(frame)
        
        return selected_frames[:self.max_images]
```

### Privacy and Configuration Management

#### 1. **Privacy-Aware Configuration**

```python
@dataclass
class GenerativeAIConfig:
    """Configuration for generative AI with privacy controls."""
    
    # Basic enablement (default: disabled)
    enabled: bool = False
    provider: str = "disabled"  # disabled, openai, gemini, ollama
    
    # Privacy controls
    privacy_consent_given: bool = False
    require_explicit_consent: bool = True
    log_privacy_decisions: bool = True
    
    # Provider-specific settings
    provider_config: Dict[str, Any] = field(default_factory=dict)
    
    # Image processing controls
    max_images_per_event: int = 3
    resize_images_for_api: bool = True
    max_image_dimensions: Tuple[int, int] = (512, 512)
    strip_image_metadata: bool = True
    
    # Cost and usage controls
    daily_cost_limit: float = 5.0
    monthly_cost_limit: float = 50.0
    enable_cost_alerts: bool = True
    
    # Quality controls
    min_event_duration: float = 2.0
    min_image_quality: float = 0.7
    max_description_length: int = 300
    
    # Error handling
    timeout_seconds: float = 30.0
    max_retries: int = 2
    fallback_on_error: bool = True
```

#### 2. **Privacy Consent Management**

```python
class PrivacyConsentManager:
    """Manage privacy consent for AI features."""
    
    def __init__(self, storage_path: Path):
        """Initialize consent management."""
        self.storage_path = storage_path
        self.consent_records: Dict[str, ConsentRecord] = {}
        self._load_consent_records()
    
    def request_consent(self, provider: GenerativeAIProvider, 
                       user_context: str) -> ConsentDecision:
        """Request user consent for AI provider usage."""
        implications = provider.get_privacy_implications()
        
        # Present privacy implications to user
        consent_request = ConsentRequest(
            provider_name=provider.__class__.__name__,
            privacy_implications=implications,
            estimated_cost_per_month=provider.estimate_monthly_cost(),
            alternative_options=self._get_privacy_alternatives(),
            user_context=user_context
        )
        
        # In a real implementation, this would present UI to user
        # For this ADR, we show the conceptual framework
        decision = self._present_consent_ui(consent_request)
        
        # Record consent decision
        self._record_consent_decision(decision)
        
        return decision
    
    def _get_privacy_alternatives(self) -> List[str]:
        """Get privacy-preserving alternatives."""
        return [
            "Use local Ollama model (requires powerful hardware)",
            "Disable AI descriptions (manual event review only)",
            "Use cloud provider with image anonymization",
            "Schedule periodic AI processing in batches"
        ]
```

### Error Handling and Resilience

```python
class AIGenerationErrorHandler:
    """Handle errors in AI generation gracefully."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize error handling."""
        self.max_retries = config.get('max_retries', 2)
        self.retry_delay = config.get('retry_delay', 5.0)
        self.fallback_enabled = config.get('fallback_enabled', True)
        
        # Error tracking
        self.error_counts: Dict[str, int] = {}
        self.last_success_time = time.time()
        
    def handle_generation_error(self, error: Exception, 
                               provider: GenerativeAIProvider,
                               event: EventData) -> Optional[AIDescription]:
        """Handle AI generation error with fallback strategies."""
        
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        logger.warning(f"AI generation failed for {provider.__class__.__name__}: {error}")
        
        # Attempt retry with exponential backoff
        if self.error_counts[error_type] <= self.max_retries:
            delay = self.retry_delay * (2 ** (self.error_counts[error_type] - 1))
            time.sleep(delay)
            
            try:
                # Retry with reduced image set
                reduced_images = event.data.get('captured_frames', [])[:1]
                return provider.generate_description(
                    reduced_images, 
                    EventContext.from_event(event)
                )
            except Exception as retry_error:
                logger.error(f"AI generation retry failed: {retry_error}")
        
        # Fallback to rule-based description
        if self.fallback_enabled:
            return self._generate_fallback_description(event)
        
        return None
    
    def _generate_fallback_description(self, event: EventData) -> AIDescription:
        """Generate simple rule-based description as fallback."""
        
        # Analyze event data for basic description
        has_faces = bool(event.data.get('face_results', []))
        duration = event.data.get('duration', 0)
        motion_detected = bool(event.data.get('motion_score', 0) > 0.5)
        
        # Build simple description
        description_parts = []
        
        if has_faces:
            face_count = len(event.data.get('face_results', []))
            if face_count == 1:
                description_parts.append("A person approached the doorbell")
            else:
                description_parts.append(f"{face_count} people approached the doorbell")
        else:
            description_parts.append("Motion detected at the doorbell")
        
        if duration > 10:
            description_parts.append(f"and remained for {duration:.0f} seconds")
        
        fallback_text = " ".join(description_parts) + "."
        
        return AIDescription(
            text=fallback_text,
            confidence=0.5,  # Lower confidence for rule-based
            provider="rule_based_fallback",
            cost=0.0,
            privacy_level="fully_local"
        )
```

## Implementation Status

### Core AI Integration ✅ (Issue #11)
- [x] Provider-agnostic AI interface and factory pattern
- [x] OpenAI Vision and Gemini provider implementations
- [x] Local Ollama provider for privacy-first processing
- [x] Event enrichment integration with pipeline
- [x] Privacy consent management and controls

### Production Features 🔄 (Issues #15-16)
- [ ] Advanced cost management and usage analytics
- [ ] Enhanced error handling and retry mechanisms
- [ ] Performance optimization for local model deployment
- [ ] Comprehensive privacy audit and compliance validation

### Testing and Validation ✅
- [x] Mock provider implementations for testing
- [x] Privacy compliance validation framework
- [x] Cost estimation and tracking systems
- [x] Error scenario testing and fallback validation

## Consequences

### Positive Impacts ✅

**Enhanced User Experience:**
- **Rich event descriptions**: Human-readable summaries improve event browsing and search
- **Contextual information**: AI provides context that pure detection cannot capture
- **Search capability**: Generated descriptions enable natural language event search
- **Historical insights**: Pattern recognition in event descriptions over time

**Privacy and Control Benefits:**
- **Opt-in only**: AI features never activate without explicit user consent
- **Provider choice**: Users can choose between cloud, local, or no AI processing
- **Cost transparency**: Clear cost estimation and limits for cloud providers
- **Privacy alternatives**: Local processing options for privacy-conscious users

**Technical Benefits:**
- **Non-intrusive integration**: AI enrichment doesn't impact core system performance
- **Graceful degradation**: System operates normally if AI generation fails
- **Provider flexibility**: Easy to add new AI providers as they become available
- **Configurable quality**: Adjustable parameters for accuracy vs. cost trade-offs

### Negative Impacts ⚠️

**Complexity and Maintenance:**
- **Configuration complexity**: Multiple provider options and settings increase complexity
- **API dependencies**: Reliance on external APIs introduces potential failure points
- **Cost management**: Ongoing monitoring required for cloud provider usage
- **Privacy compliance**: Additional considerations for data protection and consent

**Performance and Resource Usage:**
- **Additional processing**: AI generation adds computational and time overhead
- **Network requirements**: Cloud providers require reliable internet connectivity
- **Local resource demands**: Local AI models require significant computational resources
- **Storage overhead**: Generated descriptions and metadata require additional storage

**Privacy and Security Considerations:**
- **External data sharing**: Cloud providers involve sending images to third parties
- **API security**: Additional attack vectors through AI provider APIs
- **Consent management**: Complex privacy consent workflows for users
- **Data retention**: Understanding and managing AI provider data retention policies

### Mitigation Strategies

**Privacy Protection:**
- Clear documentation of privacy implications for each provider option
- Strong default privacy settings with explicit opt-in requirements
- Local processing options for users requiring complete privacy
- Regular privacy audits and compliance validation

**Cost and Resource Management:**
- Intelligent image selection to minimize API costs
- Configurable cost limits with automatic cutoffs
- Local provider options to eliminate ongoing costs
- Resource monitoring and optimization for local deployments

**Reliability and Performance:**
- Robust error handling with graceful fallback mechanisms
- Provider health monitoring and automatic failover
- Configurable timeouts and retry policies
- Performance impact monitoring and optimization

## Related ADRs
- **ADR-001**: System Architecture (event enrichment integration)
- **ADR-009**: Security Architecture (privacy and consent management)
- **ADR-011**: Internal Notification System (event processing pipeline)

## References
- Frigate Generative AI Documentation: [https://docs.frigate.video/configuration/genai](https://docs.frigate.video/configuration/genai)
- OpenAI Vision API Documentation: [https://platform.openai.com/docs/guides/vision](https://platform.openai.com/docs/guides/vision)
- Google Gemini Vision API: [https://ai.google.dev/docs/gemini_api_overview](https://ai.google.dev/docs/gemini_api_overview)
- Ollama Local AI Models: [https://ollama.ai/](https://ollama.ai/)
- GDPR Compliance for AI Systems and Privacy-by-Design Principles

---

**This implementation provides enhanced event understanding while maintaining strict privacy controls and giving users complete autonomy over their data sharing decisions.**