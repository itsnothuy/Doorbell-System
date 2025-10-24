# Architecture Documentation

## 🏗️ System Architecture Overview

The Doorbell Security System is designed as a modular, privacy-first face recognition system that operates locally on edge devices. The architecture emphasizes modularity, cross-platform compatibility, and security while maintaining simplicity and reliability.

## 🎯 Architectural Principles

### Design Principles

1. **Privacy by Design**: All biometric processing happens locally
2. **Modularity**: Loosely coupled components with clear interfaces
3. **Cross-Platform**: Works on Raspberry Pi, macOS, Linux, and Docker
4. **Fail-Safe**: Graceful degradation when components are unavailable
5. **Extensibility**: Easy to add new features and integrations
6. **Security First**: Security considerations built into every component
7. **Resource Conscious**: Optimized for resource-constrained devices

### Architectural Patterns

- **Layered Architecture**: Clear separation between hardware, business logic, and presentation
- **Event-Driven**: Asynchronous event processing for real-time responsiveness
- **Plugin Architecture**: Configurable components that can be enabled/disabled
- **Factory Pattern**: Platform-specific implementations with common interfaces
- **Observer Pattern**: Event notification and callback mechanisms

## 🏛️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Presentation Layer                       │
├─────────────────────────┬───────────────────────────────────────┤
│     Web Interface       │         External APIs               │
│   (Flask Dashboard)     │      (Telegram Bot API)             │
└─────────────────────────┴───────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                          │
├─────────────────────────────────────────────────────────────────┤
│                 Doorbell Security System                       │
│              (Main Orchestrator & Controller)                  │
└─────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                       Business Layer                            │
├───────────────┬─────────────────┬───────────────────────────────┤
│ Face Manager  │ Telegram        │      Event Processing        │
│ (Recognition) │ Notifier        │    (Threading & Queuing)     │
└───────────────┴─────────────────┴───────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                     Hardware Abstraction                        │
├───────────────┬─────────────────┬───────────────────────────────┤
│ Camera        │ GPIO Handler    │    Platform Detector         │
│ Handler       │ (Pi/Mock)       │   (Pi/macOS/Linux/Docker)     │
└───────────────┴─────────────────┴───────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                      Infrastructure                             │
├───────────────┬─────────────────┬───────────────────────────────┤
│ Data Storage  │ Configuration   │       Logging                │
│ (Local Files) │ Management      │    (Structured Logs)         │
└───────────────┴─────────────────┴───────────────────────────────┘
```

### Component Interaction Diagram

```
    ┌─────────────────┐
    │   GPIO Event    │
    │  (Doorbell)     │
    └─────────┬───────┘
              │ trigger
              ▼
    ┌─────────────────┐      ┌─────────────────┐
    │ Doorbell        │──────│ Camera Handler  │
    │ Security System │      │                 │
    └─────────┬───────┘      └─────────────────┘
              │                       │
              │ process               │ capture
              ▼                       ▼
    ┌─────────────────┐      ┌─────────────────┐
    │ Face Manager    │      │ Image Data      │
    │                 │◄─────│                 │
    └─────────┬───────┘      └─────────────────┘
              │
              │ identify
              ▼
    ┌─────────────────┐      ┌─────────────────┐
    │ Recognition     │      │ Telegram        │
    │ Result          │─────►│ Notifier        │
    └─────────────────┘      └─────────────────┘
```

## 🧩 Component Architecture

### Core Components

#### 1. Doorbell Security System (`src/doorbell_security.py`)
**Role**: Main orchestrator and event controller
**Responsibilities**:
- Coordinate all system components
- Handle doorbell press events
- Manage event processing workflow
- Implement debouncing and threading
- Provide graceful shutdown

```python
class DoorbellSecuritySystem:
    """Main system orchestrator."""
    
    def __init__(self, config: Dict[str, Any]):
        # Initialize all components
        self.face_manager = FaceManager(config)
        self.camera_handler = CameraHandler.create()
        self.gpio_handler = GPIOHandler()
        self.telegram_notifier = TelegramNotifier(config)
        
        # Event management
        self.last_trigger_time = 0
        self.debounce_time = 5.0
        self.processing_lock = threading.Lock()
    
    def on_doorbell_pressed(self, channel: int) -> None:
        """Handle doorbell press events with debouncing."""
        
    def _process_visitor(self) -> None:
        """Process visitor detection workflow."""
```

#### 2. Face Manager (`src/face_manager.py`)
**Role**: Face recognition and database management
**Responsibilities**:
- Face detection and recognition
- Known face database management
- Blacklist functionality
- Face encoding caching
- Privacy-preserving storage

```python
class FaceManager:
    """Face recognition and database management."""
    
    def __init__(self, config: Dict[str, Any]):
        self.tolerance = config.get("tolerance", 0.6)
        self.model = config.get("model", "hog")
        self.known_faces: Dict[str, np.ndarray] = {}
        self.blacklist_faces: Dict[str, np.ndarray] = {}
        self._encoding_cache: Dict[str, List[np.ndarray]] = {}
    
    def identify_face(self, image: np.ndarray) -> Dict[str, Any]:
        """Identify face in image."""
    
    def add_known_face(self, name: str, encoding: np.ndarray) -> None:
        """Add face to known database."""
```

#### 3. Camera Handler (`src/camera_handler.py`)
**Role**: Cross-platform camera abstraction
**Responsibilities**:
- Platform detection and camera selection
- Image capture and processing
- Hardware resource management
- Mock implementation for testing

```python
class CameraHandler:
    """Abstract camera interface."""
    
    @staticmethod
    def create() -> 'CameraHandler':
        """Factory method for platform-specific camera."""
        if platform_detector.is_raspberry_pi():
            return PiCameraHandler()
        return OpenCVCameraHandler()
    
    @abstractmethod
    def capture_image(self) -> np.ndarray:
        """Capture image from camera."""
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if camera is available."""
```

#### 4. GPIO Handler (`src/gpio_handler.py`)
**Role**: Hardware interface management
**Responsibilities**:
- GPIO pin management
- Event callback registration
- LED control
- Mock implementation for non-Pi systems

```python
class GPIOHandler:
    """GPIO interface for hardware control."""
    
    def __init__(self):
        self.is_pi = platform_detector.is_raspberry_pi()
        if self.is_pi:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
    
    def setup_doorbell_pin(self, pin: int, callback: callable) -> None:
        """Setup doorbell pin with callback."""
    
    def setup_led_pin(self, pin: int) -> None:
        """Setup LED control pin."""
```

#### 5. Telegram Notifier (`src/telegram_notifier.py`)
**Role**: External notification management
**Responsibilities**:
- Telegram bot integration
- Message and photo sending
- Error handling and retries
- Rate limiting

```python
class TelegramNotifier:
    """Telegram notification service."""
    
    def __init__(self, config: Dict[str, Any]):
        self.bot_token = config.get("bot_token")
        self.chat_id = config.get("chat_id")
        self.enabled = config.get("enabled", False)
    
    def send_message(self, message: str) -> bool:
        """Send text message."""
    
    def send_photo(self, image: np.ndarray, caption: str) -> bool:
        """Send photo with caption."""
```

#### 6. Web Interface (`src/web_interface.py`)
**Role**: Web dashboard and API
**Responsibilities**:
- Flask application setup
- Dashboard UI
- REST API endpoints
- Authentication and security
- Real-time updates

```python
def create_app(config: Dict[str, Any]) -> Flask:
    """Create Flask application."""
    app = Flask(__name__)
    
    # Configure security
    app.config['SECRET_KEY'] = config.get('secret_key')
    
    # Register routes
    @app.route('/')
    def dashboard():
        """Main dashboard."""
    
    @app.route('/api/faces')
    def api_faces():
        """API endpoint for face management."""
```

## 🔄 Data Flow Architecture

### Event Processing Flow

```
1. GPIO Event Trigger
   ├─ Doorbell press detected
   ├─ Debouncing applied
   └─ Event queued for processing

2. Image Capture
   ├─ Camera handler invoked
   ├─ Image captured and validated
   └─ Image passed to face manager

3. Face Recognition
   ├─ Face detection performed
   ├─ Face encodings extracted
   ├─ Database comparison
   └─ Recognition result generated

4. Decision Processing
   ├─ Result classification (known/unknown/blacklist)
   ├─ Alert level determination
   └─ Response action selection

5. Notification Dispatch
   ├─ Message composition
   ├─ Telegram notification sent
   └─ LED status update

6. Data Persistence
   ├─ Event logging
   ├─ Image storage (if configured)
   └─ Statistics update
```

### Data Storage Architecture

```
data/
├── known_faces/           # Known person encodings
│   ├── person1.pkl       # Encrypted face encoding
│   └── person2.pkl
├── blacklist_faces/       # Blacklisted person encodings
│   └── unwanted.pkl
├── captures/              # Temporary image captures
│   └── capture_YYYYMMDD_HHMMSS.jpg
├── cropped_faces/         # Processed face crops
│   ├── known/            # Crops of known faces
│   └── unknown/          # Crops of unknown faces
└── logs/                  # Application logs
    ├── doorbell.log      # Main application log
    ├── security.log      # Security events
    └── performance.log   # Performance metrics
```

## 🔧 Technology Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Language** | Python 3.11+ | Main development language |
| **Face Recognition** | face_recognition | Face detection and encoding |
| **Computer Vision** | OpenCV | Image processing |
| **Web Framework** | Flask | Web interface and API |
| **Hardware Interface** | RPi.GPIO, PiCamera2 | Raspberry Pi integration |
| **Messaging** | python-telegram-bot | Telegram notifications |
| **Concurrency** | Threading | Asynchronous processing |
| **Data Storage** | Pickle, JSON | Local data persistence |
| **Configuration** | YAML/JSON | Configuration management |

### Infrastructure Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Containerization** | Docker | Application packaging |
| **Orchestration** | Docker Compose | Multi-service deployment |
| **Process Management** | systemd | Service management |
| **Reverse Proxy** | nginx | Web server and SSL |
| **Monitoring** | Prometheus, Grafana | System monitoring |
| **CI/CD** | GitHub Actions | Automated testing and deployment |

## 🔒 Security Architecture

### Security Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Security                     │
├─────────────────────────────────────────────────────────────┤
│ • Input validation and sanitization                        │
│ • Authentication and authorization                         │
│ • CSRF protection and secure headers                       │
│ • Error handling without information disclosure            │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Data Security                          │
├─────────────────────────────────────────────────────────────┤
│ • Local-only processing (no cloud)                         │
│ • Encrypted face encoding storage                          │
│ • Secure file permissions                                  │
│ • Automatic cleanup of temporary data                      │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Network Security                         │
├─────────────────────────────────────────────────────────────┤
│ • TLS/SSL for web interface                                 │
│ • Secure Telegram API communication                        │
│ • Firewall-friendly design                                 │
│ • No unnecessary external dependencies                     │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    System Security                          │
├─────────────────────────────────────────────────────────────┤
│ • Minimal privilege execution                               │
│ • Resource limit enforcement                                │
│ • Secure configuration defaults                            │
│ • Regular security updates                                 │
└─────────────────────────────────────────────────────────────┘
```

### Privacy Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Privacy by Design                         │
├─────────────────────────────────────────────────────────────┤
│ Local Processing Only                                       │
│ ├─ Face detection: Local OpenCV/dlib                       │
│ ├─ Face recognition: Local face_recognition library        │
│ ├─ Data storage: Local filesystem only                     │
│ └─ No cloud services or external APIs for biometrics       │
├─────────────────────────────────────────────────────────────┤
│ Data Minimization                                           │
│ ├─ Store only face encodings, not images                   │
│ ├─ Automatic cleanup of temporary captures                 │
│ ├─ Configurable data retention periods                     │
│ └─ Optional anonymization of logs                          │
├─────────────────────────────────────────────────────────────┤
│ User Control                                                │
│ ├─ Opt-in data collection                                   │
│ ├─ Easy deletion of face data                              │
│ ├─ Transparent data usage                                  │
│ └─ Export functionality for data portability               │
└─────────────────────────────────────────────────────────────┘
```

## ⚡ Performance Architecture

### Performance Optimization Strategies

#### 1. Face Recognition Optimization
```python
class PerformanceOptimizedFaceManager:
    def __init__(self):
        # Use HOG model for speed on CPU
        self.model = "hog"  # vs "cnn" for better accuracy but slower
        
        # Implement encoding cache
        self._encoding_cache = LRUCache(maxsize=100)
        
        # Pre-load known faces at startup
        self.load_known_faces()
    
    def identify_face(self, image: np.ndarray) -> Dict[str, Any]:
        # Resize image for faster processing
        small_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        
        # Use cached encodings when possible
        cache_key = self._get_image_hash(small_image)
        if cache_key in self._encoding_cache:
            return self._encoding_cache[cache_key]
```

#### 2. Memory Management
```python
class MemoryOptimizedSystem:
    def __init__(self):
        # Limit concurrent processing
        self.processing_semaphore = threading.Semaphore(2)
        
        # Implement cleanup routines
        self.cleanup_scheduler = BackgroundScheduler()
        self.cleanup_scheduler.add_job(
            self._cleanup_temp_files, 
            'interval', 
            minutes=10
        )
    
    def _cleanup_temp_files(self):
        """Regular cleanup of temporary files."""
        # Remove old captures
        # Clear encoding cache if memory usage high
        # Cleanup logs if size exceeds limit
```

#### 3. Threading Architecture
```python
class ThreadingOptimizedSystem:
    def __init__(self):
        # Use thread pool for processing
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Separate threads for different concerns
        self.camera_thread = None
        self.processing_thread = None
        self.notification_thread = None
    
    def on_doorbell_pressed(self, channel: int):
        # Submit to thread pool for processing
        future = self.executor.submit(self._process_visitor)
        # Don't block on result
```

## 🔄 Scalability Architecture

### Horizontal Scaling Considerations

While the system is primarily designed for single-device operation, it can be scaled:

#### 1. Multi-Device Deployment
```yaml
# docker-compose-multi.yml
version: '3.8'
services:
  doorbell-front:
    image: doorbell-system:latest
    environment:
      - DEVICE_LOCATION=front_door
      - MQTT_BROKER=mqtt.local
  
  doorbell-back:
    image: doorbell-system:latest
    environment:
      - DEVICE_LOCATION=back_door
      - MQTT_BROKER=mqtt.local
  
  central-coordinator:
    image: doorbell-coordinator:latest
    environment:
      - MQTT_BROKER=mqtt.local
```

#### 2. Event Streaming Architecture
```python
class ScalableEventSystem:
    def __init__(self, config):
        # MQTT for event distribution
        self.mqtt_client = mqtt.Client()
        
        # Event serialization
        self.event_serializer = JSONEventSerializer()
    
    def publish_detection_event(self, event_data):
        """Publish detection event to MQTT."""
        topic = f"doorbell/{self.device_id}/detection"
        payload = self.event_serializer.serialize(event_data)
        self.mqtt_client.publish(topic, payload)
```

## 🧪 Testing Architecture

### Testing Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                      Test Pyramid                           │
├─────────────────────────────────────────────────────────────┤
│                    E2E Tests                               │
│                   ┌─────────┐                              │
│                   │ 5-10%   │                              │
│                   └─────────┘                              │
│                Integration Tests                            │
│              ┌─────────────────────┐                       │
│              │      15-25%         │                       │
│              └─────────────────────┘                       │
│                   Unit Tests                               │
│        ┌─────────────────────────────────────┐             │
│        │              65-80%                 │             │
│        └─────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

### Test Isolation Architecture

```python
class TestArchitecture:
    """Testing architecture patterns."""
    
    # Dependency Injection for testability
    def __init__(self, camera_handler=None, gpio_handler=None):
        self.camera_handler = camera_handler or CameraHandler.create()
        self.gpio_handler = gpio_handler or GPIOHandler()
    
    # Interface segregation
    class CameraInterface(ABC):
        @abstractmethod
        def capture_image(self) -> np.ndarray: pass
    
    # Mock implementations
    class MockCameraHandler(CameraInterface):
        def capture_image(self) -> np.ndarray:
            return self._generate_test_image()
```

## 📊 Monitoring and Observability Architecture

### Metrics Collection

```python
class MetricsCollector:
    """System metrics collection."""
    
    def __init__(self):
        # Prometheus metrics
        self.recognition_duration = Histogram(
            'face_recognition_duration_seconds',
            'Time spent on face recognition'
        )
        
        self.detection_counter = Counter(
            'face_detections_total',
            'Total face detections',
            ['status']  # known, unknown, blacklisted
        )
    
    def record_recognition(self, duration: float, status: str):
        self.recognition_duration.observe(duration)
        self.detection_counter.labels(status=status).inc()
```

### Health Monitoring

```python
class HealthMonitor:
    """System health monitoring."""
    
    def check_system_health(self) -> Dict[str, Any]:
        return {
            "camera": self._check_camera_health(),
            "storage": self._check_storage_health(),
            "memory": self._check_memory_usage(),
            "face_database": self._check_face_db_health()
        }
    
    def _check_camera_health(self) -> Dict[str, Any]:
        try:
            test_image = self.camera_handler.capture_image()
            return {"status": "healthy", "resolution": test_image.shape}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
```

## 🔄 Configuration Architecture

### Configuration Management

```python
class ConfigurationManager:
    """Centralized configuration management."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()
        self.watchers = []
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with validation."""
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        
        # Validate configuration
        self._validate_config(config)
        return config
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration against schema."""
        # JSON Schema validation
        # Type checking
        # Range validation
```

### Environment-Specific Configuration

```yaml
# config/environments/production.yml
face_recognition:
  tolerance: 0.6
  model: "hog"
  cache_size: 100

camera:
  width: 1280
  height: 720
  fps: 30

security:
  encryption_enabled: true
  auth_required: true
  rate_limiting: true

# config/environments/development.yml
face_recognition:
  tolerance: 0.8  # More lenient for testing
  model: "hog"
  cache_size: 10

camera:
  width: 640
  height: 480
  fps: 15

security:
  encryption_enabled: false
  auth_required: false
  rate_limiting: false
```

## 🚀 Deployment Architecture

### Container Architecture

```dockerfile
# Multi-stage build for optimization
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app

# Copy dependencies from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY app.py .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:5000/health')"

CMD ["python", "app.py"]
```

### Service Architecture

```yaml
# docker-compose-production.yml
version: '3.8'
services:
  doorbell-app:
    build: .
    restart: unless-stopped
    volumes:
      - ./data:/app/data
      - ./config:/app/config:ro
    environment:
      - ENVIRONMENT=production
    networks:
      - doorbell-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
  
  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    networks:
      - doorbell-network
    depends_on:
      - doorbell-app

networks:
  doorbell-network:
    driver: bridge
```

---

This architecture documentation provides a comprehensive overview of the Doorbell Security System's design, enabling developers to understand the system structure, make informed decisions about modifications, and maintain the codebase effectively.