# GitHub Copilot Instructions

## 🤖 Project Context

You are working on the **Doorbell Security System**, an AI-powered privacy-first security solution that uses local face recognition to identify visitors at a doorbell. The system combines hardware integration (Raspberry Pi, camera, GPIO), computer vision (OpenCV, face_recognition), and smart notifications (Telegram bot) to create a comprehensive security monitoring solution.

## 🎯 Project Overview

### Core Functionality
- **Face Recognition**: Detect and identify visitors using local AI processing
- **Hardware Integration**: GPIO doorbell detection, camera capture, LED status indicators
- **Smart Notifications**: Real-time Telegram alerts with photos and visitor identification
- **Web Dashboard**: Flask-based interface for system management and monitoring
- **Privacy-First**: All biometric processing happens locally, no cloud dependencies

### Technical Stack
- **Language**: Python 3.11+
- **AI/ML**: face_recognition, OpenCV, dlib
- **Hardware**: RPi.GPIO, PiCamera2 (Raspberry Pi)
- **Web**: Flask, HTML/CSS/JavaScript
- **Messaging**: python-telegram-bot
- **Deployment**: Docker, Docker Compose
- **CI/CD**: GitHub Actions

## 📁 Codebase Structure

```
doorbell-system/
├── src/                          # Core application modules
│   ├── doorbell_security.py     # Main orchestrator and event handler
│   ├── face_manager.py          # Face recognition and database management
│   ├── camera_handler.py        # Cross-platform camera abstraction
│   ├── gpio_handler.py          # Hardware GPIO interface
│   ├── telegram_notifier.py     # Telegram bot integration
│   ├── web_interface.py         # Flask web application
│   └── platform_detector.py     # Platform detection utilities
├── config/                       # Configuration management
│   ├── settings.py              # Main configuration loader
│   ├── credentials_template.py  # Template for credentials
│   └── logging_config.py        # Logging configuration
├── data/                        # Data storage directories
│   ├── known_faces/             # Known person face encodings
│   ├── blacklist_faces/         # Blacklisted person encodings
│   ├── captures/                # Temporary image captures
│   └── logs/                    # Application logs
├── templates/                   # Flask HTML templates
├── static/                      # Web interface assets
├── tests/                       # Comprehensive test suite
├── docs/                        # Documentation
└── deployment files            # Docker, CI/CD, configuration
```

## 🔧 Key Components

### 1. Doorbell Security System (`src/doorbell_security.py`)
**Purpose**: Main orchestrator that coordinates all system components
**Key Features**:
- Event-driven architecture with GPIO callback handling
- Debouncing mechanism to prevent spam triggers
- Threading for non-blocking processing
- Graceful error handling and recovery

### 2. Face Manager (`src/face_manager.py`)
**Purpose**: Face recognition engine and database management
**Key Features**:
- Local face detection using OpenCV/dlib
- Face encoding comparison with tolerance settings
- Known faces and blacklist management
- Privacy-preserving encrypted storage
- Caching for performance optimization

### 3. Camera Handler (`src/camera_handler.py`)
**Purpose**: Cross-platform camera abstraction layer
**Key Features**:
- Factory pattern for platform-specific implementations
- PiCamera2 support for Raspberry Pi
- OpenCV fallback for other platforms
- Mock implementation for testing
- Automatic resource management

### 4. GPIO Handler (`src/gpio_handler.py`)
**Purpose**: Hardware interface for Raspberry Pi GPIO
**Key Features**:
- Doorbell button interrupt handling
- LED status indicator control
- Mock implementation for non-Pi systems
- Safe cleanup and resource management

### 5. Web Interface (`src/web_interface.py`)
**Purpose**: Flask-based dashboard and API
**Key Features**:
- Real-time system status monitoring
- Face database management interface
- Configuration management
- REST API for mobile/external integration
- Security features (CSRF protection, input validation)

## 🏗️ Architectural Patterns

### Design Patterns Used
- **Factory Pattern**: Platform-specific camera implementations
- **Observer Pattern**: GPIO event callbacks
- **Singleton Pattern**: Configuration management
- **Strategy Pattern**: Different face recognition models
- **Template Method**: Common interface patterns

### Key Principles
- **Modularity**: Loosely coupled components with clear interfaces
- **Cross-Platform**: Works on Pi, macOS, Linux, Docker
- **Privacy-First**: Local processing only, no cloud dependencies
- **Fail-Safe**: Graceful degradation when hardware unavailable
- **Testability**: Comprehensive mocking and test coverage

## 💡 Development Guidelines

### Code Style
- Follow PEP 8 with 100-character line limit
- Use type hints for all public functions
- Google-style docstrings for documentation
- Comprehensive error handling with specific exceptions
- Logging at appropriate levels (DEBUG, INFO, WARNING, ERROR)

### Security Best Practices
- Input validation and sanitization for all user inputs
- No hardcoded secrets or credentials
- Secure file permissions for sensitive data
- Rate limiting for API endpoints
- CSRF protection for web forms

### Performance Considerations
- Face recognition optimization for Pi hardware
- Memory management for long-running processes
- Caching strategies for repeated operations
- Threading for non-blocking operations
- Resource cleanup and lifecycle management

### Testing Requirements
- Unit tests for all core components
- Integration tests for component interactions
- Mock implementations for hardware dependencies
- Performance tests for face recognition
- Security tests for input validation

## 🎯 Common Tasks and Patterns

### Adding New Face Recognition Features
```python
class FaceManager:
    def new_recognition_feature(self, image: np.ndarray) -> Dict[str, Any]:
        """Template for new recognition features."""
        try:
            # Validate input
            if image is None or image.size == 0:
                raise ValueError("Invalid image")
            
            # Process image
            result = self._process_image(image)
            
            # Log operation
            logger.info(f"Recognition feature completed: {result['status']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Recognition feature failed: {e}")
            raise
```

### Adding New Hardware Interfaces
```python
class NewHardwareHandler:
    def __init__(self):
        self.is_available = platform_detector.has_hardware()
        
    def initialize(self) -> bool:
        """Initialize hardware with error handling."""
        try:
            if not self.is_available:
                logger.warning("Hardware not available, using mock")
                return False
            
            # Initialize hardware
            return True
            
        except Exception as e:
            logger.error(f"Hardware initialization failed: {e}")
            return False
```

### Adding New Web API Endpoints
```python
@app.route('/api/new_feature', methods=['POST'])
def api_new_feature():
    """Template for new API endpoints."""
    try:
        # Validate input
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Process request
        result = process_feature(data)
        
        # Return response
        return jsonify({"status": "success", "data": result})
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": "Internal server error"}), 500
```

## 🔍 Debugging and Troubleshooting

### Common Issues and Solutions

#### Face Recognition Issues
- **Low accuracy**: Adjust tolerance settings, improve lighting
- **Slow performance**: Use HOG model instead of CNN, optimize image size
- **Memory issues**: Implement face encoding caching, cleanup old data

#### Hardware Issues
- **Camera not working**: Check permissions, verify hardware availability
- **GPIO not responding**: Verify pin configuration, check for conflicts
- **Resource conflicts**: Implement proper cleanup and resource management

#### Network/Integration Issues
- **Telegram not working**: Verify bot token, check network connectivity
- **Web interface errors**: Check Flask configuration, verify permissions
- **Docker issues**: Review container configuration, check port bindings

### Logging and Monitoring
```python
# Enable debug logging for troubleshooting
import logging
logging.basicConfig(level=logging.DEBUG)

# Check system health
def check_system_health():
    return {
        "camera": camera_handler.is_available(),
        "gpio": gpio_handler.is_available(),
        "face_db": len(face_manager.known_faces),
        "telegram": telegram_notifier.test_connection()
    }
```

## 🧪 Testing Patterns

### Unit Test Template
```python
class TestComponentName:
    @pytest.fixture
    def component(self, test_config):
        return ComponentClass(test_config)
    
    def test_basic_functionality(self, component):
        # Arrange
        test_input = create_test_data()
        
        # Act
        result = component.method(test_input)
        
        # Assert
        assert result is not None
        assert result['status'] == 'expected'
```

### Integration Test Template
```python
def test_full_workflow(mock_camera, mock_gpio, mock_telegram):
    # Setup system with mocks
    system = DoorbellSecuritySystem(test_config)
    
    # Trigger event
    system.on_doorbell_pressed(channel=18)
    
    # Verify workflow
    assert mock_camera.capture_image.called
    assert mock_telegram.send_message.called
```

## 📝 Documentation Standards

### Code Documentation
- Use clear, descriptive function and variable names
- Add docstrings for all public functions and classes
- Include type hints for parameters and return values
- Comment complex logic and business rules

### Commit Messages
Use conventional commit format:
- `feat:` - New features
- `fix:` - Bug fixes  
- `docs:` - Documentation updates
- `test:` - Test additions/modifications
- `refactor:` - Code refactoring
- `perf:` - Performance improvements

### Pull Request Guidelines
- Clear description of changes and motivation
- Reference related issues
- Include test coverage for new features
- Update documentation as needed
- Verify CI passes before requesting review

## 🎯 Feature Development Workflow

### 1. Planning Phase
- Create GitHub issue with detailed requirements
- Review architectural implications
- Plan testing strategy
- Consider security implications

### 2. Implementation Phase
- Create feature branch from develop
- Implement core functionality
- Add comprehensive tests
- Update documentation

### 3. Integration Phase
- Test with existing components
- Verify cross-platform compatibility
- Run full test suite
- Performance testing if applicable

### 4. Review Phase
- Code review with security focus
- Test coverage verification
- Documentation review
- CI/CD pipeline validation

## 🔐 Security Considerations

### Data Protection
- All face encodings stored encrypted
- Temporary images automatically cleaned up
- No biometric data transmitted to external services
- Secure file permissions for sensitive data

### Input Validation
- Validate all user inputs (web forms, API calls)
- Sanitize file names and paths
- Check image format and size limits
- Prevent path traversal attacks

### Authentication & Authorization
- Implement authentication for web interface
- Use CSRF tokens for state-changing operations
- Rate limiting for API endpoints
- Secure session management

## 🚀 Performance Optimization

### Face Recognition Optimization
- Use HOG model for CPU efficiency
- Implement encoding caching
- Optimize image preprocessing
- Batch processing where possible

### Memory Management
- Regular cleanup of temporary files
- Limit concurrent processing
- Monitor memory usage
- Implement garbage collection triggers

### Resource Management
- Proper cleanup of camera resources
- GPIO cleanup on shutdown
- Thread pool management
- Connection pooling for external services

---

## 📋 Quick Reference

### Key Files to Understand
1. `src/doorbell_security.py` - Main system orchestrator
2. `src/face_manager.py` - Face recognition engine
3. `config/settings.py` - Configuration management
4. `tests/conftest.py` - Test configuration and fixtures

### Important Configuration
- Face recognition tolerance: 0.6 (adjustable)
- Debounce time: 5 seconds (prevents spam)
- Image retention: 7 days (configurable)
- Log rotation: 10MB files, 5 backups

### Development Commands
```bash
# Setup development environment
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest tests/ -v --cov=src

# Start development server
python app.py

# Run with Docker
docker-compose up --build
```

Remember: This is a privacy-first security system. Always consider privacy implications, maintain local processing, and ensure robust error handling for a reliable security solution.