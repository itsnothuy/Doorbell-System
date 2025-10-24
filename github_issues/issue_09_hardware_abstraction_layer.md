# Issue #9: Hardware Abstraction Layer Implementation

## 📋 **Overview**

Implement a comprehensive hardware abstraction layer (HAL) that provides cross-platform compatibility and seamless hardware integration for the Frigate-inspired doorbell security system. This issue involves migrating existing hardware handlers into a modular architecture with platform detection, mock implementations for testing, and optimized drivers for production deployment.

## 🎯 **Objectives**

### **Primary Goals**
1. **Cross-Platform Compatibility**: Support macOS, Linux, Windows, and Raspberry Pi
2. **Hardware Abstraction**: Unified interface for cameras, GPIO, and sensors
3. **Mock Testing Support**: Complete mock implementations for development
4. **Performance Optimization**: Hardware-specific optimizations and drivers
5. **Hot-Swapping**: Runtime hardware detection and switching

### **Success Criteria**
- Seamless operation across all target platforms
- Zero code changes required when switching hardware
- Complete mock coverage for testing and development
- Performance optimizations for Raspberry Pi hardware
- Runtime hardware detection and graceful fallbacks

## 🏗️ **Architecture Requirements**

### **Abstraction Layers**
```
Application Layer
    ↓
Hardware Abstraction Layer (HAL)
    ↓
Platform-Specific Drivers
    ↓
Hardware/Mock Implementations
```

### **Component Integration**
- **Frame Capture Worker**: Uses camera abstraction
- **GPIO Handler**: Uses GPIO abstraction  
- **Platform Detector**: Provides runtime platform information
- **Configuration System**: Platform-specific configurations
- **Testing Framework**: Mock hardware implementations

## 📁 **Implementation Specifications**

### **File Structure**
```
src/hardware/                           # Hardware abstraction layer
    __init__.py
    base_hardware.py                     # Abstract hardware interfaces
    camera_handler.py                    # Migrated and enhanced camera handler
    gpio_handler.py                      # Migrated and enhanced GPIO handler
    sensor_handler.py                    # Additional sensor interfaces
    platform/                           # Platform-specific implementations
        __init__.py
        raspberry_pi.py                  # Raspberry Pi specific drivers
        macos.py                         # macOS specific implementations
        linux.py                         # Generic Linux implementations
        windows.py                       # Windows implementations
    mock/                               # Mock implementations for testing
        __init__.py
        mock_camera.py                   # Mock camera for testing
        mock_gpio.py                     # Mock GPIO for testing
        mock_sensors.py                  # Mock sensors for testing
config/hardware_config.py               # Hardware configuration management
tests/test_hardware_abstraction.py      # Hardware abstraction tests
tests/test_platform_detection.py        # Platform detection tests
```

### **Core Component: Hardware Abstraction Layer**
```python
class HardwareAbstractionLayer:
    """Central hardware abstraction layer coordinator."""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.platform_info = platform_detector.get_platform_info()
        
        # Hardware components
        self.camera_handler = None
        self.gpio_handler = None
        self.sensor_handler = None
        
        # Hardware state
        self.hardware_available = {}
        self.mock_mode = config.mock_mode
        
        # Initialize hardware
        self._initialize_hardware()
        
    def _initialize_hardware(self) -> None:
        """Initialize all hardware components based on platform."""
        
    def get_camera_handler(self) -> CameraHandler:
        """Get appropriate camera handler for current platform."""
        
    def get_gpio_handler(self) -> GPIOHandler:
        """Get appropriate GPIO handler for current platform."""
        
    def get_sensor_handler(self) -> SensorHandler:
        """Get appropriate sensor handler for current platform."""
        
    def detect_hardware_changes(self) -> List[HardwareChange]:
        """Detect hardware changes at runtime."""
        
    def switch_to_mock_mode(self) -> None:
        """Switch all hardware to mock implementations."""
        
    def health_check(self) -> HardwareHealthStatus:
        """Perform comprehensive hardware health check."""
```

### **Abstract Hardware Interfaces**
```python
class CameraHandler(ABC):
    """Abstract camera handler interface."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize camera hardware."""
        
    @abstractmethod
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture single frame from camera."""
        
    @abstractmethod
    def start_stream(self) -> bool:
        """Start continuous camera stream."""
        
    @abstractmethod
    def stop_stream(self) -> None:
        """Stop camera stream."""
        
    @abstractmethod
    def get_camera_info(self) -> CameraInfo:
        """Get camera hardware information."""
        
    @abstractmethod
    def set_camera_settings(self, settings: CameraSettings) -> bool:
        """Configure camera settings."""
        
    @abstractmethod
    def is_available(self) -> bool:
        """Check if camera is available."""
        
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup camera resources."""

class GPIOHandler(ABC):
    """Abstract GPIO handler interface."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize GPIO hardware."""
        
    @abstractmethod
    def setup_pin(self, pin: int, mode: GPIOMode, **kwargs) -> bool:
        """Setup GPIO pin with specified mode."""
        
    @abstractmethod
    def read_pin(self, pin: int) -> Optional[bool]:
        """Read digital value from GPIO pin."""
        
    @abstractmethod
    def write_pin(self, pin: int, value: bool) -> bool:
        """Write digital value to GPIO pin."""
        
    @abstractmethod
    def setup_interrupt(self, pin: int, callback: Callable, edge: GPIOEdge) -> bool:
        """Setup interrupt handler for GPIO pin."""
        
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup GPIO resources."""

class SensorHandler(ABC):
    """Abstract sensor handler interface."""
    
    @abstractmethod
    def read_temperature(self) -> Optional[float]:
        """Read temperature from sensor."""
        
    @abstractmethod
    def read_humidity(self) -> Optional[float]:
        """Read humidity from sensor."""
        
    @abstractmethod
    def read_motion_sensor(self) -> Optional[bool]:
        """Read motion sensor state."""
        
    @abstractmethod
    def get_sensor_status(self) -> Dict[str, Any]:
        """Get status of all sensors."""
```

### **Platform-Specific Implementations**
```python
class RaspberryPiCameraHandler(CameraHandler):
    """Raspberry Pi specific camera implementation."""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.camera = None
        self.stream_active = False
        
        # Pi-specific optimizations
        self.use_gpu_memory = config.use_gpu_memory
        self.camera_module_version = config.camera_module_version
        
    def initialize(self) -> bool:
        """Initialize Pi camera with optimizations."""
        try:
            from picamera2 import Picamera2
            
            self.camera = Picamera2()
            
            # Configure camera for performance
            camera_config = self.camera.create_still_configuration(
                main={"size": self.config.resolution},
                lores={"size": self.config.preview_resolution},
                display="lores"
            )
            
            self.camera.configure(camera_config)
            self.camera.start()
            
            logger.info("Raspberry Pi camera initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pi camera initialization failed: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Optimized frame capture for Pi."""
        try:
            if not self.camera:
                return None
            
            # Capture frame using GPU acceleration if available
            frame = self.camera.capture_array()
            
            # Apply Pi-specific optimizations
            if self.use_gpu_memory:
                frame = self._optimize_with_gpu(frame)
            
            return frame
            
        except Exception as e:
            logger.error(f"Pi camera frame capture failed: {e}")
            return None

class MacOSCameraHandler(CameraHandler):
    """macOS specific camera implementation."""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.capture = None
        
    def initialize(self) -> bool:
        """Initialize macOS camera using OpenCV."""
        try:
            import cv2
            
            self.capture = cv2.VideoCapture(self.config.camera_index)
            
            # Configure camera settings
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
            self.capture.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # Verify camera is working
            ret, _ = self.capture.read()
            if not ret:
                raise Exception("Unable to read from camera")
            
            logger.info("macOS camera initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"macOS camera initialization failed: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture frame using OpenCV on macOS."""
        try:
            if not self.capture:
                return None
            
            ret, frame = self.capture.read()
            return frame if ret else None
            
        except Exception as e:
            logger.error(f"macOS camera frame capture failed: {e}")
            return None

class RaspberryPiGPIOHandler(GPIOHandler):
    """Raspberry Pi specific GPIO implementation."""
    
    def __init__(self, config: GPIOConfig):
        self.config = config
        self.gpio = None
        self.setup_pins = set()
        
    def initialize(self) -> bool:
        """Initialize Raspberry Pi GPIO."""
        try:
            import RPi.GPIO as GPIO
            self.gpio = GPIO
            
            # Set GPIO mode
            self.gpio.setmode(self.config.gpio_mode)
            self.gpio.setwarnings(False)
            
            logger.info("Raspberry Pi GPIO initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pi GPIO initialization failed: {e}")
            return False
    
    def setup_pin(self, pin: int, mode: GPIOMode, **kwargs) -> bool:
        """Setup GPIO pin on Raspberry Pi."""
        try:
            if not self.gpio:
                return False
            
            gpio_mode = self.gpio.IN if mode == GPIOMode.INPUT else self.gpio.OUT
            
            # Setup pin with pull-up/down if specified
            if mode == GPIOMode.INPUT:
                pull_up_down = kwargs.get('pull_up_down', self.gpio.PUD_OFF)
                self.gpio.setup(pin, gpio_mode, pull_up_down=pull_up_down)
            else:
                initial = kwargs.get('initial', self.gpio.LOW)
                self.gpio.setup(pin, gpio_mode, initial=initial)
            
            self.setup_pins.add(pin)
            return True
            
        except Exception as e:
            logger.error(f"Pi GPIO pin setup failed: {e}")
            return False
```

### **Mock Implementations**
```python
class MockCameraHandler(CameraHandler):
    """Mock camera implementation for testing."""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.is_initialized = False
        self.stream_active = False
        
        # Mock data generation
        self.frame_counter = 0
        self.test_images = self._load_test_images()
        
    def initialize(self) -> bool:
        """Mock camera initialization."""
        logger.info("Mock camera initialized")
        self.is_initialized = True
        return True
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Generate mock frame data."""
        if not self.is_initialized:
            return None
        
        # Cycle through test images or generate synthetic frames
        if self.test_images:
            frame = self.test_images[self.frame_counter % len(self.test_images)]
        else:
            frame = self._generate_synthetic_frame()
        
        self.frame_counter += 1
        return frame
    
    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generate synthetic frame with face-like patterns."""
        height, width = self.config.resolution
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Add face-like pattern occasionally
        if self.frame_counter % 30 == 0:  # Every 30 frames
            self._add_synthetic_face(frame)
        
        return frame
    
    def _load_test_images(self) -> List[np.ndarray]:
        """Load test images for mock frames."""
        test_images = []
        test_image_dir = Path("tests/data/test_images")
        
        if test_image_dir.exists():
            for image_path in test_image_dir.glob("*.jpg"):
                try:
                    image = cv2.imread(str(image_path))
                    if image is not None:
                        test_images.append(image)
                except Exception as e:
                    logger.warning(f"Failed to load test image {image_path}: {e}")
        
        return test_images

class MockGPIOHandler(GPIOHandler):
    """Mock GPIO implementation for testing."""
    
    def __init__(self, config: GPIOConfig):
        self.config = config
        self.pin_states = {}
        self.pin_modes = {}
        self.interrupt_callbacks = {}
        self.is_initialized = False
        
        # Simulate doorbell button press
        self._simulate_doorbell_events()
    
    def initialize(self) -> bool:
        """Mock GPIO initialization."""
        logger.info("Mock GPIO initialized")
        self.is_initialized = True
        return True
    
    def setup_pin(self, pin: int, mode: GPIOMode, **kwargs) -> bool:
        """Mock GPIO pin setup."""
        if not self.is_initialized:
            return False
        
        self.pin_modes[pin] = mode
        if mode == GPIOMode.INPUT:
            self.pin_states[pin] = False
        else:
            self.pin_states[pin] = kwargs.get('initial', False)
        
        logger.debug(f"Mock GPIO pin {pin} setup as {mode}")
        return True
    
    def read_pin(self, pin: int) -> Optional[bool]:
        """Mock GPIO pin read."""
        return self.pin_states.get(pin)
    
    def write_pin(self, pin: int, value: bool) -> bool:
        """Mock GPIO pin write."""
        if pin in self.pin_states:
            self.pin_states[pin] = value
            return True
        return False
    
    def setup_interrupt(self, pin: int, callback: Callable, edge: GPIOEdge) -> bool:
        """Mock GPIO interrupt setup."""
        if pin not in self.pin_modes:
            return False
        
        self.interrupt_callbacks[pin] = (callback, edge)
        logger.debug(f"Mock GPIO interrupt setup for pin {pin}")
        return True
    
    def _simulate_doorbell_events(self) -> None:
        """Simulate periodic doorbell button presses."""
        def doorbell_simulator():
            while True:
                time.sleep(random.randint(30, 120))  # Random interval
                
                # Simulate doorbell press
                doorbell_pin = self.config.doorbell_pin
                if doorbell_pin in self.interrupt_callbacks:
                    callback, edge = self.interrupt_callbacks[doorbell_pin]
                    try:
                        callback(doorbell_pin)
                        logger.info("Mock doorbell event triggered")
                    except Exception as e:
                        logger.error(f"Mock doorbell callback failed: {e}")
        
        # Start simulator in background thread
        import threading
        simulator_thread = threading.Thread(target=doorbell_simulator, daemon=True)
        simulator_thread.start()
```

## 🔄 **Migration Specifications**

### **Existing Code Migration**
```python
# Migration from src/camera_handler.py to src/hardware/camera_handler.py
class CameraHandlerMigration:
    """Handle migration of existing camera handler."""
    
    @staticmethod
    def migrate_existing_code() -> None:
        """Migrate existing camera handler to new architecture."""
        
        # 1. Preserve existing functionality
        # 2. Add hardware abstraction layer
        # 3. Implement platform detection
        # 4. Add mock support
        # 5. Update configuration system
        
    @staticmethod
    def create_compatibility_layer() -> None:
        """Create compatibility layer for existing code."""
        
        # Provide backward compatibility wrapper
        # Allow gradual migration of dependent code
        # Deprecation warnings for old interfaces

# Migration from src/gpio_handler.py to src/hardware/gpio_handler.py  
class GPIOHandlerMigration:
    """Handle migration of existing GPIO handler."""
    
    @staticmethod
    def migrate_platform_specific_code() -> None:
        """Migrate platform-specific GPIO code."""
        
        # Extract platform-specific logic
        # Create abstract interface
        # Implement platform handlers
        # Add mock implementations
```

### **Configuration Migration**
```python
@dataclass
class HardwareConfig:
    """Comprehensive hardware configuration."""
    
    # Platform configuration
    platform_override: Optional[str] = None
    mock_mode: bool = False
    auto_detect_hardware: bool = True
    
    # Camera configuration
    camera_config: CameraConfig = field(default_factory=CameraConfig)
    
    # GPIO configuration  
    gpio_config: GPIOConfig = field(default_factory=GPIOConfig)
    
    # Sensor configuration
    sensor_config: SensorConfig = field(default_factory=SensorConfig)
    
    # Performance configuration
    performance_config: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    @classmethod
    def from_legacy_config(cls, legacy_config: Dict[str, Any]) -> 'HardwareConfig':
        """Create new config from legacy configuration."""
        
    def to_platform_specific_config(self, platform: str) -> Dict[str, Any]:
        """Generate platform-specific configuration."""
```

## 🧪 **Testing Requirements**

### **Unit Tests**
```python
class TestHardwareAbstraction:
    """Test hardware abstraction layer."""
    
    def test_platform_detection(self):
        """Test automatic platform detection."""
        
    def test_hardware_initialization(self):
        """Test hardware component initialization."""
        
    def test_mock_implementations(self):
        """Test mock hardware implementations."""
        
    def test_cross_platform_compatibility(self):
        """Test cross-platform compatibility."""
        
    def test_hardware_hot_swapping(self):
        """Test runtime hardware detection and switching."""
        
    def test_error_handling(self):
        """Test error handling and graceful degradation."""
        
    def test_configuration_migration(self):
        """Test migration from legacy configuration."""

class TestPlatformSpecific:
    """Test platform-specific implementations."""
    
    @pytest.mark.skipif(not platform_detector.is_raspberry_pi(), reason="Pi only")
    def test_raspberry_pi_hardware(self):
        """Test Raspberry Pi specific hardware."""
        
    @pytest.mark.skipif(not platform_detector.is_macos(), reason="macOS only")
    def test_macos_hardware(self):
        """Test macOS specific hardware."""
```

### **Integration Tests**
```python
def test_hardware_pipeline_integration():
    """Test hardware integration with pipeline."""
    
def test_mock_to_real_hardware_switching():
    """Test switching from mock to real hardware."""
    
def test_hardware_performance():
    """Test hardware performance characteristics."""
```

## 📊 **Performance Targets**

### **Camera Performance**
- **Frame Capture**: <50ms latency on Raspberry Pi
- **Stream Initialization**: <2 seconds
- **Memory Usage**: <50MB for camera operations
- **CPU Usage**: <20% for 720p@15fps

### **GPIO Performance**
- **Pin Read/Write**: <1ms latency
- **Interrupt Response**: <5ms
- **Setup Time**: <100ms for all pins
- **CPU Usage**: <1% for GPIO operations

### **Platform Compatibility**
- **Initialization Time**: <5 seconds on all platforms
- **Mock Mode**: Zero hardware dependencies
- **Cross-Platform**: Identical API across platforms
- **Fallback Performance**: Graceful degradation on hardware failure

## 🔧 **Configuration Example**

### **hardware_config.py**
```python
# Hardware Configuration
HARDWARE_CONFIG = {
    # Platform configuration
    "platform_override": None,  # Auto-detect
    "mock_mode": False,
    "auto_detect_hardware": True,
    
    # Camera configuration
    "camera_config": {
        "resolution": (640, 480),
        "fps": 15,
        "camera_index": 0,
        "use_gpu_memory": True,  # Pi only
        "camera_module_version": 2  # Pi only
    },
    
    # GPIO configuration
    "gpio_config": {
        "gpio_mode": "BCM",  # Pi only
        "doorbell_pin": 18,
        "status_led_pin": 24,
        "cleanup_on_exit": True
    },
    
    # Sensor configuration
    "sensor_config": {
        "temperature_sensor": {
            "enabled": False,
            "pin": 4
        },
        "motion_sensor": {
            "enabled": False,
            "pin": 17
        }
    },
    
    # Performance configuration
    "performance_config": {
        "camera_buffer_size": 3,
        "gpio_debounce_time": 0.2,
        "health_check_interval": 60.0
    }
}
```

## 📈 **Monitoring and Metrics**

### **Key Metrics to Track**
- Hardware initialization success rates
- Camera frame capture performance
- GPIO operation latency
- Platform detection accuracy
- Mock/real hardware switching performance

### **Health Checks**
- Hardware availability monitoring
- Performance benchmark validation
- Resource usage tracking
- Error rate monitoring

## 🎯 **Definition of Done**

### **Functional Requirements**
- [ ] Complete hardware abstraction layer with unified interfaces
- [ ] Platform-specific implementations for all target platforms
- [ ] Comprehensive mock implementations for testing
- [ ] Migration from existing hardware handlers
- [ ] Cross-platform configuration management

### **Non-Functional Requirements**
- [ ] Performance targets met on all platforms
- [ ] Zero hardware dependencies in mock mode
- [ ] Seamless runtime hardware detection and switching
- [ ] Memory usage optimized for edge devices
- [ ] Comprehensive error handling and graceful degradation

### **Documentation Requirements**
- [ ] Hardware abstraction layer architecture documentation
- [ ] Platform-specific implementation guides
- [ ] Migration guide from existing code
- [ ] Configuration reference for all platforms
- [ ] Testing guide for mock implementations

---

## 🔗 **Dependencies**

### **Previous Issues**
- **Issue #4**: Frame Capture Worker (uses camera abstraction)
- **Issue #1**: Core Communication Infrastructure (message bus)

### **Next Issues**
- **Issue #10**: Storage Layer (database abstraction)
- **Issue #12**: Pipeline Orchestrator (hardware coordination)

### **External Dependencies**
- **Raspberry Pi**: picamera2, RPi.GPIO
- **macOS/Linux/Windows**: OpenCV for camera
- **Testing**: Mock hardware implementations
- **Configuration**: Platform detection utilities

---

## 🤖 **For Coding Agents: Auto-Close Setup**

### **Branch Naming Convention**
When implementing this issue, create your branch using one of these patterns:
- `issue-9/hardware-abstraction-layer`
- `9-hardware-abstraction-layer` 
- `issue-9/implement-hardware-abstraction`

### **PR Creation**
The GitHub Action will automatically append `Closes #9` to your PR description when you follow the branch naming convention above. This ensures the issue closes automatically when your PR is merged to the default branch.

### **Manual Alternative**
If you prefer manual control, include one of these in your PR description:
```
Closes #9
Fixes #9
Resolves #9
```

---

**This issue establishes a comprehensive hardware abstraction layer that enables cross-platform compatibility, seamless testing, and optimized hardware performance for the Frigate-inspired doorbell security system.**