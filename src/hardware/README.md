# Hardware Abstraction Layer (HAL)

## Overview

The Hardware Abstraction Layer provides a unified interface for all hardware components across different platforms including Raspberry Pi, macOS, Linux, and Windows. It enables seamless hardware integration, comprehensive testing with mock implementations, and platform-specific optimizations.

## Architecture

```
Application Layer
    ↓
Hardware Abstraction Layer (HAL)
    ↓
Platform-Specific Drivers
    ↓
Hardware/Mock Implementations
```

## Key Features

- **Cross-Platform Compatibility**: Supports Raspberry Pi, macOS, Linux, and Windows
- **Unified Interface**: Consistent API across all platforms
- **Mock Testing**: Complete mock implementations for development and testing
- **Hot-Swapping**: Runtime hardware detection and switching
- **Health Monitoring**: Comprehensive hardware health checks
- **Backward Compatible**: Works seamlessly with existing code

## Components

### Abstract Interfaces (`base_hardware.py`)

Defines the contracts for all hardware components:

- **CameraHandler**: Frame capture, streaming, configuration
- **GPIOHandler**: Pin control, interrupts, edge detection
- **SensorHandler**: Environmental sensors (temperature, humidity, motion)

### Platform-Specific Implementations (`platform/`)

Optimized implementations for each platform:

- **Raspberry Pi**: picamera2, RPi.GPIO, native sensor support
- **macOS**: OpenCV with webcam support
- **Linux**: V4L2-optimized camera support
- **Windows**: DirectShow-based camera support

### Mock Implementations (`mock/`)

Complete mock hardware for testing:

- **MockCameraHandler**: Generates synthetic frames with face-like patterns
- **MockGPIOHandler**: Simulates GPIO pins and doorbell events
- **MockSensorHandler**: Simulates environmental sensors

## Usage

### Basic Setup

```python
from src.hardware import get_hal
from config.hardware_config import HardwareConfig

# Create configuration
config = HardwareConfig(mock_mode=True)

# Get HAL instance
hal = get_hal(config.to_dict())

# Get hardware handlers
camera = hal.get_camera_handler()
gpio = hal.get_gpio_handler()
```

### Camera Operations

```python
# Initialize camera
camera.initialize()

# Capture frame
frame = camera.capture_frame()

# Get camera info
info = camera.get_camera_info()
print(f"Camera: {info.name}, Resolution: {info.resolution}")

# Cleanup
camera.cleanup()
```

### GPIO Operations

```python
from src.hardware.base_hardware import GPIOMode, GPIOEdge

# Initialize GPIO
gpio.initialize()

# Setup pins
gpio.setup_pin(18, GPIOMode.INPUT, pull_up_down='PUD_UP')
gpio.setup_pin(21, GPIOMode.OUTPUT, initial=False)

# Read and write
state = gpio.read_pin(18)
gpio.write_pin(21, True)

# Setup interrupt
def doorbell_callback(pin):
    print(f"Doorbell pressed on pin {pin}")

gpio.setup_interrupt(18, doorbell_callback, GPIOEdge.FALLING)

# Cleanup
gpio.cleanup()
```

### Health Monitoring

```python
# Perform health check
health = hal.health_check()

if health.healthy:
    print("All hardware is healthy")
else:
    print(f"Errors: {health.errors}")
    print(f"Warnings: {health.warnings}")

# Check specific components
if health.components['camera'] == HardwareStatus.AVAILABLE:
    print("Camera is available")
```

### Configuration

#### From Dictionary

```python
config = {
    'mock_mode': False,
    'camera_config': {
        'resolution': (1920, 1080),
        'fps': 30.0,
        'brightness': 60.0
    },
    'gpio_config': {
        'gpio_mode': 'BCM',
        'doorbell_pin': 18,
        'debounce_time': 300.0
    }
}

hal = HardwareAbstractionLayer(config)
```

#### From Environment Variables

```python
# Set environment variables
export CAMERA_RESOLUTION="1920,1080"
export CAMERA_FPS="30.0"
export GPIO_MODE="BCM"
export DOORBELL_PIN="18"

# Load from environment
config = HardwareConfig.from_environment()
hal = get_hal(config.to_dict())
```

#### From Legacy Settings

```python
from config.settings import Settings

# Migrate from legacy settings
settings = Settings()
config = HardwareConfig.from_legacy_config(settings)
hal = get_hal(config.to_dict())
```

### Platform-Specific Configuration

```python
config = HardwareConfig()

# Get Raspberry Pi specific config
pi_config = config.to_platform_specific_config('raspberry_pi')

# Get macOS specific config
macos_config = config.to_platform_specific_config('macos')

# Use platform-specific config
hal = HardwareAbstractionLayer(pi_config)
```

## Testing

### Unit Tests

Run unit tests for HAL components:

```bash
pytest tests/test_hardware_abstraction.py -v
```

### Integration Tests

Run integration tests with existing handlers:

```bash
pytest tests/test_hardware_integration.py -v
```

### Mock Hardware Testing

```python
# Use mock hardware in tests
config = {'mock_mode': True}
hal = HardwareAbstractionLayer(config)

camera = hal.get_camera_handler()
camera.initialize()

# Capture synthetic frames
for i in range(10):
    frame = camera.capture_frame()
    assert frame is not None
```

## Backward Compatibility

The HAL is designed to work seamlessly with existing code:

```python
# Existing camera handler code still works
from src.camera_handler import CameraHandler

camera = CameraHandler()
camera.initialize()
frame = camera.capture_image()  # Uses HAL underneath if available
```

## Performance Targets

### Camera Performance
- **Frame Capture**: <50ms latency on Raspberry Pi
- **Stream Initialization**: <2 seconds
- **Memory Usage**: <50MB for camera operations
- **CPU Usage**: <20% for 720p@15fps

### GPIO Performance
- **Pin Read/Write**: <1ms latency
- **Interrupt Response**: <5ms
- **Setup Time**: <100ms for all pins
- **CPU Usage**: <1% for GPIO operations

## Extending the HAL

### Adding New Hardware

1. Create a new handler class that implements the appropriate interface:

```python
from src.hardware.base_hardware import CameraHandler

class CustomCameraHandler(CameraHandler):
    def __init__(self, config):
        self.config = config
    
    def initialize(self) -> bool:
        # Custom initialization
        pass
    
    def capture_frame(self) -> Optional[np.ndarray]:
        # Custom frame capture
        pass
    
    # Implement other required methods...
```

2. Register the handler in the HAL:

```python
# In src/hardware/__init__.py
def _get_camera_handler(self, config):
    if self.config.get('use_custom_camera'):
        from src.hardware.custom import CustomCameraHandler
        return CustomCameraHandler(config)
    # ... existing logic
```

### Adding New Platforms

1. Create platform-specific implementation in `src/hardware/platform/`:

```python
# src/hardware/platform/custom_platform.py
class CustomPlatformCameraHandler(CameraHandler):
    # Platform-specific implementation
    pass
```

2. Update HAL to detect and use the new platform:

```python
def _get_camera_handler(self, config):
    if platform_detector.is_custom_platform:
        from src.hardware.platform.custom_platform import CustomPlatformCameraHandler
        return CustomPlatformCameraHandler(config)
```

## Troubleshooting

### Camera Not Detected

```python
# Check camera availability
hal = get_hal()
camera = hal.get_camera_handler()

if not camera.is_available():
    # Try switching to mock mode
    hal.switch_to_mock_mode()
    camera = hal.get_camera_handler()
    camera.initialize()
```

### GPIO Not Working

```python
# Verify GPIO handler initialization
gpio = hal.get_gpio_handler()

if not gpio.initialized:
    # Check platform detection
    print(f"Platform: {hal.platform_info}")
    
    # Try mock GPIO
    hal.switch_to_mock_mode()
    gpio = hal.get_gpio_handler()
```

### Health Check Failures

```python
# Detailed health check
health = hal.health_check()

for component, status in health.components.items():
    print(f"{component}: {status}")

if health.errors:
    for error in health.errors:
        print(f"Error: {error}")
```

## Migration Guide

### Migrating from Legacy Handlers

1. **Update imports** (optional - backward compatibility maintained):

```python
# Old
from src.camera_handler import CameraHandler

# New (explicit HAL usage)
from src.hardware import get_hal
camera = get_hal().get_camera_handler()
```

2. **Update configuration**:

```python
# Old
settings = Settings()

# New (optional)
from config.hardware_config import HardwareConfig
config = HardwareConfig.from_legacy_config(settings)
```

3. **No code changes required** - existing handlers automatically use HAL!

## License

This Hardware Abstraction Layer is part of the Doorbell Security System and is licensed under the MIT License.

## Contributing

Contributions are welcome! When adding new hardware support:

1. Implement the appropriate interface from `base_hardware.py`
2. Add comprehensive tests
3. Update this documentation
4. Ensure backward compatibility
5. Submit a pull request

## Support

For issues or questions:
- GitHub Issues: https://github.com/itsnothuy/Doorbell-System/issues
- Documentation: See project README.md
