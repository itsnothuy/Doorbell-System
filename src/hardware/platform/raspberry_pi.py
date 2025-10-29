"""
Raspberry Pi specific hardware implementations

Provides optimized implementations for Raspberry Pi hardware including
Pi Camera, GPIO, and environmental sensors.
"""

import logging
import time
from typing import Optional, Dict, Any, Callable
import numpy as np

from src.hardware.base_hardware import (
    CameraHandler,
    CameraInfo,
    CameraSettings,
    GPIOHandler,
    GPIOMode,
    GPIOEdge,
    SensorHandler
)

logger = logging.getLogger(__name__)

# Try to import Raspberry Pi specific libraries
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    logger.warning("picamera2 not available")

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logger.warning("RPi.GPIO not available")


class RaspberryPiCameraHandler(CameraHandler):
    """Raspberry Pi specific camera implementation using picamera2."""
    
    def __init__(self, config: dict):
        """
        Initialize Raspberry Pi camera handler.
        
        Args:
            config: Configuration dictionary with camera settings
        """
        if not PICAMERA2_AVAILABLE:
            raise ImportError("picamera2 library not available")
        
        self.config = config
        self.camera = None
        self.stream_active = False
        self.is_initialized = False
        
        # Camera settings
        self.settings = CameraSettings(
            resolution=config.get('resolution', (1280, 720)),
            fps=config.get('fps', 15.0),
            brightness=config.get('brightness', 50.0),
            contrast=config.get('contrast', 50.0),
            rotation=config.get('rotation', 0)
        )
        
        # Pi-specific optimizations
        self.use_gpu_memory = config.get('use_gpu_memory', True)
        self.camera_module_version = config.get('camera_module_version', 2)
        
        logger.info("Raspberry Pi camera handler created")
    
    def initialize(self) -> bool:
        """Initialize Pi camera with optimizations."""
        try:
            logger.info("Initializing Raspberry Pi camera...")
            
            self.camera = Picamera2()
            
            # Configure camera for performance
            camera_config = self.camera.create_still_configuration(
                main={"size": self.settings.resolution},
                lores={"size": (640, 480)},  # Lower resolution for preview
                display="lores"
            )
            
            self.camera.configure(camera_config)
            
            # Set camera properties
            controls = {}
            if self.settings.brightness:
                controls["Brightness"] = self.settings.brightness / 100.0
            if self.settings.contrast:
                controls["Contrast"] = self.settings.contrast / 100.0
            
            if controls:
                self.camera.set_controls(controls)
            
            # Start camera
            self.camera.start()
            time.sleep(2)  # Camera warm-up time
            
            self.is_initialized = True
            logger.info("Raspberry Pi camera initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pi camera initialization failed: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Optimized frame capture for Pi."""
        if not self.is_initialized or not self.camera:
            return None
        
        try:
            # Capture frame using GPU acceleration if available
            frame = self.camera.capture_array()
            
            # Apply Pi-specific optimizations
            if self.use_gpu_memory and hasattr(self, '_optimize_with_gpu'):
                frame = self._optimize_with_gpu(frame)
            
            return frame
            
        except Exception as e:
            logger.error(f"Pi camera frame capture failed: {e}")
            return None
    
    def start_stream(self) -> bool:
        """Start Pi camera stream."""
        if not self.is_initialized:
            return False
        
        try:
            if not self.stream_active:
                self.stream_active = True
                logger.info("Pi camera stream started")
            return True
        except Exception as e:
            logger.error(f"Failed to start Pi camera stream: {e}")
            return False
    
    def stop_stream(self) -> None:
        """Stop Pi camera stream."""
        self.stream_active = False
        logger.info("Pi camera stream stopped")
    
    def get_camera_info(self) -> CameraInfo:
        """Get Pi camera information."""
        return CameraInfo(
            name="Raspberry Pi Camera",
            resolution=self.settings.resolution,
            fps=self.settings.fps,
            backend="picamera2",
            available=self.is_initialized
        )
    
    def set_camera_settings(self, settings: CameraSettings) -> bool:
        """Update Pi camera settings."""
        try:
            self.settings = settings
            
            if self.is_initialized and self.camera:
                # Update camera controls
                controls = {
                    "Brightness": settings.brightness / 100.0,
                    "Contrast": settings.contrast / 100.0
                }
                self.camera.set_controls(controls)
            
            logger.info(f"Pi camera settings updated: {settings}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update Pi camera settings: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if Pi camera is available."""
        return self.is_initialized and self.camera is not None
    
    def cleanup(self) -> None:
        """Cleanup Pi camera resources."""
        try:
            if self.camera:
                self.camera.stop()
                self.camera.close()
                self.camera = None
            
            self.is_initialized = False
            self.stream_active = False
            logger.info("Pi camera cleanup completed")
            
        except Exception as e:
            logger.error(f"Pi camera cleanup failed: {e}")


class RaspberryPiGPIOHandler(GPIOHandler):
    """Raspberry Pi specific GPIO implementation."""
    
    def __init__(self, config: dict):
        """
        Initialize Raspberry Pi GPIO handler.
        
        Args:
            config: Configuration dictionary with GPIO settings
        """
        if not GPIO_AVAILABLE:
            raise ImportError("RPi.GPIO library not available")
        
        self.config = config
        self.gpio = GPIO
        self.setup_pins = set()
        self.initialized = False
        
        # GPIO mode (BCM or BOARD)
        self.gpio_mode = config.get('gpio_mode', 'BCM')
        
        logger.info("Raspberry Pi GPIO handler created")
    
    def initialize(self) -> bool:
        """Initialize Raspberry Pi GPIO."""
        try:
            logger.info("Initializing Raspberry Pi GPIO...")
            
            # Set GPIO mode
            mode = self.gpio.BCM if self.gpio_mode == 'BCM' else self.gpio.BOARD
            self.gpio.setmode(mode)
            self.gpio.setwarnings(False)
            
            self.initialized = True
            logger.info("Raspberry Pi GPIO initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pi GPIO initialization failed: {e}")
            return False
    
    def setup_pin(self, pin: int, mode: GPIOMode, **kwargs) -> bool:
        """Setup GPIO pin on Raspberry Pi."""
        if not self.initialized:
            return False
        
        try:
            gpio_mode = self.gpio.IN if mode == GPIOMode.INPUT else self.gpio.OUT
            
            # Setup pin with pull-up/down if specified
            if mode == GPIOMode.INPUT:
                pull_option = kwargs.get('pull_up_down', 'PUD_OFF')
                pull_map = {
                    'PUD_UP': self.gpio.PUD_UP,
                    'PUD_DOWN': self.gpio.PUD_DOWN,
                    'PUD_OFF': self.gpio.PUD_OFF
                }
                pull_up_down = pull_map.get(pull_option, self.gpio.PUD_OFF)
                self.gpio.setup(pin, gpio_mode, pull_up_down=pull_up_down)
            else:
                initial = kwargs.get('initial', self.gpio.LOW)
                self.gpio.setup(pin, gpio_mode, initial=initial)
            
            self.setup_pins.add(pin)
            logger.debug(f"Pi GPIO pin {pin} setup as {mode.value}")
            return True
            
        except Exception as e:
            logger.error(f"Pi GPIO pin setup failed: {e}")
            return False
    
    def read_pin(self, pin: int) -> Optional[bool]:
        """Read digital value from GPIO pin."""
        if not self.initialized or pin not in self.setup_pins:
            return None
        
        try:
            value = self.gpio.input(pin)
            return value == self.gpio.HIGH
        except Exception as e:
            logger.error(f"Pi GPIO read failed: {e}")
            return None
    
    def write_pin(self, pin: int, value: bool) -> bool:
        """Write digital value to GPIO pin."""
        if not self.initialized or pin not in self.setup_pins:
            return False
        
        try:
            gpio_value = self.gpio.HIGH if value else self.gpio.LOW
            self.gpio.output(pin, gpio_value)
            return True
        except Exception as e:
            logger.error(f"Pi GPIO write failed: {e}")
            return False
    
    def setup_interrupt(self, pin: int, callback: Callable, edge: GPIOEdge) -> bool:
        """Setup interrupt handler for GPIO pin."""
        if not self.initialized or pin not in self.setup_pins:
            return False
        
        try:
            # Map edge type
            edge_map = {
                GPIOEdge.RISING: self.gpio.RISING,
                GPIOEdge.FALLING: self.gpio.FALLING,
                GPIOEdge.BOTH: self.gpio.BOTH
            }
            gpio_edge = edge_map.get(edge, self.gpio.FALLING)
            
            # Add event detection
            bouncetime = self.config.get('debounce_time', 200)
            self.gpio.add_event_detect(
                pin,
                gpio_edge,
                callback=callback,
                bouncetime=int(bouncetime)
            )
            
            logger.info(f"Pi GPIO interrupt setup for pin {pin} on {edge.value} edge")
            return True
            
        except Exception as e:
            logger.error(f"Pi GPIO interrupt setup failed: {e}")
            return False
    
    def cleanup(self) -> None:
        """Cleanup GPIO resources and reset pins."""
        try:
            if self.initialized:
                # Remove all event detection
                for pin in self.setup_pins:
                    try:
                        self.gpio.remove_event_detect(pin)
                    except:
                        pass
                
                # Cleanup all pins
                self.gpio.cleanup()
                
                self.initialized = False
                self.setup_pins.clear()
                logger.info("Pi GPIO cleanup completed")
                
        except Exception as e:
            logger.error(f"Pi GPIO cleanup failed: {e}")


class RaspberryPiSensorHandler(SensorHandler):
    """Raspberry Pi specific sensor implementation with comprehensive sensor support."""
    
    def __init__(self, config: dict):
        """
        Initialize Raspberry Pi sensor handler.
        
        Args:
            config: Configuration dictionary with sensor settings
        """
        self.config = config
        self._is_available = False
        self._sensor_manager = None
        self._initialized = False
        self.running = False
        
        # Import sensor components conditionally
        try:
            from src.hardware.sensors import (
                SensorManager, SensorConfig, SensorType
            )
            self._SensorManager = SensorManager
            self._SensorConfig = SensorConfig
            self._SensorType = SensorType
            self._sensor_system_available = True
        except ImportError:
            logger.warning("Sensor system not available, using fallback mode")
            self._sensor_system_available = False
        
        logger.info("Raspberry Pi sensor handler created")
    
    def initialize(self) -> bool:
        """Initialize sensor system."""
        if not self._sensor_system_available:
            logger.warning("Sensor system not available")
            return False
        
        try:
            # Initialize sensor manager
            self._sensor_manager = self._SensorManager(self.config)
            
            # Configure sensors from config
            sensors_config = self.config.get('sensors', {})
            
            # Add temperature/humidity sensor if configured
            if sensors_config.get('temperature_humidity', {}).get('enabled', False):
                dht_config = sensors_config['temperature_humidity']
                sensor_config = self._SensorConfig(
                    sensor_id='dht22_main',
                    sensor_type=self._SensorType.TEMPERATURE_HUMIDITY,
                    enabled=True,
                    pin=dht_config.get('pin', 4),
                    polling_interval=dht_config.get('polling_interval', 30.0)
                )
                self._sensor_manager.add_sensor(sensor_config)
            
            # Add motion sensor if configured
            if sensors_config.get('motion', {}).get('enabled', False):
                motion_config = sensors_config['motion']
                sensor_config = self._SensorConfig(
                    sensor_id='pir_main',
                    sensor_type=self._SensorType.MOTION_PIR,
                    enabled=True,
                    pin=motion_config.get('pin', 17),
                    polling_interval=motion_config.get('polling_interval', 1.0)
                )
                self._sensor_manager.add_sensor(sensor_config)
            
            # Add DS18B20 temperature sensor if configured
            if sensors_config.get('temperature_only', {}).get('enabled', False):
                temp_config = sensors_config['temperature_only']
                sensor_config = self._SensorConfig(
                    sensor_id='ds18b20_main',
                    sensor_type=self._SensorType.TEMPERATURE_ONLY,
                    enabled=True,
                    pin=temp_config.get('pin', 4),
                    polling_interval=temp_config.get('polling_interval', 30.0)
                )
                self._sensor_manager.add_sensor(sensor_config)
            
            # Add pressure sensor if configured
            if sensors_config.get('pressure', {}).get('enabled', False):
                pressure_config = sensors_config['pressure']
                sensor_config = self._SensorConfig(
                    sensor_id='bmp280_main',
                    sensor_type=self._SensorType.PRESSURE,
                    enabled=True,
                    i2c_address=pressure_config.get('i2c_address', 0x77),
                    polling_interval=pressure_config.get('polling_interval', 60.0)
                )
                self._sensor_manager.add_sensor(sensor_config)
            
            self._initialized = True
            self._is_available = True
            logger.info("Raspberry Pi sensor handler initialized with sensor system")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize sensor handler: {e}")
            self._is_available = False
            return False
    
    def read_temperature(self) -> Optional[float]:
        """Read temperature from sensor."""
        if not self._initialized or not self._sensor_manager:
            logger.warning("Pi temperature sensor not initialized")
            return None
        
        try:
            readings = self._sensor_manager.get_latest_readings()
            
            # Try DHT22 first
            if 'dht22_main' in readings and readings['dht22_main'].value:
                value = readings['dht22_main'].value
                if isinstance(value, dict) and 'temperature' in value:
                    return value['temperature']
            
            # Try DS18B20
            if 'ds18b20_main' in readings and readings['ds18b20_main'].value:
                return readings['ds18b20_main'].value
            
            # Try BMP280
            if 'bmp280_main' in readings and readings['bmp280_main'].value:
                value = readings['bmp280_main'].value
                if isinstance(value, dict) and 'temperature' in value:
                    return value['temperature']
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to read temperature: {e}")
            return None
    
    def read_humidity(self) -> Optional[float]:
        """Read humidity from sensor."""
        if not self._initialized or not self._sensor_manager:
            logger.warning("Pi humidity sensor not initialized")
            return None
        
        try:
            readings = self._sensor_manager.get_latest_readings()
            
            # Try DHT22
            if 'dht22_main' in readings and readings['dht22_main'].value:
                value = readings['dht22_main'].value
                if isinstance(value, dict) and 'humidity' in value:
                    return value['humidity']
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to read humidity: {e}")
            return None
    
    def read_motion_sensor(self) -> Optional[bool]:
        """Read motion sensor state."""
        if not self._initialized or not self._sensor_manager:
            logger.warning("Pi motion sensor not initialized")
            return None
        
        try:
            readings = self._sensor_manager.get_latest_readings()
            
            # Try PIR sensor
            if 'pir_main' in readings and readings['pir_main'].value is not None:
                return bool(readings['pir_main'].value)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to read motion sensor: {e}")
            return None
    
    def get_sensor_status(self) -> Dict[str, Any]:
        """Get status of all sensors."""
        if not self._initialized or not self._sensor_manager:
            return {
                'available': False,
                'temperature': {'enabled': False},
                'humidity': {'enabled': False},
                'motion': {'enabled': False}
            }
        
        try:
            stats = self._sensor_manager.get_sensor_statistics()
            readings = self._sensor_manager.get_latest_readings()
            
            # Build status from sensor statistics
            status = {
                'available': self._is_available,
                'temperature': {
                    'enabled': False,
                    'current': None,
                    'unit': '°C',
                    'source': None
                },
                'humidity': {
                    'enabled': False,
                    'current': None,
                    'unit': '%RH',
                    'source': None
                },
                'motion': {
                    'enabled': False,
                    'current': False,
                    'unit': 'boolean',
                    'source': None
                }
            }
            
            # Update with DHT22 data if available
            if 'dht22_main' in readings:
                value = readings['dht22_main'].value
                if isinstance(value, dict):
                    if 'temperature' in value:
                        status['temperature'].update({
                            'enabled': True,
                            'current': value['temperature'],
                            'source': 'dht22_main'
                        })
                    if 'humidity' in value:
                        status['humidity'].update({
                            'enabled': True,
                            'current': value['humidity'],
                            'source': 'dht22_main'
                        })
            
            # Update with DS18B20 data if available
            if 'ds18b20_main' in readings and not status['temperature']['enabled']:
                status['temperature'].update({
                    'enabled': True,
                    'current': readings['ds18b20_main'].value,
                    'source': 'ds18b20_main'
                })
            
            # Update with BMP280 data if available
            if 'bmp280_main' in readings:
                value = readings['bmp280_main'].value
                if isinstance(value, dict) and not status['temperature']['enabled']:
                    if 'temperature' in value:
                        status['temperature'].update({
                            'enabled': True,
                            'current': value['temperature'],
                            'source': 'bmp280_main'
                        })
            
            # Update with PIR data if available
            if 'pir_main' in readings:
                status['motion'].update({
                    'enabled': True,
                    'current': bool(readings['pir_main'].value),
                    'source': 'pir_main'
                })
            
            # Add detailed statistics
            status['statistics'] = stats
            status['timestamp'] = time.time()
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get sensor status: {e}")
            return {
                'available': False,
                'temperature': {'enabled': False},
                'humidity': {'enabled': False},
                'motion': {'enabled': False},
                'error': str(e)
            }
    
    def is_available(self) -> bool:
        """Check if sensors are available."""
        return self._is_available
    
    def cleanup(self) -> None:
        """Cleanup sensor resources."""
        if self._sensor_manager and self.running:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._sensor_manager.stop_monitoring())
                else:
                    asyncio.run(self._sensor_manager.stop_monitoring())
            except Exception as e:
                logger.error(f"Error stopping sensor monitoring: {e}")
        
        self._is_available = False
        self._initialized = False
        logger.info("Pi sensor cleanup completed")
    
    async def start_async_monitoring(self) -> None:
        """Start async sensor monitoring (for use in async contexts)."""
        if self._initialized and self._sensor_manager:
            await self._sensor_manager.start_monitoring()
            self.running = True
    
    async def stop_async_monitoring(self) -> None:
        """Stop async sensor monitoring (for use in async contexts)."""
        if self._sensor_manager:
            await self._sensor_manager.stop_monitoring()
            self.running = False


__all__ = [
    'RaspberryPiCameraHandler',
    'RaspberryPiGPIOHandler',
    'RaspberryPiSensorHandler',
]
