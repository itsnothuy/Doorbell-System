# Issue #19: Raspberry Pi Sensor Integration Implementation

## Issue Summary

**Priority**: High  
**Type**: Hardware Integration, Platform Feature  
**Component**: Hardware Abstraction Layer, IoT Sensors  
**Estimated Effort**: 35-45 hours  
**Dependencies**: Hardware Platform Detection, GPIO Handler  

## Overview

Implement comprehensive sensor integration for Raspberry Pi deployment, including temperature, humidity, motion detection, and environmental monitoring. This enhances the doorbell system with IoT capabilities, providing environmental context for security events and enabling advanced automation features.

## Current State Analysis

### Existing Sensor Placeholders
```python
# Current incomplete implementation in src/hardware/platform/raspberry_pi.py

class RaspberryPiSensorHandler:
    def __init__(self, config: dict):
        self.config = config
        self._is_available = False
        
        # TODO: Initialize actual sensor hardware (DHT22, DS18B20, PIR, etc.)
        logger.info("Raspberry Pi sensor handler created")
    
    def read_temperature(self) -> Optional[float]:
        """Read temperature from sensor."""
        # TODO: Implement actual sensor reading
        logger.warning("Pi temperature sensor not implemented")
        return None
    
    def read_humidity(self) -> Optional[float]:
        """Read humidity from sensor."""
        # TODO: Implement actual sensor reading  
        logger.warning("Pi humidity sensor not implemented")
        return None
    
    def read_motion_sensor(self) -> Optional[bool]:
        """Read motion sensor state."""
        # TODO: Implement actual sensor reading
        logger.warning("Pi motion sensor not implemented")
        return None
```

### Missing Capabilities
- **Environmental Monitoring**: No temperature/humidity sensing
- **Motion Detection**: No PIR sensor integration  
- **Status Monitoring**: No system health sensors
- **Automation Features**: No sensor-based triggers
- **Data Logging**: No sensor data persistence

## Technical Specifications

### Comprehensive Sensor Integration Framework

#### Supported Sensor Types
```python
#!/usr/bin/env python3
"""
Production Raspberry Pi Sensor Integration System
"""

import asyncio
import logging
import threading
import time
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
from pathlib import Path
import statistics

# Conditional imports for Raspberry Pi specific libraries
try:
    import board
    import digitalio
    import busio
    import adafruit_dht
    import adafruit_ds18x20
    from adafruit_onewire.bus import OneWireBus
    import adafruit_bmp280
    import adafruit_tsl2591
    PI_LIBRARIES_AVAILABLE = True
except ImportError:
    PI_LIBRARIES_AVAILABLE = False
    logging.warning("Raspberry Pi sensor libraries not available - using mock implementations")

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

logger = logging.getLogger(__name__)


class SensorType(Enum):
    """Supported sensor types."""
    TEMPERATURE_HUMIDITY = "temperature_humidity"
    TEMPERATURE_ONLY = "temperature_only"
    MOTION_PIR = "motion_pir"
    PRESSURE = "pressure"
    LIGHT = "light"
    MAGNETIC_DOOR = "magnetic_door"
    VIBRATION = "vibration"
    SOUND_LEVEL = "sound_level"
    AIR_QUALITY = "air_quality"


class SensorStatus(Enum):
    """Sensor operational status."""
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    CALIBRATING = "calibrating"
    MAINTENANCE = "maintenance"


@dataclass
class SensorReading:
    """Individual sensor reading with metadata."""
    sensor_id: str
    sensor_type: SensorType
    timestamp: float
    value: Union[float, bool, Dict[str, float]]
    unit: str
    quality: float = 1.0  # Reading quality (0-1)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert reading to dictionary."""
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type.value,
            'timestamp': self.timestamp,
            'value': self.value,
            'unit': self.unit,
            'quality': self.quality,
            'error_message': self.error_message
        }


@dataclass
class SensorConfig:
    """Configuration for individual sensor."""
    sensor_id: str
    sensor_type: SensorType
    enabled: bool = True
    pin: Optional[int] = None
    i2c_address: Optional[int] = None
    polling_interval: float = 30.0  # seconds
    calibration_offset: float = 0.0
    threshold_high: Optional[float] = None
    threshold_low: Optional[float] = None
    smoothing_window: int = 5  # readings for moving average
    retry_count: int = 3
    timeout: float = 5.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSensor(ABC):
    """Abstract base class for all sensors."""
    
    def __init__(self, config: SensorConfig):
        self.config = config
        self.status = SensorStatus.OFFLINE
        self.last_reading = None
        self.error_count = 0
        self.total_readings = 0
        
        # Reading history for smoothing
        self.reading_history = []
        
        # Callbacks for events
        self.threshold_callbacks = []
        self.error_callbacks = []
        
        logger.info(f"Initialized sensor {config.sensor_id} ({config.sensor_type.value})")
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize sensor hardware."""
        pass
    
    @abstractmethod
    async def read_raw(self) -> Union[float, bool, Dict[str, float]]:
        """Read raw value from sensor."""
        pass
    
    async def read(self) -> Optional[SensorReading]:
        """Read sensor with error handling and processing."""
        try:
            if self.status == SensorStatus.OFFLINE:
                if not await self.initialize():
                    return None
            
            # Read raw value with retries
            raw_value = None
            for attempt in range(self.config.retry_count):
                try:
                    raw_value = await asyncio.wait_for(
                        self.read_raw(), 
                        timeout=self.config.timeout
                    )
                    break
                except asyncio.TimeoutError:
                    logger.warning(f"Sensor {self.config.sensor_id} read timeout (attempt {attempt + 1})")
                    if attempt == self.config.retry_count - 1:
                        raise
                except Exception as e:
                    logger.warning(f"Sensor {self.config.sensor_id} read error: {e} (attempt {attempt + 1})")
                    if attempt == self.config.retry_count - 1:
                        raise
                    await asyncio.sleep(0.5)  # Brief delay between retries
            
            if raw_value is None:
                return None
            
            # Apply calibration and processing
            processed_value = self._process_value(raw_value)
            
            # Create reading
            reading = SensorReading(
                sensor_id=self.config.sensor_id,
                sensor_type=self.config.sensor_type,
                timestamp=time.time(),
                value=processed_value,
                unit=self._get_unit(),
                quality=self._calculate_quality()
            )
            
            # Update statistics
            self.last_reading = reading
            self.total_readings += 1
            self.status = SensorStatus.ONLINE
            
            # Update history for smoothing
            self._update_history(processed_value)
            
            # Check thresholds
            self._check_thresholds(processed_value)
            
            logger.debug(f"Sensor {self.config.sensor_id} reading: {processed_value}")
            return reading
            
        except Exception as e:
            self.error_count += 1
            self.status = SensorStatus.ERROR
            
            error_reading = SensorReading(
                sensor_id=self.config.sensor_id,
                sensor_type=self.config.sensor_type,
                timestamp=time.time(),
                value=None,
                unit=self._get_unit(),
                quality=0.0,
                error_message=str(e)
            )
            
            logger.error(f"Sensor {self.config.sensor_id} error: {e}")
            
            # Notify error callbacks
            for callback in self.error_callbacks:
                try:
                    callback(self.config.sensor_id, e)
                except Exception:
                    pass
            
            return error_reading
    
    def _process_value(self, raw_value: Union[float, bool, Dict[str, float]]) -> Union[float, bool, Dict[str, float]]:
        """Process raw sensor value with calibration and smoothing."""
        if isinstance(raw_value, bool):
            return raw_value
        
        if isinstance(raw_value, dict):
            # Process dictionary values (e.g., temperature and humidity)
            processed = {}
            for key, value in raw_value.items():
                if isinstance(value, (int, float)):
                    processed[key] = value + self.config.calibration_offset
                else:
                    processed[key] = value
            return processed
        
        if isinstance(raw_value, (int, float)):
            # Apply calibration offset
            calibrated_value = raw_value + self.config.calibration_offset
            
            # Apply smoothing if enabled
            if self.config.smoothing_window > 1 and self.reading_history:
                # Use moving average
                recent_values = self.reading_history[-(self.config.smoothing_window-1):]
                recent_values.append(calibrated_value)
                smoothed_value = statistics.mean(recent_values)
                return smoothed_value
            
            return calibrated_value
        
        return raw_value
    
    def _update_history(self, value: Union[float, bool, Dict[str, float]]) -> None:
        """Update reading history for smoothing."""
        if isinstance(value, (int, float)):
            self.reading_history.append(value)
            
            # Keep only last N readings
            max_history = max(self.config.smoothing_window * 2, 10)
            if len(self.reading_history) > max_history:
                self.reading_history = self.reading_history[-max_history:]
    
    def _check_thresholds(self, value: Union[float, bool, Dict[str, float]]) -> None:
        """Check if value exceeds configured thresholds."""
        if not isinstance(value, (int, float)):
            return
        
        if self.config.threshold_high and value > self.config.threshold_high:
            self._trigger_threshold_callback('high', value)
        
        if self.config.threshold_low and value < self.config.threshold_low:
            self._trigger_threshold_callback('low', value)
    
    def _trigger_threshold_callback(self, threshold_type: str, value: float) -> None:
        """Trigger threshold callbacks."""
        for callback in self.threshold_callbacks:
            try:
                callback(self.config.sensor_id, threshold_type, value)
            except Exception as e:
                logger.error(f"Threshold callback error: {e}")
    
    def _calculate_quality(self) -> float:
        """Calculate reading quality based on error rate."""
        if self.total_readings == 0:
            return 1.0
        
        error_rate = self.error_count / self.total_readings
        return max(0.0, 1.0 - error_rate)
    
    @abstractmethod
    def _get_unit(self) -> str:
        """Get unit of measurement for this sensor."""
        pass
    
    def add_threshold_callback(self, callback: Callable[[str, str, float], None]) -> None:
        """Add callback for threshold events."""
        self.threshold_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[str, Exception], None]) -> None:
        """Add callback for error events."""
        self.error_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get sensor statistics."""
        uptime = self.total_readings * self.config.polling_interval if self.total_readings > 0 else 0
        
        return {
            'sensor_id': self.config.sensor_id,
            'sensor_type': self.config.sensor_type.value,
            'status': self.status.value,
            'total_readings': self.total_readings,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(1, self.total_readings),
            'quality': self._calculate_quality(),
            'uptime_seconds': uptime,
            'last_reading_time': self.last_reading.timestamp if self.last_reading else None,
            'configuration': {
                'polling_interval': self.config.polling_interval,
                'enabled': self.config.enabled,
                'pin': self.config.pin,
                'threshold_high': self.config.threshold_high,
                'threshold_low': self.config.threshold_low
            }
        }


class DHT22Sensor(BaseSensor):
    """DHT22 temperature and humidity sensor."""
    
    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self.dht_device = None
    
    async def initialize(self) -> bool:
        """Initialize DHT22 sensor."""
        if not PI_LIBRARIES_AVAILABLE:
            logger.warning("DHT22 sensor libraries not available, using mock")
            self.dht_device = MockDHT22()
            self.status = SensorStatus.ONLINE
            return True
        
        try:
            if self.config.pin is None:
                raise ValueError("DHT22 sensor requires pin configuration")
            
            # Initialize DHT22 device
            pin = getattr(board, f'D{self.config.pin}')
            self.dht_device = adafruit_dht.DHT22(pin)
            
            self.status = SensorStatus.ONLINE
            logger.info(f"DHT22 sensor initialized on pin {self.config.pin}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DHT22 sensor: {e}")
            self.status = SensorStatus.ERROR
            return False
    
    async def read_raw(self) -> Dict[str, float]:
        """Read temperature and humidity from DHT22."""
        if not self.dht_device:
            raise RuntimeError("DHT22 sensor not initialized")
        
        try:
            temperature = self.dht_device.temperature
            humidity = self.dht_device.humidity
            
            if temperature is None or humidity is None:
                raise RuntimeError("DHT22 sensor returned None values")
            
            return {
                'temperature': float(temperature),
                'humidity': float(humidity)
            }
            
        except Exception as e:
            logger.error(f"DHT22 read error: {e}")
            raise
    
    def _get_unit(self) -> str:
        return "°C, %RH"


class DS18B20Sensor(BaseSensor):
    """DS18B20 temperature sensor (1-Wire)."""
    
    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self.ds18_device = None
        self.onewire_bus = None
    
    async def initialize(self) -> bool:
        """Initialize DS18B20 sensor."""
        if not PI_LIBRARIES_AVAILABLE:
            logger.warning("DS18B20 sensor libraries not available, using mock")
            self.ds18_device = MockDS18B20()
            self.status = SensorStatus.ONLINE
            return True
        
        try:
            if self.config.pin is None:
                raise ValueError("DS18B20 sensor requires pin configuration")
            
            # Initialize 1-Wire bus
            pin = getattr(board, f'D{self.config.pin}')
            self.onewire_bus = OneWireBus(pin)
            
            # Find DS18B20 devices
            devices = adafruit_ds18x20.DS18X20(self.onewire_bus)
            if not devices.rom_addresses:
                raise RuntimeError("No DS18B20 devices found on 1-Wire bus")
            
            self.ds18_device = devices
            self.status = SensorStatus.ONLINE
            logger.info(f"DS18B20 sensor initialized on pin {self.config.pin}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DS18B20 sensor: {e}")
            self.status = SensorStatus.ERROR
            return False
    
    async def read_raw(self) -> float:
        """Read temperature from DS18B20."""
        if not self.ds18_device:
            raise RuntimeError("DS18B20 sensor not initialized")
        
        try:
            # Read from first device
            rom_address = self.ds18_device.rom_addresses[0]
            temperature = self.ds18_device.temperature(rom_address)
            
            if temperature is None:
                raise RuntimeError("DS18B20 sensor returned None value")
            
            return float(temperature)
            
        except Exception as e:
            logger.error(f"DS18B20 read error: {e}")
            raise
    
    def _get_unit(self) -> str:
        return "°C"


class PIRMotionSensor(BaseSensor):
    """PIR motion detection sensor."""
    
    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self.pir_pin = None
    
    async def initialize(self) -> bool:
        """Initialize PIR sensor."""
        if not GPIO_AVAILABLE:
            logger.warning("GPIO not available, using mock PIR sensor")
            self.status = SensorStatus.ONLINE
            return True
        
        try:
            if self.config.pin is None:
                raise ValueError("PIR sensor requires pin configuration")
            
            # Setup GPIO pin
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.config.pin, GPIO.IN)
            
            self.pir_pin = self.config.pin
            self.status = SensorStatus.ONLINE
            logger.info(f"PIR sensor initialized on pin {self.config.pin}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PIR sensor: {e}")
            self.status = SensorStatus.ERROR
            return False
    
    async def read_raw(self) -> bool:
        """Read motion state from PIR sensor."""
        if not GPIO_AVAILABLE:
            # Mock implementation
            import random
            return random.random() < 0.1  # 10% chance of motion
        
        if self.pir_pin is None:
            raise RuntimeError("PIR sensor not initialized")
        
        try:
            motion_detected = GPIO.input(self.pir_pin) == GPIO.HIGH
            return motion_detected
            
        except Exception as e:
            logger.error(f"PIR read error: {e}")
            raise
    
    def _get_unit(self) -> str:
        return "boolean"


class BMP280PressureSensor(BaseSensor):
    """BMP280 pressure and temperature sensor (I2C)."""
    
    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self.bmp_device = None
        self.i2c = None
    
    async def initialize(self) -> bool:
        """Initialize BMP280 sensor."""
        if not PI_LIBRARIES_AVAILABLE:
            logger.warning("BMP280 sensor libraries not available, using mock")
            self.bmp_device = MockBMP280()
            self.status = SensorStatus.ONLINE
            return True
        
        try:
            # Initialize I2C bus
            self.i2c = busio.I2C(board.SCL, board.SDA)
            
            # Initialize BMP280 with I2C address
            address = self.config.i2c_address or 0x77
            self.bmp_device = adafruit_bmp280.Adafruit_BMP280_I2C(self.i2c, address=address)
            
            # Set sea level pressure for altitude calculation
            self.bmp_device.sea_level_pressure = 1013.25
            
            self.status = SensorStatus.ONLINE
            logger.info(f"BMP280 sensor initialized on I2C address 0x{address:02x}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize BMP280 sensor: {e}")
            self.status = SensorStatus.ERROR
            return False
    
    async def read_raw(self) -> Dict[str, float]:
        """Read pressure, temperature, and altitude from BMP280."""
        if not self.bmp_device:
            raise RuntimeError("BMP280 sensor not initialized")
        
        try:
            pressure = self.bmp_device.pressure
            temperature = self.bmp_device.temperature
            altitude = self.bmp_device.altitude
            
            return {
                'pressure': float(pressure),
                'temperature': float(temperature),
                'altitude': float(altitude)
            }
            
        except Exception as e:
            logger.error(f"BMP280 read error: {e}")
            raise
    
    def _get_unit(self) -> str:
        return "hPa, °C, m"


# Mock implementations for testing without hardware
class MockDHT22:
    """Mock DHT22 for testing."""
    @property
    def temperature(self) -> float:
        import random
        return random.uniform(18.0, 28.0)
    
    @property
    def humidity(self) -> float:
        import random
        return random.uniform(30.0, 70.0)


class MockDS18B20:
    """Mock DS18B20 for testing."""
    def __init__(self):
        self.rom_addresses = ["mock_address"]
    
    def temperature(self, rom_address: str) -> float:
        import random
        return random.uniform(15.0, 30.0)


class MockBMP280:
    """Mock BMP280 for testing."""
    @property
    def pressure(self) -> float:
        import random
        return random.uniform(1000.0, 1020.0)
    
    @property
    def temperature(self) -> float:
        import random
        return random.uniform(18.0, 25.0)
    
    @property
    def altitude(self) -> float:
        import random
        return random.uniform(100.0, 200.0)


class SensorManager:
    """Manages multiple sensors with unified interface."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sensors: Dict[str, BaseSensor] = {}
        self.running = False
        self.polling_tasks = {}
        
        # Data storage
        self.data_dir = Path(config.get('data_dir', 'data/sensors'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Event callbacks
        self.reading_callbacks = []
        self.alert_callbacks = []
        
        logger.info("Sensor manager initialized")
    
    def add_sensor(self, sensor_config: SensorConfig) -> bool:
        """Add a sensor to the manager."""
        try:
            # Create sensor instance based on type
            sensor = self._create_sensor(sensor_config)
            
            if sensor:
                self.sensors[sensor_config.sensor_id] = sensor
                logger.info(f"Added sensor: {sensor_config.sensor_id}")
                return True
            else:
                logger.error(f"Failed to create sensor: {sensor_config.sensor_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding sensor {sensor_config.sensor_id}: {e}")
            return False
    
    def _create_sensor(self, config: SensorConfig) -> Optional[BaseSensor]:
        """Create sensor instance based on configuration."""
        sensor_map = {
            SensorType.TEMPERATURE_HUMIDITY: DHT22Sensor,
            SensorType.TEMPERATURE_ONLY: DS18B20Sensor,
            SensorType.MOTION_PIR: PIRMotionSensor,
            SensorType.PRESSURE: BMP280PressureSensor
        }
        
        sensor_class = sensor_map.get(config.sensor_type)
        if sensor_class:
            return sensor_class(config)
        else:
            logger.error(f"Unsupported sensor type: {config.sensor_type}")
            return None
    
    async def start_monitoring(self) -> None:
        """Start monitoring all sensors."""
        if self.running:
            logger.warning("Sensor monitoring already running")
            return
        
        self.running = True
        
        # Initialize all sensors
        for sensor_id, sensor in self.sensors.items():
            try:
                await sensor.initialize()
                
                # Start polling task for each sensor
                task = asyncio.create_task(self._poll_sensor(sensor))
                self.polling_tasks[sensor_id] = task
                
            except Exception as e:
                logger.error(f"Failed to start monitoring for sensor {sensor_id}: {e}")
        
        logger.info(f"Started monitoring {len(self.polling_tasks)} sensors")
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring all sensors."""
        self.running = False
        
        # Cancel all polling tasks
        for task in self.polling_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.polling_tasks:
            await asyncio.gather(*self.polling_tasks.values(), return_exceptions=True)
        
        self.polling_tasks.clear()
        logger.info("Stopped sensor monitoring")
    
    async def _poll_sensor(self, sensor: BaseSensor) -> None:
        """Poll individual sensor at configured interval."""
        sensor_id = sensor.config.sensor_id
        interval = sensor.config.polling_interval
        
        logger.info(f"Started polling sensor {sensor_id} every {interval} seconds")
        
        while self.running and sensor.config.enabled:
            try:
                # Read sensor
                reading = await sensor.read()
                
                if reading:
                    # Store reading
                    await self._store_reading(reading)
                    
                    # Notify callbacks
                    for callback in self.reading_callbacks:
                        try:
                            callback(reading)
                        except Exception as e:
                            logger.error(f"Reading callback error: {e}")
                
                # Wait for next poll
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error polling sensor {sensor_id}: {e}")
                await asyncio.sleep(min(interval, 30))  # Wait before retry
        
        logger.info(f"Stopped polling sensor {sensor_id}")
    
    async def _store_reading(self, reading: SensorReading) -> None:
        """Store sensor reading to file."""
        try:
            # Create daily log file
            date_str = time.strftime('%Y-%m-%d', time.localtime(reading.timestamp))
            log_file = self.data_dir / f"sensor_data_{date_str}.jsonl"
            
            # Append reading to log file
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(reading.to_dict()) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to store sensor reading: {e}")
    
    def get_latest_readings(self) -> Dict[str, SensorReading]:
        """Get latest reading from each sensor."""
        readings = {}
        
        for sensor_id, sensor in self.sensors.items():
            if sensor.last_reading:
                readings[sensor_id] = sensor.last_reading
        
        return readings
    
    def get_sensor_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all sensors."""
        stats = {}
        
        for sensor_id, sensor in self.sensors.items():
            stats[sensor_id] = sensor.get_statistics()
        
        return stats
    
    def add_reading_callback(self, callback: Callable[[SensorReading], None]) -> None:
        """Add callback for new sensor readings."""
        self.reading_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[str, str, Any], None]) -> None:
        """Add callback for sensor alerts."""
        self.alert_callbacks.append(callback)
```

## Implementation Plan

### Phase 1: Core Sensor Framework (Week 1-2)
1. **Base Infrastructure**
   - [ ] Implement `BaseSensor` abstract class
   - [ ] Create `SensorReading` and `SensorConfig` data structures
   - [ ] Build error handling and retry mechanisms
   - [ ] Add sensor status management

2. **DHT22 Temperature/Humidity Sensor**
   - [ ] Implement DHT22 sensor class
   - [ ] Add temperature and humidity reading
   - [ ] Test with physical hardware
   - [ ] Create mock implementation for testing

### Phase 2: Additional Sensors (Week 2-3)
1. **Motion Detection**
   - [ ] Implement PIR motion sensor
   - [ ] Add motion detection logic
   - [ ] Integrate with event system
   - [ ] Test motion triggers

2. **Temperature Sensors**
   - [ ] Implement DS18B20 1-Wire temperature sensor
   - [ ] Add multiple sensor support
   - [ ] Create temperature monitoring
   - [ ] Test accuracy and calibration

### Phase 3: Advanced Sensors (Week 3-4)
1. **Environmental Monitoring**
   - [ ] Implement BMP280 pressure sensor
   - [ ] Add light level sensor (TSL2591)
   - [ ] Create air quality monitoring
   - [ ] Build environmental dashboards

2. **Security Sensors**
   - [ ] Add magnetic door sensors
   - [ ] Implement vibration detection
   - [ ] Create sound level monitoring
   - [ ] Integrate with security events

### Phase 4: Integration and Automation (Week 4-5)
1. **Data Management**
   - [ ] Build sensor data storage system
   - [ ] Create historical data analysis
   - [ ] Add data export capabilities
   - [ ] Implement data retention policies

2. **Automation Features**
   - [ ] Create sensor-based triggers
   - [ ] Add conditional logic
   - [ ] Build notification rules
   - [ ] Integrate with main system

## Testing Strategy

### Hardware Testing
```python
def test_dht22_sensor_reading():
    """Test DHT22 sensor reading functionality."""
    config = SensorConfig(
        sensor_id="test_dht22",
        sensor_type=SensorType.TEMPERATURE_HUMIDITY,
        pin=18,
        polling_interval=10.0
    )
    
    sensor = DHT22Sensor(config)
    
    # Test initialization
    success = asyncio.run(sensor.initialize())
    assert success
    
    # Test reading
    reading = asyncio.run(sensor.read())
    assert reading is not None
    assert isinstance(reading.value, dict)
    assert 'temperature' in reading.value
    assert 'humidity' in reading.value

def test_pir_motion_sensor():
    """Test PIR motion sensor functionality."""
    config = SensorConfig(
        sensor_id="test_pir",
        sensor_type=SensorType.MOTION_PIR,
        pin=24,
        polling_interval=1.0
    )
    
    sensor = PIRMotionSensor(config)
    
    # Test initialization
    success = asyncio.run(sensor.initialize())
    assert success
    
    # Test reading
    reading = asyncio.run(sensor.read())
    assert reading is not None
    assert isinstance(reading.value, bool)
```

### Integration Testing
```python
def test_sensor_manager_integration():
    """Test sensor manager with multiple sensors."""
    config = {
        'data_dir': '/tmp/test_sensors'
    }
    
    manager = SensorManager(config)
    
    # Add test sensors
    dht22_config = SensorConfig(
        sensor_id="dht22_1",
        sensor_type=SensorType.TEMPERATURE_HUMIDITY,
        pin=18
    )
    
    pir_config = SensorConfig(
        sensor_id="pir_1", 
        sensor_type=SensorType.MOTION_PIR,
        pin=24
    )
    
    assert manager.add_sensor(dht22_config)
    assert manager.add_sensor(pir_config)
    
    # Test monitoring
    asyncio.run(manager.start_monitoring())
    time.sleep(5)  # Let it run for a bit
    asyncio.run(manager.stop_monitoring())
    
    # Check readings were collected
    readings = manager.get_latest_readings()
    assert len(readings) > 0
```

## Acceptance Criteria

### Hardware Support Requirements
- [ ] DHT22 temperature/humidity sensor fully implemented
- [ ] PIR motion detection working correctly
- [ ] DS18B20 temperature sensor functional
- [ ] BMP280 pressure sensor operational
- [ ] All sensors work with mock implementations for testing

### Data Management Requirements
- [ ] Sensor readings stored with timestamps
- [ ] Historical data accessible for analysis
- [ ] Data export functionality available
- [ ] Automatic log rotation implemented

### Integration Requirements
- [ ] Sensors integrate with main security system
- [ ] Motion detection triggers security events
- [ ] Environmental data available in dashboard
- [ ] Threshold-based alerts working

### Performance Requirements
- [ ] Sensor polling adds <5% CPU overhead
- [ ] Data storage efficient and scalable
- [ ] Real-time readings available within 1 second
- [ ] System stable with 10+ sensors running

This implementation transforms the Raspberry Pi into a comprehensive IoT sensor hub, enhancing the doorbell system with environmental monitoring and advanced automation capabilities.