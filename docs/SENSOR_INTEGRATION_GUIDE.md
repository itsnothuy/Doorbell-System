# Raspberry Pi Sensor Integration Guide

## Overview

The Doorbell Security System now includes comprehensive sensor integration for Raspberry Pi, enabling environmental monitoring, motion detection, and advanced automation features.

## Supported Sensors

### 1. DHT22 Temperature/Humidity Sensor
- **Type**: Digital temperature and humidity sensor
- **Interface**: GPIO (single-wire protocol)
- **Measurements**: Temperature (°C) and Relative Humidity (%)
- **Accuracy**: ±0.5°C, ±2-5% RH
- **Pin**: Any GPIO pin (default: GPIO 4)

### 2. DS18B20 Temperature Sensor
- **Type**: Digital temperature sensor
- **Interface**: 1-Wire protocol
- **Measurements**: Temperature (°C)
- **Accuracy**: ±0.5°C
- **Pin**: Any GPIO pin (default: GPIO 4)
- **Features**: Multiple sensors on same bus, waterproof versions available

### 3. PIR Motion Sensor
- **Type**: Passive Infrared motion detector
- **Interface**: GPIO digital input
- **Measurements**: Motion detection (boolean)
- **Range**: Typically 5-7 meters
- **Pin**: Any GPIO pin (default: GPIO 17)

### 4. BMP280 Pressure Sensor
- **Type**: Barometric pressure and temperature sensor
- **Interface**: I2C
- **Measurements**: Pressure (hPa), Temperature (°C), Altitude (m)
- **Accuracy**: ±1 hPa, ±1°C
- **I2C Address**: 0x76 or 0x77 (default: 0x77)

## Hardware Setup

### DHT22 Wiring
```
DHT22 Sensor    Raspberry Pi
-----------     ------------
VCC (+)    ->   3.3V (Pin 1)
DATA       ->   GPIO 4 (Pin 7)
GND (-)    ->   Ground (Pin 6)

Note: Add 10kΩ pull-up resistor between VCC and DATA
```

### DS18B20 Wiring
```
DS18B20 Sensor  Raspberry Pi
--------------  ------------
VCC (Red)  ->   3.3V (Pin 1)
DATA (Yel) ->   GPIO 4 (Pin 7)
GND (Blk)  ->   Ground (Pin 6)

Note: Add 4.7kΩ pull-up resistor between VCC and DATA
Enable 1-Wire: Add 'dtoverlay=w1-gpio' to /boot/config.txt
```

### PIR Motion Sensor Wiring
```
PIR Sensor      Raspberry Pi
----------      ------------
VCC        ->   5V (Pin 2)
OUT        ->   GPIO 17 (Pin 11)
GND        ->   Ground (Pin 9)
```

### BMP280 Wiring (I2C)
```
BMP280 Sensor   Raspberry Pi
-------------   ------------
VCC        ->   3.3V (Pin 1)
GND        ->   Ground (Pin 6)
SCL        ->   SCL (Pin 5)
SDA        ->   SDA (Pin 3)

Enable I2C: Use 'sudo raspi-config' -> Interface Options -> I2C
```

## Software Configuration

### 1. Install Dependencies

```bash
# Install Raspberry Pi sensor libraries
pip install -r requirements-pi.txt

# Or install individually
pip install adafruit-blinka
pip install adafruit-circuitpython-dht
pip install adafruit-circuitpython-ds18x20
pip install adafruit-circuitpython-bmp280
```

### 2. Enable Required Interfaces

```bash
# Enable I2C for BMP280
sudo raspi-config
# Select: Interface Options -> I2C -> Enable

# Enable 1-Wire for DS18B20
echo "dtoverlay=w1-gpio" | sudo tee -a /boot/config.txt
sudo reboot
```

### 3. Configure Sensors in Code

#### Option A: Configuration Dictionary

```python
from src.hardware.platform.raspberry_pi import RaspberryPiSensorHandler

# Create sensor configuration
sensor_config = {
    'data_dir': 'data/sensors',
    'sensors': {
        'temperature_humidity': {
            'enabled': True,
            'pin': 4,
            'polling_interval': 30.0  # seconds
        },
        'motion': {
            'enabled': True,
            'pin': 17,
            'polling_interval': 1.0  # seconds
        },
        'temperature_only': {
            'enabled': False,
            'pin': 4,
            'polling_interval': 30.0
        },
        'pressure': {
            'enabled': True,
            'i2c_address': 0x77,
            'polling_interval': 60.0
        }
    }
}

# Initialize sensor handler
handler = RaspberryPiSensorHandler(sensor_config)
handler.initialize()
```

#### Option B: Using HardwareConfig

```python
from config.hardware_config import HardwareConfig, SensorConfig

# Create configuration
hardware_config = HardwareConfig()
hardware_config.sensor_config.enabled = True
hardware_config.sensor_config.temperature_humidity = {
    'enabled': True,
    'pin': 4,
    'polling_interval': 30.0
}
hardware_config.sensor_config.motion = {
    'enabled': True,
    'pin': 17,
    'polling_interval': 1.0
}

# Convert to sensor handler config
sensor_handler_config = hardware_config.sensor_config.get_sensor_manager_config()

# Initialize
from src.hardware.platform.raspberry_pi import RaspberryPiSensorHandler
handler = RaspberryPiSensorHandler(sensor_handler_config)
handler.initialize()
```

## Usage Examples

### Basic Sensor Reading

```python
from src.hardware.platform.raspberry_pi import RaspberryPiSensorHandler
import asyncio

# Configuration
sensor_config = {
    'data_dir': 'data/sensors',
    'sensors': {
        'temperature_humidity': {
            'enabled': True,
            'pin': 4,
            'polling_interval': 30.0
        }
    }
}

# Initialize
handler = RaspberryPiSensorHandler(sensor_config)
handler.initialize()

# Start async monitoring
async def monitor_sensors():
    await handler.start_async_monitoring()
    
    # Let it run for a while
    await asyncio.sleep(60)
    
    # Stop monitoring
    await handler.stop_async_monitoring()

# Run
asyncio.run(monitor_sensors())

# Read current values
temperature = handler.read_temperature()
humidity = handler.read_humidity()
motion = handler.read_motion_sensor()

print(f"Temperature: {temperature}°C")
print(f"Humidity: {humidity}%")
print(f"Motion detected: {motion}")

# Get detailed status
status = handler.get_sensor_status()
print(f"Sensor status: {status}")
```

### Advanced Usage with SensorManager

```python
from src.hardware.sensors import (
    SensorManager, SensorConfig, SensorType, SensorReading
)
import asyncio

# Create sensor manager
config = {'data_dir': 'data/sensors'}
manager = SensorManager(config)

# Add DHT22 sensor
dht22_config = SensorConfig(
    sensor_id='living_room_dht22',
    sensor_type=SensorType.TEMPERATURE_HUMIDITY,
    pin=4,
    polling_interval=30.0,
    calibration_offset=0.5,  # Adjust if sensor reads high/low
    threshold_high=28.0,     # Alert if temperature exceeds 28°C
    threshold_low=15.0       # Alert if temperature drops below 15°C
)
manager.add_sensor(dht22_config)

# Add PIR motion sensor
pir_config = SensorConfig(
    sensor_id='front_door_pir',
    sensor_type=SensorType.MOTION_PIR,
    pin=17,
    polling_interval=1.0
)
manager.add_sensor(pir_config)

# Add reading callback
def on_reading(reading: SensorReading):
    print(f"[{reading.sensor_id}] {reading.value} {reading.unit}")
    
    # Check if motion detected
    if reading.sensor_type == SensorType.MOTION_PIR and reading.value:
        print("⚠️  Motion detected at front door!")

manager.add_reading_callback(on_reading)

# Start monitoring
async def run():
    await manager.start_monitoring()
    
    # Run for 5 minutes
    await asyncio.sleep(300)
    
    # Get statistics
    stats = manager.get_sensor_statistics()
    for sensor_id, sensor_stats in stats.items():
        print(f"\n{sensor_id} Statistics:")
        print(f"  Total readings: {sensor_stats['total_readings']}")
        print(f"  Error rate: {sensor_stats['error_rate']:.2%}")
        print(f"  Quality: {sensor_stats['quality']:.2%}")
    
    # Stop monitoring
    await manager.stop_monitoring()

asyncio.run(run())
```

### Integration with Doorbell Events

```python
from src.hardware.platform.raspberry_pi import RaspberryPiSensorHandler
from src.hardware.sensors import SensorType

# Initialize sensor handler
sensor_config = {
    'data_dir': 'data/sensors',
    'sensors': {
        'temperature_humidity': {'enabled': True, 'pin': 4},
        'motion': {'enabled': True, 'pin': 17}
    }
}
handler = RaspberryPiSensorHandler(sensor_config)
handler.initialize()

# Doorbell event callback
def on_doorbell_event():
    """Called when doorbell is pressed."""
    
    # Get environmental context
    temperature = handler.read_temperature()
    humidity = handler.read_humidity()
    
    # Include in event data
    event_data = {
        'timestamp': time.time(),
        'temperature': temperature,
        'humidity': humidity,
        'source': 'doorbell'
    }
    
    print(f"Doorbell pressed! Temp: {temperature}°C, Humidity: {humidity}%")
    
    # Process face recognition with environmental context
    # ... your doorbell logic here ...
```

### Threshold Alerts

```python
from src.hardware.sensors import SensorManager, SensorConfig, SensorType

manager = SensorManager({'data_dir': 'data/sensors'})

# Configure sensor with thresholds
sensor_config = SensorConfig(
    sensor_id='outdoor_temp',
    sensor_type=SensorType.TEMPERATURE_ONLY,
    pin=4,
    threshold_high=30.0,  # Alert if > 30°C
    threshold_low=0.0     # Alert if < 0°C (freezing)
)

# Add sensor
sensor = manager._create_sensor(sensor_config)

# Add threshold callback
def temperature_alert(sensor_id, threshold_type, value):
    if threshold_type == 'high':
        print(f"🔥 HIGH TEMPERATURE ALERT: {value}°C")
        # Send notification, trigger cooling, etc.
    elif threshold_type == 'low':
        print(f"❄️ LOW TEMPERATURE ALERT: {value}°C")
        # Send notification, trigger heating, etc.

sensor.add_threshold_callback(temperature_alert)

# Start monitoring
manager.sensors['outdoor_temp'] = sensor
asyncio.run(manager.start_monitoring())
```

## Data Storage

Sensor readings are automatically stored in JSONL (JSON Lines) format:

**Location**: `data/sensors/sensor_data_YYYY-MM-DD.jsonl`

**Format**:
```json
{
  "sensor_id": "dht22_main",
  "sensor_type": "temperature_humidity",
  "timestamp": 1698765432.123,
  "value": {"temperature": 22.5, "humidity": 45.0},
  "unit": "°C, %RH",
  "quality": 0.98,
  "error_message": null
}
```

### Reading Sensor Data

```python
import json
from pathlib import Path
from datetime import datetime

# Read today's sensor data
today = datetime.now().strftime('%Y-%m-%d')
log_file = Path(f'data/sensors/sensor_data_{today}.jsonl')

if log_file.exists():
    with open(log_file, 'r') as f:
        for line in f:
            reading = json.loads(line)
            print(f"[{reading['sensor_id']}] {reading['value']} at {reading['timestamp']}")
```

## Troubleshooting

### Sensor Not Detected

**DHT22/DS18B20 Issues:**
```bash
# Check if sensor is connected
sudo i2cdetect -y 1  # For I2C sensors

# For 1-Wire (DS18B20)
ls /sys/bus/w1/devices/
# Should show 28-xxxxxxxxxxxx directories

# Check DHT22 pin
sudo gpio readall  # Verify pin number
```

**BMP280 Not Found:**
```bash
# Check I2C is enabled
sudo raspi-config  # Enable I2C

# Detect I2C address
sudo i2cdetect -y 1
# Should show 76 or 77

# Check wiring
# Ensure SDA/SCL are not swapped
```

### Permission Errors

```bash
# Add user to GPIO group
sudo usermod -a -G gpio $USER

# Add user to I2C group
sudo usermod -a -G i2c $USER

# Reboot to apply changes
sudo reboot
```

### Import Errors

```bash
# Reinstall sensor libraries
pip install --upgrade adafruit-blinka
pip install --upgrade adafruit-circuitpython-dht
pip install --upgrade adafruit-circuitpython-ds18x20
pip install --upgrade adafruit-circuitpython-bmp280

# Check Python version (requires 3.7+)
python --version
```

### High Error Rate

1. **Check wiring**: Ensure pull-up resistors are present
2. **Power supply**: Ensure stable 3.3V/5V power
3. **Adjust timeout**: Increase `timeout` in SensorConfig
4. **Reduce polling rate**: Increase `polling_interval`
5. **Check calibration**: Adjust `calibration_offset` if readings are consistently off

## Best Practices

1. **Use appropriate polling intervals**:
   - Temperature/humidity: 30-60 seconds
   - Motion detection: 0.5-2 seconds
   - Pressure: 60-120 seconds

2. **Enable smoothing** for stable readings:
   ```python
   config = SensorConfig(..., smoothing_window=5)
   ```

3. **Set realistic thresholds** based on your environment

4. **Monitor data directory size**:
   ```bash
   # Clean old sensor data (older than 30 days)
   find data/sensors/ -name "*.jsonl" -mtime +30 -delete
   ```

5. **Handle sensor failures gracefully**:
   - Always check for `None` return values
   - Monitor error rates in statistics
   - Implement fallback logic

## Performance Considerations

- **CPU Usage**: Minimal when using appropriate polling intervals
- **Memory**: ~10-50MB depending on number of sensors
- **Storage**: ~1-5MB per day per sensor (with 30s polling)
- **I/O**: Asynchronous, non-blocking sensor reads

## Next Steps

- Integrate with alerting system
- Create web dashboard for sensor visualization
- Add automated actions based on sensor readings
- Implement machine learning for anomaly detection
- Add support for additional sensor types

## Additional Resources

- [Adafruit DHT22 Guide](https://learn.adafruit.com/dht)
- [DS18B20 Setup Guide](https://learn.adafruit.com/adafruits-raspberry-pi-lesson-11-ds18b20-temperature-sensing)
- [Raspberry Pi GPIO Pinout](https://pinout.xyz/)
- [I2C Configuration](https://learn.adafruit.com/adafruits-raspberry-pi-lesson-4-gpio-setup/configuring-i2c)
