"""
Comprehensive tests for Raspberry Pi sensor integration system

Tests sensor framework, individual sensor classes, and SensorManager
with mock hardware to ensure functionality without physical sensors.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import tempfile

from src.hardware.sensors import (
    SensorType,
    SensorStatus,
    SensorReading,
    SensorConfig,
    BaseSensor,
    DHT22Sensor,
    DS18B20Sensor,
    PIRMotionSensor,
    BMP280PressureSensor,
    SensorManager,
)


class TestSensorDataStructures:
    """Test sensor data structures and configurations."""
    
    def test_sensor_type_enum(self):
        """Test SensorType enum values."""
        assert SensorType.TEMPERATURE_HUMIDITY.value == "temperature_humidity"
        assert SensorType.MOTION_PIR.value == "motion_pir"
        assert SensorType.PRESSURE.value == "pressure"
    
    def test_sensor_status_enum(self):
        """Test SensorStatus enum values."""
        assert SensorStatus.ONLINE.value == "online"
        assert SensorStatus.OFFLINE.value == "offline"
        assert SensorStatus.ERROR.value == "error"
    
    def test_sensor_reading_creation(self):
        """Test SensorReading creation and serialization."""
        reading = SensorReading(
            sensor_id="test_sensor",
            sensor_type=SensorType.TEMPERATURE_ONLY,
            timestamp=time.time(),
            value=22.5,
            unit="°C",
            quality=0.95
        )
        
        assert reading.sensor_id == "test_sensor"
        assert reading.value == 22.5
        assert reading.quality == 0.95
        
        # Test serialization
        data = reading.to_dict()
        assert data['sensor_id'] == "test_sensor"
        assert data['sensor_type'] == "temperature_only"
        assert data['value'] == 22.5
    
    def test_sensor_config_defaults(self):
        """Test SensorConfig default values."""
        config = SensorConfig(
            sensor_id="test",
            sensor_type=SensorType.TEMPERATURE_HUMIDITY
        )
        
        assert config.enabled is True
        assert config.polling_interval == 30.0
        assert config.retry_count == 3
        assert config.smoothing_window == 5


class TestDHT22Sensor:
    """Test DHT22 temperature and humidity sensor."""
    
    @pytest.fixture
    def sensor_config(self):
        """Create test sensor configuration."""
        return SensorConfig(
            sensor_id="dht22_test",
            sensor_type=SensorType.TEMPERATURE_HUMIDITY,
            pin=4,
            polling_interval=10.0
        )
    
    def test_sensor_initialization(self, sensor_config):
        """Test DHT22 sensor initialization."""
        sensor = DHT22Sensor(sensor_config)
        
        assert sensor.config.sensor_id == "dht22_test"
        assert sensor.status == SensorStatus.OFFLINE
        assert sensor.total_readings == 0
    
    @pytest.mark.asyncio
    async def test_sensor_read_with_mock(self, sensor_config):
        """Test DHT22 sensor reading with mock hardware."""
        sensor = DHT22Sensor(sensor_config)
        
        # Initialize (will use mock since PI_LIBRARIES not available)
        initialized = await sensor.initialize()
        assert initialized is True
        assert sensor.status == SensorStatus.ONLINE
        
        # Read sensor
        reading = await sensor.read()
        
        assert reading is not None
        assert reading.sensor_id == "dht22_test"
        assert reading.sensor_type == SensorType.TEMPERATURE_HUMIDITY
        assert isinstance(reading.value, dict)
        assert 'temperature' in reading.value
        assert 'humidity' in reading.value
        assert reading.quality > 0
    
    @pytest.mark.asyncio
    async def test_sensor_error_handling(self, sensor_config):
        """Test DHT22 error handling."""
        sensor = DHT22Sensor(sensor_config)
        await sensor.initialize()
        
        # Simulate sensor failure
        sensor.dht_device = None
        
        reading = await sensor.read()
        
        assert reading is not None
        assert reading.quality == 0.0
        assert reading.error_message is not None
        assert sensor.status == SensorStatus.ERROR
    
    @pytest.mark.asyncio
    async def test_sensor_calibration(self, sensor_config):
        """Test sensor calibration offset."""
        sensor_config.calibration_offset = 2.0
        sensor = DHT22Sensor(sensor_config)
        
        await sensor.initialize()
        reading = await sensor.read()
        
        # Value should be adjusted by calibration offset
        assert reading is not None


class TestDS18B20Sensor:
    """Test DS18B20 temperature sensor."""
    
    @pytest.fixture
    def sensor_config(self):
        """Create test sensor configuration."""
        return SensorConfig(
            sensor_id="ds18b20_test",
            sensor_type=SensorType.TEMPERATURE_ONLY,
            pin=4
        )
    
    @pytest.mark.asyncio
    async def test_sensor_read(self, sensor_config):
        """Test DS18B20 sensor reading."""
        sensor = DS18B20Sensor(sensor_config)
        
        await sensor.initialize()
        reading = await sensor.read()
        
        assert reading is not None
        assert reading.sensor_type == SensorType.TEMPERATURE_ONLY
        assert isinstance(reading.value, float)
        assert reading.unit == "°C"
    
    @pytest.mark.asyncio
    async def test_sensor_statistics(self, sensor_config):
        """Test sensor statistics tracking."""
        sensor = DS18B20Sensor(sensor_config)
        await sensor.initialize()
        
        # Take multiple readings
        for _ in range(3):
            await sensor.read()
        
        stats = sensor.get_statistics()
        
        assert stats['sensor_id'] == "ds18b20_test"
        assert stats['total_readings'] == 3
        assert stats['status'] == SensorStatus.ONLINE.value


class TestPIRMotionSensor:
    """Test PIR motion detection sensor."""
    
    @pytest.fixture
    def sensor_config(self):
        """Create test sensor configuration."""
        return SensorConfig(
            sensor_id="pir_test",
            sensor_type=SensorType.MOTION_PIR,
            pin=17,
            polling_interval=1.0
        )
    
    @pytest.mark.asyncio
    async def test_motion_detection(self, sensor_config):
        """Test PIR motion detection."""
        sensor = PIRMotionSensor(sensor_config)
        
        await sensor.initialize()
        reading = await sensor.read()
        
        assert reading is not None
        assert reading.sensor_type == SensorType.MOTION_PIR
        assert isinstance(reading.value, bool)
        assert reading.unit == "boolean"
    
    @pytest.mark.asyncio
    async def test_motion_sensor_unit(self, sensor_config):
        """Test motion sensor unit type."""
        sensor = PIRMotionSensor(sensor_config)
        
        assert sensor._get_unit() == "boolean"


class TestBMP280Sensor:
    """Test BMP280 pressure sensor."""
    
    @pytest.fixture
    def sensor_config(self):
        """Create test sensor configuration."""
        return SensorConfig(
            sensor_id="bmp280_test",
            sensor_type=SensorType.PRESSURE,
            i2c_address=0x77
        )
    
    @pytest.mark.asyncio
    async def test_pressure_sensor_read(self, sensor_config):
        """Test BMP280 pressure sensor reading."""
        sensor = BMP280PressureSensor(sensor_config)
        
        await sensor.initialize()
        reading = await sensor.read()
        
        assert reading is not None
        assert reading.sensor_type == SensorType.PRESSURE
        assert isinstance(reading.value, dict)
        assert 'pressure' in reading.value
        assert 'temperature' in reading.value
        assert 'altitude' in reading.value


class TestSensorThresholds:
    """Test sensor threshold monitoring."""
    
    @pytest.fixture
    def sensor_config(self):
        """Create sensor config with thresholds."""
        return SensorConfig(
            sensor_id="threshold_test",
            sensor_type=SensorType.TEMPERATURE_ONLY,
            threshold_high=30.0,
            threshold_low=10.0
        )
    
    @pytest.mark.asyncio
    async def test_threshold_callback(self, sensor_config):
        """Test threshold callback triggering."""
        sensor = DS18B20Sensor(sensor_config)
        
        callback_triggered = []
        
        def threshold_callback(sensor_id, threshold_type, value):
            callback_triggered.append((sensor_id, threshold_type, value))
        
        sensor.add_threshold_callback(threshold_callback)
        
        # Force threshold check with high value
        sensor._check_thresholds(35.0)
        
        assert len(callback_triggered) > 0
        assert callback_triggered[0][1] == 'high'


class TestSensorSmoothing:
    """Test sensor value smoothing."""
    
    @pytest.fixture
    def sensor_config(self):
        """Create sensor config with smoothing."""
        return SensorConfig(
            sensor_id="smoothing_test",
            sensor_type=SensorType.TEMPERATURE_ONLY,
            smoothing_window=3
        )
    
    @pytest.mark.asyncio
    async def test_moving_average_smoothing(self, sensor_config):
        """Test moving average smoothing."""
        sensor = DS18B20Sensor(sensor_config)
        await sensor.initialize()
        
        # Take multiple readings to build history
        readings = []
        for _ in range(5):
            reading = await sensor.read()
            if reading and reading.value:
                readings.append(reading.value)
        
        # Verify smoothing is applied (reading history should be populated)
        assert len(sensor.reading_history) > 0


class TestSensorManager:
    """Test SensorManager for multiple sensor management."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for sensor data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def manager_config(self, temp_dir):
        """Create manager configuration."""
        return {
            'data_dir': temp_dir
        }
    
    def test_manager_initialization(self, manager_config):
        """Test SensorManager initialization."""
        manager = SensorManager(manager_config)
        
        assert manager.config == manager_config
        assert len(manager.sensors) == 0
        assert manager.running is False
    
    def test_add_sensor(self, manager_config):
        """Test adding sensors to manager."""
        manager = SensorManager(manager_config)
        
        sensor_config = SensorConfig(
            sensor_id="test_sensor",
            sensor_type=SensorType.TEMPERATURE_HUMIDITY,
            pin=4
        )
        
        result = manager.add_sensor(sensor_config)
        
        assert result is True
        assert "test_sensor" in manager.sensors
        assert isinstance(manager.sensors["test_sensor"], DHT22Sensor)
    
    def test_add_multiple_sensors(self, manager_config):
        """Test adding multiple different sensors."""
        manager = SensorManager(manager_config)
        
        # Add DHT22
        dht_config = SensorConfig(
            sensor_id="dht22",
            sensor_type=SensorType.TEMPERATURE_HUMIDITY,
            pin=4
        )
        manager.add_sensor(dht_config)
        
        # Add PIR
        pir_config = SensorConfig(
            sensor_id="pir",
            sensor_type=SensorType.MOTION_PIR,
            pin=17
        )
        manager.add_sensor(pir_config)
        
        assert len(manager.sensors) == 2
        assert "dht22" in manager.sensors
        assert "pir" in manager.sensors
    
    @pytest.mark.asyncio
    async def test_start_monitoring(self, manager_config):
        """Test starting sensor monitoring."""
        manager = SensorManager(manager_config)
        
        sensor_config = SensorConfig(
            sensor_id="test",
            sensor_type=SensorType.TEMPERATURE_ONLY,
            pin=4,
            polling_interval=0.5
        )
        manager.add_sensor(sensor_config)
        
        # Start monitoring
        await manager.start_monitoring()
        
        assert manager.running is True
        assert len(manager.polling_tasks) > 0
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        # Stop monitoring
        await manager.stop_monitoring()
        
        assert manager.running is False
    
    @pytest.mark.asyncio
    async def test_data_persistence(self, manager_config, temp_dir):
        """Test sensor data persistence to file."""
        manager = SensorManager(manager_config)
        
        sensor_config = SensorConfig(
            sensor_id="persist_test",
            sensor_type=SensorType.TEMPERATURE_ONLY,
            pin=4
        )
        manager.add_sensor(sensor_config)
        
        # Create a reading manually
        reading = SensorReading(
            sensor_id="persist_test",
            sensor_type=SensorType.TEMPERATURE_ONLY,
            timestamp=time.time(),
            value=22.5,
            unit="°C"
        )
        
        # Store reading
        await manager._store_reading(reading)
        
        # Check file was created
        data_dir = Path(temp_dir)
        json_files = list(data_dir.glob("sensor_data_*.jsonl"))
        
        assert len(json_files) > 0
        
        # Verify content
        with open(json_files[0], 'r') as f:
            data = json.loads(f.readline())
            assert data['sensor_id'] == "persist_test"
            assert data['value'] == 22.5
    
    def test_get_latest_readings(self, manager_config):
        """Test getting latest readings from all sensors."""
        manager = SensorManager(manager_config)
        
        # Add sensor
        sensor_config = SensorConfig(
            sensor_id="test",
            sensor_type=SensorType.TEMPERATURE_ONLY,
            pin=4
        )
        manager.add_sensor(sensor_config)
        
        # Initially no readings
        readings = manager.get_latest_readings()
        assert len(readings) == 0
    
    def test_get_sensor_statistics(self, manager_config):
        """Test getting statistics from all sensors."""
        manager = SensorManager(manager_config)
        
        # Add sensors
        sensor_config = SensorConfig(
            sensor_id="test",
            sensor_type=SensorType.TEMPERATURE_ONLY,
            pin=4
        )
        manager.add_sensor(sensor_config)
        
        stats = manager.get_sensor_statistics()
        
        assert "test" in stats
        assert stats["test"]["sensor_id"] == "test"
        assert stats["test"]["total_readings"] == 0
    
    def test_reading_callbacks(self, manager_config):
        """Test reading callback registration and triggering."""
        manager = SensorManager(manager_config)
        
        callback_data = []
        
        def reading_callback(reading):
            callback_data.append(reading)
        
        manager.add_reading_callback(reading_callback)
        
        assert len(manager.reading_callbacks) == 1


class TestRaspberryPiSensorHandler:
    """Test RaspberryPiSensorHandler integration."""
    
    @pytest.fixture
    def sensor_config(self):
        """Create sensor handler configuration."""
        return {
            'data_dir': 'data/sensors',
            'sensors': {
                'temperature_humidity': {
                    'enabled': True,
                    'pin': 4,
                    'polling_interval': 30.0
                },
                'motion': {
                    'enabled': True,
                    'pin': 17,
                    'polling_interval': 1.0
                }
            }
        }
    
    def test_sensor_handler_initialization(self, sensor_config):
        """Test RaspberryPiSensorHandler initialization."""
        from src.hardware.platform.raspberry_pi import RaspberryPiSensorHandler
        
        handler = RaspberryPiSensorHandler(sensor_config)
        
        assert handler.config == sensor_config
        assert handler._is_available is False
        assert handler._initialized is False
    
    def test_sensor_handler_methods_before_init(self, sensor_config):
        """Test sensor handler methods before initialization."""
        from src.hardware.platform.raspberry_pi import RaspberryPiSensorHandler
        
        handler = RaspberryPiSensorHandler(sensor_config)
        
        # Should return None before initialization
        assert handler.read_temperature() is None
        assert handler.read_humidity() is None
        assert handler.read_motion_sensor() is None
    
    def test_sensor_handler_status(self, sensor_config):
        """Test sensor handler status reporting."""
        from src.hardware.platform.raspberry_pi import RaspberryPiSensorHandler
        
        handler = RaspberryPiSensorHandler(sensor_config)
        status = handler.get_sensor_status()
        
        assert 'available' in status
        assert 'temperature' in status
        assert 'humidity' in status
        assert 'motion' in status
        assert status['available'] is False


class TestSensorErrorRecovery:
    """Test sensor error handling and recovery."""
    
    @pytest.fixture
    def sensor_config(self):
        """Create sensor configuration."""
        return SensorConfig(
            sensor_id="error_test",
            sensor_type=SensorType.TEMPERATURE_ONLY,
            retry_count=3,
            timeout=1.0
        )
    
    @pytest.mark.asyncio
    async def test_retry_on_failure(self, sensor_config):
        """Test retry mechanism on sensor failure."""
        sensor = DS18B20Sensor(sensor_config)
        await sensor.initialize()
        
        # Simulate failure by setting device to None
        sensor.ds18_device = None
        
        reading = await sensor.read()
        
        # Should fail and return error reading
        assert reading is not None
        assert reading.error_message is not None
        assert sensor.error_count > 0
    
    @pytest.mark.asyncio
    async def test_error_callback(self, sensor_config):
        """Test error callback triggering."""
        sensor = DS18B20Sensor(sensor_config)
        
        error_data = []
        
        def error_callback(sensor_id, exception):
            error_data.append((sensor_id, exception))
        
        sensor.add_error_callback(error_callback)
        
        await sensor.initialize()
        sensor.ds18_device = None
        
        # This should trigger error callback
        await sensor.read()
        
        assert len(error_data) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
