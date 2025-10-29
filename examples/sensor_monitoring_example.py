#!/usr/bin/env python3
"""
Example: Raspberry Pi Sensor Monitoring

This example demonstrates how to use the sensor integration system
to monitor environmental conditions on a Raspberry Pi.

Features:
- Multiple sensor types (DHT22, PIR, BMP280)
- Real-time monitoring with callbacks
- Threshold alerts
- Data logging
- Statistics reporting

Usage:
    python examples/sensor_monitoring_example.py
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hardware.sensors import (
    SensorManager,
    SensorConfig,
    SensorType,
    SensorReading,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SensorMonitor:
    """Example sensor monitoring application."""
    
    def __init__(self):
        self.manager = None
        self.running = False
        
    def setup_sensors(self):
        """Configure and initialize sensors."""
        logger.info("Setting up sensors...")
        
        # Create sensor manager
        config = {
            'data_dir': 'data/sensors'
        }
        self.manager = SensorManager(config)
        
        # Configure DHT22 Temperature/Humidity Sensor
        dht22_config = SensorConfig(
            sensor_id='indoor_climate',
            sensor_type=SensorType.TEMPERATURE_HUMIDITY,
            pin=4,
            polling_interval=30.0,
            calibration_offset=0.0,
            threshold_high=28.0,  # Alert if too hot
            threshold_low=15.0,   # Alert if too cold
            smoothing_window=3    # Smooth readings over 3 samples
        )
        self.manager.add_sensor(dht22_config)
        
        # Configure PIR Motion Sensor
        pir_config = SensorConfig(
            sensor_id='entrance_motion',
            sensor_type=SensorType.MOTION_PIR,
            pin=17,
            polling_interval=1.0  # Check every second
        )
        self.manager.add_sensor(pir_config)
        
        # Configure BMP280 Pressure Sensor
        bmp280_config = SensorConfig(
            sensor_id='weather_station',
            sensor_type=SensorType.PRESSURE,
            i2c_address=0x77,
            polling_interval=60.0  # Check every minute
        )
        self.manager.add_sensor(bmp280_config)
        
        # Add callbacks
        self.manager.add_reading_callback(self.on_sensor_reading)
        
        # Add threshold callbacks to individual sensors
        for sensor in self.manager.sensors.values():
            sensor.add_threshold_callback(self.on_threshold_alert)
            sensor.add_error_callback(self.on_sensor_error)
        
        logger.info(f"Configured {len(self.manager.sensors)} sensors")
    
    def on_sensor_reading(self, reading: SensorReading):
        """Handle new sensor readings."""
        timestamp = datetime.fromtimestamp(reading.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        if reading.sensor_type == SensorType.TEMPERATURE_HUMIDITY:
            if isinstance(reading.value, dict):
                temp = reading.value.get('temperature', 'N/A')
                humidity = reading.value.get('humidity', 'N/A')
                logger.info(
                    f"[{timestamp}] 🌡️  Indoor: {temp:.1f}°C, {humidity:.1f}% RH "
                    f"(quality: {reading.quality:.0%})"
                )
        
        elif reading.sensor_type == SensorType.MOTION_PIR:
            if reading.value:
                logger.info(f"[{timestamp}] 🚶 Motion detected at entrance!")
        
        elif reading.sensor_type == SensorType.PRESSURE:
            if isinstance(reading.value, dict):
                pressure = reading.value.get('pressure', 'N/A')
                altitude = reading.value.get('altitude', 'N/A')
                logger.info(
                    f"[{timestamp}] 🌤️  Weather: {pressure:.1f} hPa, "
                    f"altitude: {altitude:.1f}m"
                )
    
    def on_threshold_alert(self, sensor_id: str, threshold_type: str, value: float):
        """Handle threshold alerts."""
        if threshold_type == 'high':
            logger.warning(f"⚠️  HIGH ALERT [{sensor_id}]: {value:.1f} exceeds threshold!")
        elif threshold_type == 'low':
            logger.warning(f"⚠️  LOW ALERT [{sensor_id}]: {value:.1f} below threshold!")
    
    def on_sensor_error(self, sensor_id: str, exception: Exception):
        """Handle sensor errors."""
        logger.error(f"❌ Sensor error [{sensor_id}]: {exception}")
    
    async def monitor(self, duration_seconds: int = 300):
        """
        Monitor sensors for specified duration.
        
        Args:
            duration_seconds: How long to monitor (default: 5 minutes)
        """
        logger.info(f"Starting sensor monitoring for {duration_seconds} seconds...")
        
        # Start monitoring
        await self.manager.start_monitoring()
        self.running = True
        
        try:
            # Monitor for specified duration
            await asyncio.sleep(duration_seconds)
            
        except asyncio.CancelledError:
            logger.info("Monitoring cancelled")
        
        finally:
            # Stop monitoring
            await self.manager.stop_monitoring()
            self.running = False
            
            # Print statistics
            self.print_statistics()
    
    def print_statistics(self):
        """Print sensor statistics."""
        logger.info("\n" + "="*60)
        logger.info("SENSOR STATISTICS")
        logger.info("="*60)
        
        stats = self.manager.get_sensor_statistics()
        
        for sensor_id, sensor_stats in stats.items():
            logger.info(f"\n📊 {sensor_id}:")
            logger.info(f"  Status: {sensor_stats['status']}")
            logger.info(f"  Total readings: {sensor_stats['total_readings']}")
            logger.info(f"  Error count: {sensor_stats['error_count']}")
            logger.info(f"  Error rate: {sensor_stats['error_rate']:.2%}")
            logger.info(f"  Quality: {sensor_stats['quality']:.2%}")
            logger.info(f"  Uptime: {sensor_stats['uptime_seconds']:.0f} seconds")
        
        logger.info("\n" + "="*60)
        
        # Print latest readings
        logger.info("\nLATEST READINGS:")
        readings = self.manager.get_latest_readings()
        for sensor_id, reading in readings.items():
            logger.info(f"  {sensor_id}: {reading.value} {reading.unit}")
    
    async def run_continuous(self):
        """Run continuous monitoring until interrupted."""
        logger.info("Starting continuous sensor monitoring...")
        logger.info("Press Ctrl+C to stop")
        
        # Setup signal handler
        loop = asyncio.get_running_loop()
        
        def signal_handler():
            logger.info("\nShutdown signal received...")
            if self.manager and self.running:
                loop.create_task(self.manager.stop_monitoring())
        
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)
        
        # Start monitoring
        await self.manager.start_monitoring()
        self.running = True
        
        try:
            # Run until interrupted
            while self.running:
                await asyncio.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        
        finally:
            await self.manager.stop_monitoring()
            self.print_statistics()


async def main():
    """Main entry point."""
    # Create monitor
    monitor = SensorMonitor()
    
    # Setup sensors
    monitor.setup_sensors()
    
    # Choose monitoring mode
    if len(sys.argv) > 1:
        # Run for specified duration
        duration = int(sys.argv[1])
        logger.info(f"Running for {duration} seconds...")
        await monitor.monitor(duration_seconds=duration)
    else:
        # Run continuously
        await monitor.run_continuous()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nMonitoring stopped by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
