#!/usr/bin/env python3
"""
Simple Sensor Test Script

Quick test script to verify sensor hardware and configuration.
Useful for troubleshooting and initial setup.

Usage:
    python examples/simple_sensor_test.py
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hardware.platform.raspberry_pi import RaspberryPiSensorHandler


def test_sensors():
    """Test all configured sensors."""
    print("="*60)
    print("Raspberry Pi Sensor Integration Test")
    print("="*60)
    print()
    
    # Configure sensors (adjust pins as needed)
    sensor_config = {
        'data_dir': 'data/sensors',
        'sensors': {
            'temperature_humidity': {
                'enabled': True,
                'pin': 4,
                'polling_interval': 5.0
            },
            'motion': {
                'enabled': True,
                'pin': 17,
                'polling_interval': 1.0
            },
            'temperature_only': {
                'enabled': False,
                'pin': 4,
                'polling_interval': 5.0
            },
            'pressure': {
                'enabled': True,
                'i2c_address': 0x77,
                'polling_interval': 10.0
            }
        }
    }
    
    # Initialize sensor handler
    print("Initializing sensor handler...")
    handler = RaspberryPiSensorHandler(sensor_config)
    
    if not handler.initialize():
        print("❌ Failed to initialize sensor handler")
        print("   This is normal if running on non-Pi hardware (will use mocks)")
    else:
        print("✅ Sensor handler initialized")
    
    print()
    print("Testing individual sensors...")
    print("-"*60)
    
    # Test temperature reading
    print("\n🌡️  Temperature Sensor:")
    temperature = handler.read_temperature()
    if temperature is not None:
        print(f"   ✅ Reading: {temperature:.1f}°C")
    else:
        print("   ⚠️  No reading available (sensor not configured or failed)")
    
    # Test humidity reading
    print("\n💧 Humidity Sensor:")
    humidity = handler.read_humidity()
    if humidity is not None:
        print(f"   ✅ Reading: {humidity:.1f}% RH")
    else:
        print("   ⚠️  No reading available (sensor not configured or failed)")
    
    # Test motion sensor
    print("\n🚶 Motion Sensor:")
    motion = handler.read_motion_sensor()
    if motion is not None:
        status = "MOTION DETECTED" if motion else "No motion"
        print(f"   ✅ Status: {status}")
    else:
        print("   ⚠️  No reading available (sensor not configured or failed)")
    
    # Get comprehensive status
    print("\n📊 Complete Sensor Status:")
    print("-"*60)
    status = handler.get_sensor_status()
    
    print(f"\nSystem Available: {status['available']}")
    
    print("\nTemperature:")
    temp_status = status.get('temperature', {})
    print(f"  Enabled: {temp_status.get('enabled', False)}")
    print(f"  Current: {temp_status.get('current', 'N/A')}")
    print(f"  Unit: {temp_status.get('unit', 'N/A')}")
    print(f"  Source: {temp_status.get('source', 'N/A')}")
    
    print("\nHumidity:")
    hum_status = status.get('humidity', {})
    print(f"  Enabled: {hum_status.get('enabled', False)}")
    print(f"  Current: {hum_status.get('current', 'N/A')}")
    print(f"  Unit: {hum_status.get('unit', 'N/A')}")
    print(f"  Source: {hum_status.get('source', 'N/A')}")
    
    print("\nMotion:")
    motion_status = status.get('motion', {})
    print(f"  Enabled: {motion_status.get('enabled', False)}")
    print(f"  Current: {motion_status.get('current', 'N/A')}")
    print(f"  Unit: {motion_status.get('unit', 'N/A')}")
    print(f"  Source: {motion_status.get('source', 'N/A')}")
    
    # Show detailed statistics if available
    if 'statistics' in status:
        print("\n📈 Detailed Statistics:")
        print("-"*60)
        for sensor_id, stats in status['statistics'].items():
            print(f"\n{sensor_id}:")
            print(f"  Status: {stats.get('status', 'N/A')}")
            print(f"  Total Readings: {stats.get('total_readings', 0)}")
            print(f"  Error Count: {stats.get('error_count', 0)}")
            print(f"  Quality: {stats.get('quality', 0):.1%}")
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)
    
    # Cleanup
    handler.cleanup()


def continuous_monitoring():
    """Run continuous monitoring for testing."""
    print("Starting continuous monitoring...")
    print("Press Ctrl+C to stop")
    print()
    
    sensor_config = {
        'data_dir': 'data/sensors',
        'sensors': {
            'temperature_humidity': {
                'enabled': True,
                'pin': 4,
                'polling_interval': 5.0
            },
            'motion': {
                'enabled': True,
                'pin': 17,
                'polling_interval': 1.0
            }
        }
    }
    
    handler = RaspberryPiSensorHandler(sensor_config)
    handler.initialize()
    
    try:
        count = 0
        while True:
            count += 1
            print(f"\n--- Reading #{count} ---")
            
            temp = handler.read_temperature()
            hum = handler.read_humidity()
            motion = handler.read_motion_sensor()
            
            if temp is not None:
                print(f"🌡️  Temperature: {temp:.1f}°C")
            if hum is not None:
                print(f"💧 Humidity: {hum:.1f}% RH")
            if motion is not None:
                status = "🚶 MOTION DETECTED!" if motion else "   No motion"
                print(status)
            
            time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
    finally:
        handler.cleanup()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'continuous':
        continuous_monitoring()
    else:
        test_sensors()
