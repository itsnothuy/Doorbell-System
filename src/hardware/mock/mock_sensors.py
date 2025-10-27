"""
Mock sensor implementation for testing and development

Provides mock environmental sensors for testing without actual hardware.
"""

import logging
import random
import time
from typing import Optional, Dict, Any

from src.hardware.base_hardware import SensorHandler

logger = logging.getLogger(__name__)


class MockSensorHandler(SensorHandler):
    """Mock sensor handler for testing without actual sensor hardware."""
    
    def __init__(self, config: dict):
        """
        Initialize mock sensor handler.
        
        Args:
            config: Configuration dictionary with sensor settings
        """
        self.config = config
        self._is_available = True
        
        # Simulated sensor values with realistic ranges
        self.base_temperature = config.get('base_temperature', 22.0)  # Celsius
        self.base_humidity = config.get('base_humidity', 45.0)  # Percentage
        self.motion_detected = False
        
        # Simulation parameters
        self.temperature_variation = 2.0  # +/- degrees
        self.humidity_variation = 5.0  # +/- percentage
        
        logger.info("Mock sensor handler created")
    
    def read_temperature(self) -> Optional[float]:
        """
        Read simulated temperature.
        
        Returns:
            Temperature in Celsius
        """
        try:
            # Add random variation to base temperature
            variation = random.uniform(-self.temperature_variation, self.temperature_variation)
            temperature = self.base_temperature + variation
            
            logger.debug(f"Mock temperature reading: {temperature:.2f}°C")
            return round(temperature, 2)
            
        except Exception as e:
            logger.error(f"Mock temperature read failed: {e}")
            return None
    
    def read_humidity(self) -> Optional[float]:
        """
        Read simulated humidity.
        
        Returns:
            Relative humidity percentage (0-100)
        """
        try:
            # Add random variation to base humidity
            variation = random.uniform(-self.humidity_variation, self.humidity_variation)
            humidity = self.base_humidity + variation
            
            # Clamp to valid range
            humidity = max(0.0, min(100.0, humidity))
            
            logger.debug(f"Mock humidity reading: {humidity:.2f}%")
            return round(humidity, 2)
            
        except Exception as e:
            logger.error(f"Mock humidity read failed: {e}")
            return None
    
    def read_motion_sensor(self) -> Optional[bool]:
        """
        Read simulated motion sensor.
        
        Returns:
            True if motion detected, False otherwise
        """
        try:
            # Randomly trigger motion detection (10% chance)
            self.motion_detected = random.random() < 0.1
            
            if self.motion_detected:
                logger.debug("Mock motion detected")
            
            return self.motion_detected
            
        except Exception as e:
            logger.error(f"Mock motion sensor read failed: {e}")
            return None
    
    def get_sensor_status(self) -> Dict[str, Any]:
        """
        Get status of all mock sensors.
        
        Returns:
            Dictionary with sensor status
        """
        return {
            'available': self._is_available,
            'temperature': {
                'enabled': True,
                'current': self.read_temperature(),
                'unit': 'celsius'
            },
            'humidity': {
                'enabled': True,
                'current': self.read_humidity(),
                'unit': 'percent'
            },
            'motion': {
                'enabled': True,
                'current': self.motion_detected,
                'unit': 'boolean'
            },
            'timestamp': time.time()
        }
    
    def is_available(self) -> bool:
        """Check if mock sensors are available."""
        return self._is_available
    
    def cleanup(self) -> None:
        """Cleanup mock sensor resources."""
        self._is_available = False
        logger.info("Mock sensor cleanup completed")
    
    def set_base_temperature(self, temperature: float) -> None:
        """
        Set base temperature for simulation.
        
        Args:
            temperature: Base temperature in Celsius
        """
        self.base_temperature = temperature
        logger.info(f"Mock sensor base temperature set to {temperature}°C")
    
    def set_base_humidity(self, humidity: float) -> None:
        """
        Set base humidity for simulation.
        
        Args:
            humidity: Base humidity percentage
        """
        self.base_humidity = max(0.0, min(100.0, humidity))
        logger.info(f"Mock sensor base humidity set to {self.base_humidity}%")
    
    def trigger_motion(self) -> None:
        """Manually trigger motion detection."""
        self.motion_detected = True
        logger.info("Mock motion sensor triggered manually")
    
    def clear_motion(self) -> None:
        """Clear motion detection."""
        self.motion_detected = False
        logger.debug("Mock motion sensor cleared")


__all__ = ['MockSensorHandler']
