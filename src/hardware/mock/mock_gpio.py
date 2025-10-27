"""
Mock GPIO implementation for testing and development

Provides a realistic mock GPIO handler that simulates hardware behavior
including pin states, interrupts, and LED control without actual hardware.
"""

import logging
import time
import random
import threading
from typing import Callable, Optional, Dict, Any

from src.hardware.base_hardware import GPIOHandler, GPIOMode, GPIOEdge

logger = logging.getLogger(__name__)


class MockGPIOHandler(GPIOHandler):
    """Mock GPIO implementation for testing without Raspberry Pi hardware."""
    
    def __init__(self, config: dict):
        """
        Initialize mock GPIO handler.
        
        Args:
            config: Configuration dictionary with GPIO settings
        """
        self.config = config
        self.pin_states = {}
        self.pin_modes = {}
        self.interrupt_callbacks = {}
        self.initialized = False
        
        # Simulation settings
        self.simulate_events = config.get('simulate_events', True)
        self.event_interval_range = config.get('event_interval_range', (30, 120))
        
        # Simulation thread
        self.simulation_thread = None
        self.simulation_running = False
        
        logger.info("Mock GPIO handler created")
    
    def initialize(self) -> bool:
        """Initialize mock GPIO."""
        try:
            logger.info("Initializing mock GPIO...")
            self.initialized = True
            
            # Start event simulation if enabled
            if self.simulate_events:
                self._start_event_simulation()
            
            logger.info("Mock GPIO initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Mock GPIO initialization failed: {e}")
            return False
    
    def setup_pin(self, pin: int, mode: GPIOMode, **kwargs) -> bool:
        """
        Setup mock GPIO pin.
        
        Args:
            pin: GPIO pin number
            mode: Pin mode (INPUT or OUTPUT)
            **kwargs: Additional options (pull_up_down, initial)
        
        Returns:
            True if successful
        """
        try:
            self.pin_modes[pin] = mode
            
            if mode == GPIOMode.INPUT:
                # For input pins, start with LOW state
                self.pin_states[pin] = False
            else:
                # For output pins, use initial value or default to LOW
                self.pin_states[pin] = kwargs.get('initial', False)
            
            logger.debug(f"Mock GPIO pin {pin} setup as {mode.value}")
            return True
            
        except Exception as e:
            logger.error(f"Mock GPIO pin setup failed: {e}")
            return False
    
    def read_pin(self, pin: int) -> Optional[bool]:
        """
        Read mock GPIO pin state.
        
        Args:
            pin: GPIO pin number
        
        Returns:
            Pin state or None if not setup
        """
        return self.pin_states.get(pin)
    
    def write_pin(self, pin: int, value: bool) -> bool:
        """
        Write mock GPIO pin state.
        
        Args:
            pin: GPIO pin number
            value: Value to write (True=HIGH, False=LOW)
        
        Returns:
            True if successful
        """
        try:
            if pin in self.pin_states:
                self.pin_states[pin] = value
                logger.debug(f"Mock GPIO pin {pin} set to {'HIGH' if value else 'LOW'}")
                return True
            else:
                logger.warning(f"Mock GPIO pin {pin} not setup")
                return False
                
        except Exception as e:
            logger.error(f"Mock GPIO write failed: {e}")
            return False
    
    def setup_interrupt(self, pin: int, callback: Callable, edge: GPIOEdge) -> bool:
        """
        Setup mock GPIO interrupt.
        
        Args:
            pin: GPIO pin number
            callback: Function to call on interrupt
            edge: Edge detection mode
        
        Returns:
            True if successful
        """
        try:
            if pin not in self.pin_modes:
                logger.warning(f"Mock GPIO pin {pin} not setup before interrupt")
                return False
            
            self.interrupt_callbacks[pin] = (callback, edge)
            logger.info(f"Mock GPIO interrupt setup for pin {pin} on {edge.value} edge")
            return True
            
        except Exception as e:
            logger.error(f"Mock GPIO interrupt setup failed: {e}")
            return False
    
    def cleanup(self) -> None:
        """Cleanup mock GPIO resources."""
        logger.info("Cleaning up mock GPIO...")
        
        # Stop simulation thread
        self.simulation_running = False
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=2)
        
        # Clear all state
        self.pin_states = {}
        self.pin_modes = {}
        self.interrupt_callbacks = {}
        self.initialized = False
        
        logger.info("Mock GPIO cleanup completed")
    
    def _start_event_simulation(self) -> None:
        """Start background thread to simulate GPIO events."""
        self.simulation_running = True
        self.simulation_thread = threading.Thread(
            target=self._simulate_gpio_events,
            daemon=True,
            name="MockGPIOSimulator"
        )
        self.simulation_thread.start()
        logger.info("Mock GPIO event simulation started")
    
    def _simulate_gpio_events(self) -> None:
        """Simulate periodic GPIO events (e.g., doorbell presses)."""
        while self.simulation_running:
            try:
                # Random interval between events
                min_interval, max_interval = self.event_interval_range
                interval = random.randint(min_interval, max_interval)
                time.sleep(interval)
                
                if not self.simulation_running:
                    break
                
                # Trigger interrupt callbacks
                for pin, (callback, edge) in list(self.interrupt_callbacks.items()):
                    try:
                        # Simulate falling edge (button press)
                        if edge in [GPIOEdge.FALLING, GPIOEdge.BOTH]:
                            logger.info(f"Mock GPIO event triggered on pin {pin}")
                            callback(pin)
                    except Exception as e:
                        logger.error(f"Mock GPIO callback error on pin {pin}: {e}")
                
            except Exception as e:
                logger.error(f"Mock GPIO simulation error: {e}")
                time.sleep(10)  # Wait before retrying
    
    def simulate_pin_change(self, pin: int, new_state: bool) -> None:
        """
        Manually simulate a pin state change.
        
        Args:
            pin: GPIO pin number
            new_state: New pin state
        """
        if pin not in self.pin_states:
            logger.warning(f"Cannot simulate change on uninitialized pin {pin}")
            return
        
        old_state = self.pin_states[pin]
        self.pin_states[pin] = new_state
        
        # Trigger interrupt if registered
        if pin in self.interrupt_callbacks:
            callback, edge = self.interrupt_callbacks[pin]
            
            # Determine if interrupt should fire based on edge type
            should_trigger = False
            if edge == GPIOEdge.RISING and not old_state and new_state:
                should_trigger = True
            elif edge == GPIOEdge.FALLING and old_state and not new_state:
                should_trigger = True
            elif edge == GPIOEdge.BOTH and old_state != new_state:
                should_trigger = True
            
            if should_trigger:
                try:
                    logger.info(f"Mock GPIO triggering interrupt on pin {pin}")
                    threading.Thread(target=callback, args=(pin,), daemon=True).start()
                except Exception as e:
                    logger.error(f"Mock GPIO interrupt trigger error: {e}")
    
    def get_pin_states(self) -> Dict[int, bool]:
        """
        Get current state of all pins.
        
        Returns:
            Dictionary mapping pin numbers to states
        """
        return self.pin_states.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get mock GPIO status.
        
        Returns:
            Status dictionary
        """
        return {
            'initialized': self.initialized,
            'pin_count': len(self.pin_states),
            'interrupt_count': len(self.interrupt_callbacks),
            'simulation_active': self.simulation_running,
            'pin_states': self.get_pin_states()
        }


class MockGPIO:
    """
    Simplified mock GPIO for backward compatibility.
    
    Provides a simple interface compatible with the existing GPIO handler
    for testing without Raspberry Pi hardware.
    """
    
    # Constants to mimic RPi.GPIO
    BCM = "BCM"
    BOARD = "BOARD"
    IN = "IN"
    OUT = "OUT"
    HIGH = True
    LOW = False
    PUD_UP = "PUD_UP"
    PUD_DOWN = "PUD_DOWN"
    PUD_OFF = "PUD_OFF"
    RISING = "RISING"
    FALLING = "FALLING"
    BOTH = "BOTH"
    
    _mode = None
    _warnings = True
    _pin_states = {}
    _pin_modes = {}
    _interrupts = {}
    
    @classmethod
    def setmode(cls, mode):
        """Set GPIO mode."""
        cls._mode = mode
        logger.debug(f"Mock GPIO mode set to {mode}")
    
    @classmethod
    def setwarnings(cls, warnings):
        """Set GPIO warnings."""
        cls._warnings = warnings
    
    @classmethod
    def setup(cls, pin, mode, **kwargs):
        """Setup GPIO pin."""
        cls._pin_modes[pin] = mode
        if mode == cls.IN:
            cls._pin_states[pin] = False
        else:
            cls._pin_states[pin] = kwargs.get('initial', cls.LOW)
        logger.debug(f"Mock GPIO pin {pin} setup as {mode}")
    
    @classmethod
    def input(cls, pin):
        """Read GPIO pin."""
        return cls._pin_states.get(pin, cls.LOW)
    
    @classmethod
    def output(cls, pin, value):
        """Write GPIO pin."""
        cls._pin_states[pin] = value
        logger.debug(f"Mock GPIO pin {pin} set to {'HIGH' if value else 'LOW'}")
    
    @classmethod
    def add_event_detect(cls, pin, edge, callback=None, bouncetime=None):
        """Add event detection."""
        cls._interrupts[pin] = {
            'edge': edge,
            'callback': callback,
            'bouncetime': bouncetime
        }
        logger.debug(f"Mock GPIO interrupt added for pin {pin}")
    
    @classmethod
    def remove_event_detect(cls, pin):
        """Remove event detection."""
        if pin in cls._interrupts:
            del cls._interrupts[pin]
            logger.debug(f"Mock GPIO interrupt removed for pin {pin}")
    
    @classmethod
    def cleanup(cls):
        """Cleanup GPIO."""
        cls._pin_states = {}
        cls._pin_modes = {}
        cls._interrupts = {}
        logger.debug("Mock GPIO cleanup completed")


__all__ = ['MockGPIOHandler', 'MockGPIO']
