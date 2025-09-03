"""
GPIO handler for cross-platform hardware interface
"""

import logging
import time
import threading
from typing import Callable, Optional
from src.platform_detector import platform_detector

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

logger = logging.getLogger(__name__)


class GPIOHandler:
    """Handles GPIO operations for doorbell and LED indicators"""
    
    def __init__(self):
        from config.settings import Settings
        self.settings = Settings()
        
        self.initialized = False
        self.doorbell_callback = None
        self.led_states = {}
        
        # LED control thread
        self.led_thread = None
        self.led_thread_running = False
        
        # Check if we should use real GPIO or mock
        gpio_config = platform_detector.get_gpio_config()
        
        if gpio_config['use_real_gpio'] and GPIO_AVAILABLE:
            self._setup_gpio()
        else:
            self._setup_mock_gpio()
            if platform_detector.is_macos:
                logger.info("Using mock GPIO for macOS development")
            else:
                logger.warning("RPi.GPIO not available - using mock GPIO")
    
    def _setup_gpio(self):
        """Setup GPIO pins"""
        try:
            # Set GPIO mode
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Setup doorbell button pin
            GPIO.setup(
                self.settings.DOORBELL_PIN, 
                GPIO.IN, 
                pull_up_down=GPIO.PUD_UP
            )
            
            # Setup LED pins
            for color, pin in self.settings.STATUS_LED_PINS.items():
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
                self.led_states[color] = False
            
            self.initialized = True
            self.use_real_gpio = True
            logger.info("GPIO initialized successfully")
            
            # Start LED control thread
            self._start_led_thread()
            
        except Exception as e:
            logger.error(f"GPIO setup failed: {e}")
            raise
    
    def _setup_mock_gpio(self):
        """Setup mock GPIO for testing"""
        try:
            # Initialize mock LED states
            for color in self.settings.STATUS_LED_PINS.keys():
                self.led_states[color] = False
            
            self.initialized = True
            self.use_real_gpio = False
            logger.info("Mock GPIO initialized successfully")
            
            # Start LED control thread for mock animations
            self._start_led_thread()
            
        except Exception as e:
            logger.error(f"Mock GPIO setup failed: {e}")
            raise
    
    def setup_doorbell_button(self, callback: Callable):
        """Setup doorbell button interrupt"""
        if not self.initialized:
            logger.error("GPIO not initialized")
            return
        
        try:
            self.doorbell_callback = callback
            
            if self.use_real_gpio:
                # Add event detection for falling edge (button press)
                GPIO.add_event_detect(
                    self.settings.DOORBELL_PIN,
                    GPIO.FALLING,
                    callback=self._doorbell_interrupt,
                    bouncetime=int(self.settings.DEBOUNCE_TIME * 1000)  # Convert to ms
                )
                logger.info(f"Doorbell button configured on GPIO {self.settings.DOORBELL_PIN}")
            else:
                # Mock mode - button will be triggered via web interface
                logger.info("Mock doorbell button configured (use web interface to trigger)")
            
        except Exception as e:
            logger.error(f"Failed to setup doorbell button: {e}")
            raise
    
    def _doorbell_interrupt(self, channel):
        """Handle doorbell button interrupt"""
        try:
            logger.debug(f"Doorbell interrupt on GPIO {channel}")
            
            if self.doorbell_callback:
                # Call the callback in a separate thread to avoid blocking
                threading.Thread(
                    target=self.doorbell_callback,
                    args=(channel,),
                    daemon=True
                ).start()
            
        except Exception as e:
            logger.error(f"Doorbell interrupt error: {e}")
    
    def simulate_doorbell_press(self):
        """Simulate doorbell button press (for testing/web interface)"""
        if self.doorbell_callback:
            logger.info("Simulating doorbell press")
            threading.Thread(
                target=self.doorbell_callback,
                args=(self.settings.DOORBELL_PIN,),
                daemon=True
            ).start()
        else:
            logger.warning("No doorbell callback configured")
    
    def set_status_led(self, status: str, duration: Optional[float] = None):
        """Set status LED based on system state"""
        if not self.initialized:
            return
        
        try:
            # Define LED patterns for different states
            led_patterns = {
                'idle': {'green': True, 'yellow': False, 'red': False},
                'processing': {'green': False, 'yellow': True, 'red': False},
                'known': {'green': True, 'yellow': False, 'red': False},
                'unknown': {'green': False, 'yellow': True, 'red': False},
                'alert': {'green': False, 'yellow': False, 'red': True},
                'error': {'green': False, 'yellow': True, 'red': True},
                'off': {'green': False, 'yellow': False, 'red': False}
            }
            
            pattern = led_patterns.get(status, led_patterns['off'])
            
            # Set LEDs
            for color, state in pattern.items():
                if color in self.settings.STATUS_LED_PINS:
                    if self.use_real_gpio:
                        pin = self.settings.STATUS_LED_PINS[color]
                        GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)
                    self.led_states[color] = state
            
            logger.debug(f"Status LED set to: {status}")
            
            # Auto-reset after duration
            if duration:
                threading.Timer(duration, lambda: self.set_status_led('idle')).start()
            
        except Exception as e:
            logger.error(f"Failed to set status LED: {e}")
    
    def blink_led(self, color: str, times: int = 3, interval: float = 0.5):
        """Blink specific LED"""
        if not self.initialized or color not in self.settings.STATUS_LED_PINS:
            return
        
        def blink_thread():
            try:
                original_state = self.led_states.get(color, False)
                
                for _ in range(times):
                    if self.use_real_gpio:
                        pin = self.settings.STATUS_LED_PINS[color]
                        GPIO.output(pin, GPIO.HIGH)
                        time.sleep(interval / 2)
                        GPIO.output(pin, GPIO.LOW)
                        time.sleep(interval / 2)
                    else:
                        # Mock blink - just update state
                        self.led_states[color] = True
                        time.sleep(interval / 2)
                        self.led_states[color] = False
                        time.sleep(interval / 2)
                
                # Restore original state
                if self.use_real_gpio:
                    pin = self.settings.STATUS_LED_PINS[color]
                    GPIO.output(pin, GPIO.HIGH if original_state else GPIO.LOW)
                self.led_states[color] = original_state
                
            except Exception as e:
                logger.error(f"LED blink error: {e}")
        
        threading.Thread(target=blink_thread, daemon=True).start()
    
    def _start_led_thread(self):
        """Start LED control thread for animations"""
        self.led_thread_running = True
        self.led_thread = threading.Thread(target=self._led_control_loop, daemon=True)
        self.led_thread.start()
    
    def _led_control_loop(self):
        """LED control loop for animations and effects"""
        while self.led_thread_running:
            try:
                # Heartbeat effect on green LED when idle
                if all(not state for state in self.led_states.values()):
                    self._heartbeat_led('green')
                
                time.sleep(2)  # Heartbeat every 2 seconds
                
            except Exception as e:
                logger.error(f"LED control loop error: {e}")
                time.sleep(1)
    
    def _heartbeat_led(self, color: str):
        """Create heartbeat effect on LED"""
        if not self.initialized or color not in self.settings.STATUS_LED_PINS:
            return
        
        try:
            if self.use_real_gpio:
                pin = self.settings.STATUS_LED_PINS[color]
                
                # Quick double blink
                GPIO.output(pin, GPIO.HIGH)
                time.sleep(0.1)
                GPIO.output(pin, GPIO.LOW)
                time.sleep(0.1)
                GPIO.output(pin, GPIO.HIGH)
                time.sleep(0.1)
                GPIO.output(pin, GPIO.LOW)
            else:
                # Mock heartbeat - just log
                logger.debug(f"Mock heartbeat: {color} LED")
            
        except Exception as e:
            logger.error(f"Heartbeat LED error: {e}")
    
    def test_leds(self):
        """Test all LEDs in sequence"""
        if not self.initialized:
            logger.error("GPIO not initialized")
            return
        
        logger.info("Testing LEDs...")
        
        for color in ['red', 'yellow', 'green']:
            if color in self.settings.STATUS_LED_PINS:
                logger.info(f"Testing {color} LED")
                self.set_status_led('off')
                time.sleep(0.5)
                
                if self.use_real_gpio:
                    pin = self.settings.STATUS_LED_PINS[color]
                    GPIO.output(pin, GPIO.HIGH)
                    time.sleep(1)
                    GPIO.output(pin, GPIO.LOW)
                else:
                    # Mock LED test
                    self.led_states[color] = True
                    time.sleep(1)
                    self.led_states[color] = False
        
        # Return to idle state
        self.set_status_led('idle')
        logger.info("LED test completed")
    
    def get_gpio_status(self) -> dict:
        """Get GPIO status information"""
        status = {
            'initialized': self.initialized,
            'doorbell_pin': self.settings.DOORBELL_PIN,
            'led_pins': self.settings.STATUS_LED_PINS.copy(),
            'led_states': self.led_states.copy()
        }
        
        if self.initialized and self.use_real_gpio:
            try:
                # Read current doorbell state
                status['doorbell_state'] = GPIO.input(self.settings.DOORBELL_PIN)
            except:
                status['doorbell_state'] = 'unknown'
        elif self.initialized:
            status['doorbell_state'] = 'mock'
        
        return status
    
    def cleanup(self):
        """Cleanup GPIO resources"""
        try:
            # Stop LED thread
            self.led_thread_running = False
            if self.led_thread and self.led_thread.is_alive():
                self.led_thread.join(timeout=1)
            
            if self.use_real_gpio and GPIO_AVAILABLE and self.initialized:
                # Remove event detection
                if hasattr(GPIO, 'remove_event_detect'):
                    try:
                        GPIO.remove_event_detect(self.settings.DOORBELL_PIN)
                    except:
                        pass
                
                # Turn off all LEDs
                for pin in self.settings.STATUS_LED_PINS.values():
                    try:
                        GPIO.output(pin, GPIO.LOW)
                    except:
                        pass
                
                # Cleanup GPIO
                GPIO.cleanup()
            
            self.initialized = False
            logger.info("GPIO cleanup completed")
            
        except Exception as e:
            logger.error(f"GPIO cleanup error: {e}")


class MockGPIOHandler(GPIOHandler):
    """Mock GPIO handler for testing without Raspberry Pi"""
    
    def __init__(self):
        from config.settings import Settings
        self.settings = Settings()
        
        self.initialized = True
        self.doorbell_callback = None
        self.led_states = {color: False for color in self.settings.STATUS_LED_PINS.keys()}
        self.button_pressed_count = 0
        
        logger.info("Mock GPIO handler initialized")
    
    def setup_doorbell_button(self, callback: Callable):
        """Mock doorbell button setup"""
        self.doorbell_callback = callback
        logger.info("Mock doorbell button configured")
    
    def simulate_doorbell_press(self):
        """Simulate doorbell button press for testing"""
        self.button_pressed_count += 1
        logger.info(f"Simulating doorbell press #{self.button_pressed_count}")
        
        if self.doorbell_callback:
            threading.Thread(
                target=self.doorbell_callback,
                args=(self.settings.DOORBELL_PIN,),
                daemon=True
            ).start()
    
    def set_status_led(self, status: str, duration: Optional[float] = None):
        """Mock LED control"""
        led_patterns = {
            'idle': {'green': True, 'yellow': False, 'red': False},
            'processing': {'green': False, 'yellow': True, 'red': False},
            'known': {'green': True, 'yellow': False, 'red': False},
            'unknown': {'green': False, 'yellow': True, 'red': False},
            'alert': {'green': False, 'yellow': False, 'red': True},
            'error': {'green': False, 'yellow': True, 'red': True},
            'off': {'green': False, 'yellow': False, 'red': False}
        }
        
        pattern = led_patterns.get(status, led_patterns['off'])
        self.led_states.update(pattern)
        
        logger.debug(f"Mock LED status: {status} - {pattern}")
        
        if duration:
            threading.Timer(duration, lambda: self.set_status_led('idle')).start()
    
    def test_leds(self):
        """Mock LED test"""
        logger.info("Mock LED test - all LEDs OK")
        for color in ['red', 'yellow', 'green']:
            self.led_states[color] = True
            time.sleep(0.5)
            self.led_states[color] = False
        self.set_status_led('idle')
    
    def cleanup(self):
        """Mock cleanup"""
        self.initialized = False
        logger.info("Mock GPIO cleanup completed")
