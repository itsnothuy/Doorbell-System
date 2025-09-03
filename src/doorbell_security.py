#!/usr/bin/env python3
"""
Doorbell Face Recognition Security System
Main application entry point

This system captures visitor images when doorbell is pressed,
performs face recognition, and sends appropriate alerts.
"""

import sys
import os
import time
import logging
import threading
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.face_manager import FaceManager
from src.camera_handler import CameraHandler
from src.telegram_notifier import TelegramNotifier
from src.gpio_handler import GPIOHandler
from src.platform_detector import platform_detector
from config.settings import Settings

# Import web interface for development mode
if platform_detector.get_gpio_config().get('web_interface', False):
    from src.web_interface import WebInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'data/logs/doorbell.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DoorbellSecuritySystem:
    """Main doorbell security system controller"""
    
    def __init__(self):
        """Initialize the security system"""
        logger.info("Initializing Doorbell Security System...")
        
        # Load configuration
        self.settings = Settings()
        
        # Initialize components
        self.face_manager = FaceManager()
        self.camera = CameraHandler()
        self.notifier = TelegramNotifier()
        self.gpio = GPIOHandler()
        
        # System state
        self.running = False
        self.processing_lock = threading.Lock()
        self.last_trigger_time = 0
        
        # Web interface for development/testing
        self.web_interface = None
        if platform_detector.get_gpio_config().get('web_interface', False):
            try:
                self.web_interface = WebInterface(self)
                logger.info("Web interface initialized for testing")
            except Exception as e:
                logger.warning(f"Web interface initialization failed: {e}")
        
        logger.info("System initialized successfully")
    
    def start(self):
        """Start the security system"""
        try:
            logger.info("Starting Doorbell Security System...")
            
            # Load face databases
            self.face_manager.load_known_faces()
            self.face_manager.load_blacklist_faces()
            
            # Initialize hardware
            self.camera.initialize()
            self.gpio.setup_doorbell_button(self.on_doorbell_pressed)
            
            # Start periodic tasks
            self._start_maintenance_tasks()
            
            self.running = True
            logger.info("✅ System is ready and monitoring...")
            
            # Start web interface if available
            if self.web_interface:
                deployment_config = platform_detector.get_deployment_config()
                web_thread = threading.Thread(
                    target=self.web_interface.run,
                    kwargs={
                        'host': '0.0.0.0' if deployment_config['platform'] != 'local' else '127.0.0.1',
                        'port': deployment_config['port'],
                        'debug': deployment_config['platform'] == 'local'
                    },
                    daemon=True
                )
                web_thread.start()
                logger.info(f"🌐 Web interface started on port {deployment_config['port']}")
            
            # Main loop
            try:
                while self.running:
                    time.sleep(0.1)  # Prevent busy waiting
            except KeyboardInterrupt:
                logger.info("Shutdown requested by user")
                
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            raise
        finally:
            self.stop()
    
    def stop(self):
        """Stop the security system"""
        logger.info("Shutting down system...")
        self.running = False
        
        # Cleanup components
        if hasattr(self, 'gpio'):
            self.gpio.cleanup()
        if hasattr(self, 'camera'):
            self.camera.cleanup()
            
        logger.info("System shutdown complete")
    
    def on_doorbell_pressed(self, channel):
        """Handle doorbell button press event"""
        current_time = time.time()
        
        # Debounce: ignore rapid successive presses
        if current_time - self.last_trigger_time < self.settings.DEBOUNCE_TIME:
            logger.debug("Ignoring rapid doorbell press (debounce)")
            return
        
        self.last_trigger_time = current_time
        
        # Process in separate thread to avoid blocking GPIO
        threading.Thread(
            target=self._process_visitor,
            args=(current_time,),
            daemon=True
        ).start()
    
    def _process_visitor(self, trigger_time):
        """Process visitor detection and recognition"""
        if not self.processing_lock.acquire(blocking=False):
            logger.warning("Previous visitor processing still in progress, skipping")
            return
        
        try:
            logger.info("🔔 Doorbell pressed! Processing visitor...")
            timestamp = datetime.fromtimestamp(trigger_time)
            
            # Turn on processing indicator
            self.gpio.set_status_led('processing')
            
            # Capture visitor image
            image = self._capture_visitor_image()
            if image is None:
                logger.warning("Failed to capture visitor image")
                self._handle_capture_failure(timestamp)
                return
            
            # Save capture for logging
            capture_path = self._save_capture(image, timestamp)
            
            # Detect faces in image
            faces = self.face_manager.detect_faces(image)
            if not faces:
                logger.info("No faces detected in image")
                self._handle_no_faces(capture_path, timestamp)
                return
            
            # Analyze each detected face
            results = []
            for i, face_encoding in enumerate(faces):
                result = self.face_manager.identify_face(face_encoding)
                results.append(result)
                logger.info(f"Face {i+1}: {result['status']} - {result.get('name', 'Unknown')}")
            
            # Determine overall response based on all faces
            response = self._determine_response(results)
            
            # Take appropriate action
            self._handle_visitor_response(response, capture_path, timestamp, results)
            
        except Exception as e:
            logger.error(f"Error processing visitor: {e}")
            self.gpio.set_status_led('error')
            
        finally:
            # Reset status
            self.gpio.set_status_led('idle')
            self.processing_lock.release()
    
    def _capture_visitor_image(self):
        """Capture image of visitor with retry logic"""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                logger.debug(f"Capture attempt {attempt + 1}/{max_attempts}")
                
                # Capture image
                image = self.camera.capture_image()
                
                # Quick face detection to ensure we have a visitor
                if self.settings.REQUIRE_FACE_FOR_CAPTURE:
                    face_locations = self.face_manager.detect_face_locations(image)
                    if face_locations:
                        logger.debug("Face detected in capture")
                        return image
                    else:
                        logger.debug("No face in capture, retrying...")
                        time.sleep(0.5)  # Brief pause before retry
                        continue
                else:
                    return image
                    
            except Exception as e:
                logger.warning(f"Capture attempt {attempt + 1} failed: {e}")
                time.sleep(0.5)
        
        return None
    
    def _save_capture(self, image, timestamp):
        """Save captured image with timestamp"""
        filename = f"visitor_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        capture_path = self.settings.CAPTURES_DIR / filename
        
        try:
            self.camera.save_image(image, capture_path)
            logger.debug(f"Saved capture: {capture_path}")
            return capture_path
        except Exception as e:
            logger.error(f"Failed to save capture: {e}")
            return None
    
    def _determine_response(self, face_results):
        """Determine system response based on face analysis results"""
        has_blacklisted = any(r['status'] == 'blacklisted' for r in face_results)
        has_unknown = any(r['status'] == 'unknown' for r in face_results)
        has_known = any(r['status'] == 'known' for r in face_results)
        
        if has_blacklisted:
            return 'blacklist_alert'
        elif has_unknown:
            return 'unknown_alert'
        elif has_known:
            return 'known_person'
        else:
            return 'no_faces'
    
    def _handle_visitor_response(self, response, capture_path, timestamp, face_results):
        """Handle visitor based on recognition results"""
        if response == 'blacklist_alert':
            self._handle_blacklist_alert(capture_path, timestamp, face_results)
        elif response == 'unknown_alert':
            self._handle_unknown_visitor(capture_path, timestamp, face_results)
        elif response == 'known_person':
            self._handle_known_visitor(capture_path, timestamp, face_results)
        else:
            self._handle_no_faces(capture_path, timestamp)
    
    def _handle_blacklist_alert(self, capture_path, timestamp, face_results):
        """Handle blacklisted person detection"""
        logger.warning("🚨 BLACKLISTED PERSON DETECTED!")
        
        # Find blacklisted person details
        blacklisted_person = next(
            (r for r in face_results if r['status'] == 'blacklisted'), 
            None
        )
        
        # Set urgent visual indicator
        self.gpio.set_status_led('alert')
        
        # Send urgent notification
        message = f"🚨 SECURITY ALERT!\n\n"
        message += f"Potential match: {blacklisted_person.get('name', 'Unknown threat')}\n"
        message += f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        message += "⚠️ This is an automated alert. Verify manually before taking action."
        
        self.notifier.send_alert(
            message=message,
            image_path=capture_path,
            priority='urgent'
        )
        
        # Log event
        self._log_event('BLACKLIST_ALERT', timestamp, face_results, capture_path)
    
    def _handle_unknown_visitor(self, capture_path, timestamp, face_results):
        """Handle unknown person detection"""
        logger.info("🟡 Unknown visitor detected")
        
        # Set status indicator
        self.gpio.set_status_led('unknown')
        
        # Send notification
        message = f"🔔 Unknown visitor at the door\n"
        message += f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        message += "Please check the attached photo."
        
        self.notifier.send_alert(
            message=message,
            image_path=capture_path,
            priority='normal'
        )
        
        # Log event
        self._log_event('UNKNOWN_VISITOR', timestamp, face_results, capture_path)
    
    def _handle_known_visitor(self, capture_path, timestamp, face_results):
        """Handle known person detection"""
        known_names = [r['name'] for r in face_results if r['status'] == 'known']
        logger.info(f"🟢 Known visitor(s): {', '.join(known_names)}")
        
        # Set status indicator
        self.gpio.set_status_led('known')
        
        # Optionally send notification for known visitors
        if self.settings.NOTIFY_ON_KNOWN_VISITORS:
            message = f"🏠 {', '.join(known_names)} arrived\n"
            message += f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            
            self.notifier.send_alert(
                message=message,
                image_path=capture_path if self.settings.INCLUDE_PHOTO_FOR_KNOWN else None,
                priority='low'
            )
        
        # Log event
        self._log_event('KNOWN_VISITOR', timestamp, face_results, capture_path)
    
    def _handle_no_faces(self, capture_path, timestamp):
        """Handle case where no faces are detected"""
        logger.info("👻 Doorbell pressed but no faces detected")
        
        # Send notification
        message = f"🔔 Doorbell pressed\n"
        message += f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        message += "No faces detected in image."
        
        self.notifier.send_alert(
            message=message,
            image_path=capture_path,
            priority='low'
        )
        
        # Log event
        self._log_event('NO_FACES', timestamp, [], capture_path)
    
    def _handle_capture_failure(self, timestamp):
        """Handle camera capture failure"""
        logger.error("📷 Failed to capture visitor image")
        
        message = f"⚠️ Camera Error\n"
        message += f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        message += "Doorbell was pressed but camera failed to capture image."
        
        self.notifier.send_alert(
            message=message,
            image_path=None,
            priority='normal'
        )
        
        # Log event
        self._log_event('CAPTURE_FAILURE', timestamp, [], None)
    
    def _log_event(self, event_type, timestamp, face_results, capture_path):
        """Log visitor event to file"""
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'event_type': event_type,
            'faces': face_results,
            'capture_path': str(capture_path) if capture_path else None
        }
        
        # Write to events log
        events_log = self.settings.LOGS_DIR / 'events.log'
        with open(events_log, 'a') as f:
            f.write(f"{log_entry}\n")
    
    def _start_maintenance_tasks(self):
        """Start background maintenance tasks"""
        # Schedule blacklist updates
        def update_blacklist():
            try:
                logger.info("Running scheduled blacklist update...")
                self.face_manager.update_blacklist_from_fbi()
                logger.info("Blacklist update completed")
            except Exception as e:
                logger.error(f"Blacklist update failed: {e}")
        
        # Run updates in background thread
        threading.Thread(
            target=self._run_scheduler,
            args=(update_blacklist,),
            daemon=True
        ).start()
    
    def _run_scheduler(self, update_func):
        """Run scheduled maintenance tasks"""
        import schedule
        
        # Schedule daily blacklist updates
        schedule.every().day.at("02:00").do(update_func)
        
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


def main():
    """Main entry point"""
    try:
        system = DoorbellSecuritySystem()
        system.start()
    except KeyboardInterrupt:
        logger.info("System stopped by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
