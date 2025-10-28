#!/usr/bin/env python3
"""
Legacy Adapter - Compatibility Layer

Provides backward compatibility with the existing DoorbellSecuritySystem API
while using the new pipeline orchestrator internally.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.pipeline.orchestrator import PipelineOrchestrator
from src.communication.events import PipelineEvent, EventType

logger = logging.getLogger(__name__)


class LegacyAdapter:
    """
    Adapter to maintain compatibility with legacy DoorbellSecuritySystem API.
    
    This allows existing code (especially web interface) to continue working
    without modification while using the new pipeline architecture internally.
    """
    
    def __init__(self, orchestrator: PipelineOrchestrator):
        """Initialize legacy adapter."""
        self.orchestrator = orchestrator
        self.last_trigger_time = 0
        self.running = False
        
        # Emulate legacy attributes that web interface expects
        self.settings = self._create_settings_proxy()
        self.face_manager = self._create_face_manager_proxy()
        self.camera = self._create_camera_proxy()
        self.gpio = self._create_gpio_proxy()
        
        logger.info("Legacy adapter initialized")
    
    def start(self) -> None:
        """Start the system (delegates to orchestrator)."""
        if not self.orchestrator.running:
            self.orchestrator.start_time = time.time()
            self.orchestrator.start()
        self.running = True
        logger.info("Legacy adapter started")
    
    def stop(self) -> None:
        """Stop the system (delegates to orchestrator)."""
        self.running = False
        self.orchestrator.shutdown()
        logger.info("Legacy adapter stopped")
    
    def on_doorbell_pressed(self, channel=None) -> None:
        """Handle doorbell press (legacy interface)."""
        current_time = time.time()
        
        # Simple debounce
        if current_time - self.last_trigger_time < 5.0:
            logger.debug("Ignoring rapid doorbell press (debounce)")
            return
        
        self.last_trigger_time = current_time
        
        # Create pipeline event
        event = PipelineEvent(
            event_type=EventType.DOORBELL_PRESSED,
            data={
                "trigger_source": "gpio",
                "channel": channel,
                "timestamp": current_time
            },
            source="legacy_adapter"
        )
        
        # Publish to pipeline
        self.orchestrator.message_bus.publish('doorbell_events', event)
        logger.info("Doorbell event published to pipeline")
    
    def get_recent_captures(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent capture events (legacy interface)."""
        try:
            # Query event database through orchestrator
            events = self.orchestrator.event_database.get_recent_events(
                event_type='face_recognition',
                limit=limit
            )
            return events if events else []
        except Exception as e:
            logger.error(f"Failed to get recent captures: {e}")
            return []
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status (legacy interface)."""
        try:
            metrics = self.orchestrator.get_pipeline_metrics()
            
            return {
                "status": "running" if self.running else "stopped",
                "uptime": metrics.get('uptime', 0),
                "pipeline_health": metrics.get('pipeline_status', 'unknown'),
                "events_processed": metrics.get('events_per_minute', 0),
                "last_activity": self.last_trigger_time
            }
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _create_settings_proxy(self):
        """Create proxy object for settings access."""
        class SettingsProxy:
            def __init__(self, orchestrator):
                self.orchestrator = orchestrator
                # Delegate to orchestrator config
                config = orchestrator.config
                
                # Common legacy attributes
                self.DEBOUNCE_TIME = 5.0
                self.CAPTURES_DIR = Path(getattr(config.storage, 'capture_path', 'data/captures'))
                self.CROPPED_FACES_DIR = Path('data/cropped_faces')
                self.KNOWN_FACES_DIR = Path(getattr(config.face_recognition, 'known_faces_path', 'data/known_faces'))
                self.BLACKLIST_FACES_DIR = Path(getattr(config.face_recognition, 'blacklist_faces_path', 'data/blacklist_faces'))
                self.LOGS_DIR = Path(getattr(config.storage, 'log_path', 'data/logs'))
                
                # Additional settings
                self.REQUIRE_FACE_FOR_CAPTURE = False
                self.NOTIFY_ON_KNOWN_VISITORS = True
                self.INCLUDE_PHOTO_FOR_KNOWN = False
                
                # Ensure directories exist
                for dir_path in [self.CAPTURES_DIR, self.CROPPED_FACES_DIR, 
                               self.KNOWN_FACES_DIR, self.BLACKLIST_FACES_DIR, self.LOGS_DIR]:
                    dir_path.mkdir(parents=True, exist_ok=True)
        
        return SettingsProxy(self.orchestrator)
    
    def _create_face_manager_proxy(self):
        """Create proxy object for face manager access."""
        class FaceManagerProxy:
            def __init__(self, orchestrator):
                self.orchestrator = orchestrator
            
            def load_known_faces(self):
                """Delegate to pipeline face database."""
                # Pipeline handles this automatically
                logger.debug("load_known_faces called (handled by pipeline)")
                pass
            
            def load_blacklist_faces(self):
                """Delegate to pipeline face database."""
                # Pipeline handles this automatically
                logger.debug("load_blacklist_faces called (handled by pipeline)")
                pass
            
            def get_known_faces_count(self) -> int:
                """Get count of known faces."""
                try:
                    # This would need to be implemented in event_database
                    # For now return a placeholder
                    return 0
                except Exception:
                    return 0
            
            def detect_faces(self, image):
                """Detect faces in image (compatibility method)."""
                # This is handled by the pipeline, but provide stub for compatibility
                logger.debug("detect_faces called (compatibility stub)")
                return []
            
            def detect_face_locations(self, image):
                """Detect face locations (compatibility method)."""
                logger.debug("detect_face_locations called (compatibility stub)")
                return []
            
            def identify_face(self, face_encoding):
                """Identify face (compatibility method)."""
                logger.debug("identify_face called (compatibility stub)")
                return {'status': 'unknown', 'name': None}
        
        return FaceManagerProxy(self.orchestrator)
    
    def _create_camera_proxy(self):
        """Create proxy object for camera access."""
        class CameraProxy:
            def __init__(self, orchestrator):
                self.orchestrator = orchestrator
            
            def initialize(self):
                """Camera initialization handled by pipeline."""
                logger.debug("Camera initialize called (handled by pipeline)")
                pass
            
            def capture_image(self):
                """Legacy capture interface."""
                # This would need to be implemented if legacy code calls it directly
                # For now, pipeline handles all captures
                logger.debug("capture_image called (handled by pipeline)")
                return None
            
            def save_image(self, image, path):
                """Save image to path."""
                logger.debug(f"save_image called for {path}")
                # Could implement if needed
                pass
            
            def cleanup(self):
                """Camera cleanup handled by pipeline."""
                logger.debug("Camera cleanup called (handled by pipeline)")
                pass
        
        return CameraProxy(self.orchestrator)
    
    def _create_gpio_proxy(self):
        """Create proxy object for GPIO access."""
        class GPIOProxy:
            def __init__(self, orchestrator):
                self.orchestrator = orchestrator
                self.callback = None
            
            def setup_doorbell_button(self, callback):
                """GPIO setup handled by pipeline."""
                # Store callback for potential use
                self.callback = callback
                logger.debug("setup_doorbell_button called (handled by pipeline)")
            
            def simulate_doorbell_press(self):
                """Simulate doorbell press for testing."""
                # Delegate to orchestrator manager
                event = PipelineEvent(
                    event_type=EventType.DOORBELL_PRESSED,
                    data={"trigger_source": "simulation", "timestamp": time.time()},
                    source="gpio_simulation"
                )
                self.orchestrator.message_bus.publish('doorbell_events', event)
                logger.info("Simulated doorbell press")
            
            def set_status_led(self, status):
                """Set status LED (compatibility method)."""
                logger.debug(f"set_status_led called with status: {status}")
                # Could implement if needed
                pass
            
            def cleanup(self):
                """GPIO cleanup handled by pipeline."""
                logger.debug("GPIO cleanup called (handled by pipeline)")
                pass
        
        return GPIOProxy(self.orchestrator)
