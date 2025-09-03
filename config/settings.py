"""
Configuration settings for the Doorbell Security System
"""

import os
from pathlib import Path


class Settings:
    """System configuration settings"""
    
    def __init__(self):
        # Project paths
        self.PROJECT_ROOT = Path(__file__).parent.parent
        self.DATA_DIR = self.PROJECT_ROOT / 'data'
        self.KNOWN_FACES_DIR = self.DATA_DIR / 'known_faces'
        self.BLACKLIST_FACES_DIR = self.DATA_DIR / 'blacklist_faces'
        self.CAPTURES_DIR = self.DATA_DIR / 'captures'
        self.LOGS_DIR = self.DATA_DIR / 'logs'
        
        # Ensure directories exist
        for directory in [self.DATA_DIR, self.KNOWN_FACES_DIR, 
                         self.BLACKLIST_FACES_DIR, self.CAPTURES_DIR, self.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # GPIO Configuration
        self.DOORBELL_PIN = int(os.getenv('DOORBELL_PIN', 18))
        self.STATUS_LED_PINS = {
            'red': int(os.getenv('RED_LED_PIN', 16)),
            'yellow': int(os.getenv('YELLOW_LED_PIN', 20)),
            'green': int(os.getenv('GREEN_LED_PIN', 21))
        }
        self.DEBOUNCE_TIME = float(os.getenv('DEBOUNCE_TIME', 2.0))
        
        # Camera Configuration
        self.CAMERA_RESOLUTION = tuple(map(int, os.getenv('CAMERA_RESOLUTION', '1280,720').split(',')))
        self.CAMERA_ROTATION = int(os.getenv('CAMERA_ROTATION', 0))
        self.CAMERA_BRIGHTNESS = int(os.getenv('CAMERA_BRIGHTNESS', 50))
        self.CAMERA_CONTRAST = int(os.getenv('CAMERA_CONTRAST', 0))
        
        # Face Recognition Configuration
        self.FACE_RECOGNITION_TOLERANCE = float(os.getenv('FACE_TOLERANCE', 0.6))
        self.BLACKLIST_TOLERANCE = float(os.getenv('BLACKLIST_TOLERANCE', 0.5))
        self.MIN_FACE_SIZE = int(os.getenv('MIN_FACE_SIZE', 50))
        self.REQUIRE_FACE_FOR_CAPTURE = os.getenv('REQUIRE_FACE_FOR_CAPTURE', 'True').lower() == 'true'
        
        # Notification Configuration
        self.NOTIFY_ON_KNOWN_VISITORS = os.getenv('NOTIFY_ON_KNOWN_VISITORS', 'False').lower() == 'true'
        self.INCLUDE_PHOTO_FOR_KNOWN = os.getenv('INCLUDE_PHOTO_FOR_KNOWN', 'False').lower() == 'true'
        
        # FBI API Configuration
        self.FBI_API_URL = os.getenv('FBI_API_URL', 'https://api.fbi.gov/wanted/v1/list')
        self.FBI_UPDATE_ENABLED = os.getenv('FBI_UPDATE_ENABLED', 'True').lower() == 'true'
        self.FBI_MAX_ENTRIES = int(os.getenv('FBI_MAX_ENTRIES', 50))
        
        # System Configuration
        self.MAX_CAPTURE_ATTEMPTS = int(os.getenv('MAX_CAPTURE_ATTEMPTS', 3))
        self.CAPTURE_RETRY_DELAY = float(os.getenv('CAPTURE_RETRY_DELAY', 0.5))
        
        # Logging Configuration
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.MAX_LOG_FILE_SIZE = int(os.getenv('MAX_LOG_FILE_SIZE', 10485760))  # 10MB
        self.LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', 5))
