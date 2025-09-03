"""
Face recognition and database management module
"""

import os
import pickle
import logging
import requests
import face_recognition
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image

logger = logging.getLogger(__name__)


class FaceManager:
    """Manages face recognition and face databases"""
    
    def __init__(self):
        from config.settings import Settings
        self.settings = Settings()
        
        # Face databases
        self.known_encodings = []
        self.known_names = []
        self.blacklist_encodings = []
        self.blacklist_names = []
        
        # Cache files for faster loading
        self.known_cache_file = self.settings.DATA_DIR / 'known_faces_cache.pkl'
        self.blacklist_cache_file = self.settings.DATA_DIR / 'blacklist_faces_cache.pkl'
        
        logger.info("Face Manager initialized")
    
    def load_known_faces(self):
        """Load known faces from directory or cache"""
        try:
            # Try to load from cache first
            if self._load_known_from_cache():
                logger.info(f"Loaded {len(self.known_names)} known faces from cache")
                return
            
            # Load from images
            self._load_known_from_images()
            self._save_known_to_cache()
            
            logger.info(f"Loaded {len(self.known_names)} known faces from images")
            
        except Exception as e:
            logger.error(f"Failed to load known faces: {e}")
            raise
    
    def load_blacklist_faces(self):
        """Load blacklist faces from directory or cache"""
        try:
            # Try to load from cache first
            if self._load_blacklist_from_cache():
                logger.info(f"Loaded {len(self.blacklist_names)} blacklist faces from cache")
                return
            
            # Load from images
            self._load_blacklist_from_images()
            self._save_blacklist_to_cache()
            
            logger.info(f"Loaded {len(self.blacklist_names)} blacklist faces from images")
            
        except Exception as e:
            logger.error(f"Failed to load blacklist faces: {e}")
            # Don't raise - system can work without blacklist
    
    def detect_face_locations(self, image) -> List[Tuple]:
        """Detect face locations in image"""
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            # Detect faces using HOG (faster on CPU)
            face_locations = face_recognition.face_locations(
                image_array,
                model="hog"  # Use HOG for speed on Raspberry Pi
            )
            
            return face_locations
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def detect_faces(self, image) -> List[np.ndarray]:
        """Detect faces and return encodings"""
        try:
            # Get face locations
            face_locations = self.detect_face_locations(image)
            
            if not face_locations:
                return []
            
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            # Generate face encodings
            face_encodings = face_recognition.face_encodings(
                image_array,
                known_face_locations=face_locations
            )
            
            return face_encodings
            
        except Exception as e:
            logger.error(f"Face encoding failed: {e}")
            return []
    
    def identify_face(self, face_encoding: np.ndarray) -> Dict:
        """Identify a face encoding against known databases"""
        result = {
            'status': 'unknown',
            'name': None,
            'confidence': 0.0,
            'distance': float('inf')
        }
        
        try:
            # Check against blacklist first (higher priority)
            if self.blacklist_encodings:
                blacklist_distances = face_recognition.face_distance(
                    self.blacklist_encodings, 
                    face_encoding
                )
                
                min_blacklist_distance = min(blacklist_distances)
                if min_blacklist_distance <= self.settings.BLACKLIST_TOLERANCE:
                    best_match_index = np.argmin(blacklist_distances)
                    result = {
                        'status': 'blacklisted',
                        'name': self.blacklist_names[best_match_index],
                        'confidence': 1.0 - min_blacklist_distance,
                        'distance': min_blacklist_distance
                    }
                    return result
            
            # Check against known faces
            if self.known_encodings:
                known_distances = face_recognition.face_distance(
                    self.known_encodings, 
                    face_encoding
                )
                
                min_known_distance = min(known_distances)
                if min_known_distance <= self.settings.FACE_RECOGNITION_TOLERANCE:
                    best_match_index = np.argmin(known_distances)
                    result = {
                        'status': 'known',
                        'name': self.known_names[best_match_index],
                        'confidence': 1.0 - min_known_distance,
                        'distance': min_known_distance
                    }
                    return result
            
            # If no matches found, return unknown
            result['status'] = 'unknown'
            return result
            
        except Exception as e:
            logger.error(f"Face identification failed: {e}")
            return result
    
    def add_known_face(self, image_path: Path, name: str) -> bool:
        """Add a new known face to the database"""
        try:
            # Load and encode the image
            image = face_recognition.load_image_file(str(image_path))
            encodings = face_recognition.face_encodings(image)
            
            if not encodings:
                logger.warning(f"No face found in {image_path}")
                return False
            
            # Use the first face found
            encoding = encodings[0]
            
            # Add to known faces
            self.known_encodings.append(encoding)
            self.known_names.append(name)
            
            # Update cache
            self._save_known_to_cache()
            
            logger.info(f"Added known face: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add known face {name}: {e}")
            return False
    
    def remove_known_face(self, name: str) -> bool:
        """Remove a known face from the database"""
        try:
            if name in self.known_names:
                index = self.known_names.index(name)
                del self.known_names[index]
                del self.known_encodings[index]
                
                # Update cache
                self._save_known_to_cache()
                
                logger.info(f"Removed known face: {name}")
                return True
            else:
                logger.warning(f"Known face not found: {name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove known face {name}: {e}")
            return False
    
    def update_blacklist_from_fbi(self) -> bool:
        """Update blacklist from FBI Most Wanted API"""
        if not self.settings.FBI_UPDATE_ENABLED:
            logger.info("FBI updates disabled in settings")
            return False
        
        try:
            logger.info("Fetching FBI Most Wanted list...")
            
            # Fetch data from FBI API
            response = requests.get(self.settings.FBI_API_URL, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            items = data.get('items', [])
            
            # Limit number of entries
            items = items[:self.settings.FBI_MAX_ENTRIES]
            
            updated_count = 0
            for item in items:
                try:
                    name = item.get('title', 'Unknown')
                    images = item.get('images', [])
                    
                    if not images:
                        continue
                    
                    # Download first image
                    image_url = images[0].get('original')
                    if not image_url:
                        continue
                    
                    # Download and save image
                    image_filename = f"fbi_{name.replace(' ', '_').lower()}.jpg"
                    image_path = self.settings.BLACKLIST_FACES_DIR / image_filename
                    
                    # Skip if already exists
                    if image_path.exists():
                        continue
                    
                    # Download image
                    img_response = requests.get(image_url, timeout=15)
                    img_response.raise_for_status()
                    
                    with open(image_path, 'wb') as f:
                        f.write(img_response.content)
                    
                    updated_count += 1
                    logger.debug(f"Downloaded: {name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to process FBI entry: {e}")
                    continue
            
            if updated_count > 0:
                logger.info(f"Downloaded {updated_count} new FBI entries")
                # Reload blacklist to include new images
                self._load_blacklist_from_images()
                self._save_blacklist_to_cache()
            else:
                logger.info("No new FBI entries to download")
            
            return True
            
        except Exception as e:
            logger.error(f"FBI update failed: {e}")
            return False
    
    def _load_known_from_images(self):
        """Load known faces from image files"""
        self.known_encodings = []
        self.known_names = []
        
        for image_path in self.settings.KNOWN_FACES_DIR.glob("*.jpg"):
            try:
                # Load image
                image = face_recognition.load_image_file(str(image_path))
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    self.known_encodings.append(encodings[0])
                    # Use filename (without extension) as name
                    name = image_path.stem
                    self.known_names.append(name)
                    logger.debug(f"Loaded known face: {name}")
                else:
                    logger.warning(f"No face found in {image_path}")
                    
            except Exception as e:
                logger.error(f"Failed to load {image_path}: {e}")
    
    def _load_blacklist_from_images(self):
        """Load blacklist faces from image files"""
        self.blacklist_encodings = []
        self.blacklist_names = []
        
        for image_path in self.settings.BLACKLIST_FACES_DIR.glob("*.jpg"):
            try:
                # Load image
                image = face_recognition.load_image_file(str(image_path))
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    self.blacklist_encodings.append(encodings[0])
                    # Use filename (without extension) as name
                    name = image_path.stem.replace('fbi_', '').replace('_', ' ').title()
                    self.blacklist_names.append(name)
                    logger.debug(f"Loaded blacklist face: {name}")
                else:
                    logger.warning(f"No face found in {image_path}")
                    
            except Exception as e:
                logger.error(f"Failed to load {image_path}: {e}")
    
    def _load_known_from_cache(self) -> bool:
        """Load known faces from cache file"""
        try:
            if not self.known_cache_file.exists():
                return False
            
            with open(self.known_cache_file, 'rb') as f:
                data = pickle.load(f)
                self.known_encodings = data['encodings']
                self.known_names = data['names']
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load known faces cache: {e}")
            return False
    
    def _load_blacklist_from_cache(self) -> bool:
        """Load blacklist faces from cache file"""
        try:
            if not self.blacklist_cache_file.exists():
                return False
            
            with open(self.blacklist_cache_file, 'rb') as f:
                data = pickle.load(f)
                self.blacklist_encodings = data['encodings']
                self.blacklist_names = data['names']
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load blacklist faces cache: {e}")
            return False
    
    def _save_known_to_cache(self):
        """Save known faces to cache file"""
        try:
            data = {
                'encodings': self.known_encodings,
                'names': self.known_names
            }
            
            with open(self.known_cache_file, 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            logger.error(f"Failed to save known faces cache: {e}")
    
    def _save_blacklist_to_cache(self):
        """Save blacklist faces to cache file"""
        try:
            data = {
                'encodings': self.blacklist_encodings,
                'names': self.blacklist_names
            }
            
            with open(self.blacklist_cache_file, 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            logger.error(f"Failed to save blacklist faces cache: {e}")
    
    def get_stats(self) -> Dict:
        """Get face database statistics"""
        return {
            'known_faces': len(self.known_names),
            'blacklist_faces': len(self.blacklist_names),
            'known_names': self.known_names.copy(),
            'blacklist_names': self.blacklist_names.copy()
        }
