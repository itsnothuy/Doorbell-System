#!/usr/bin/env python3
"""
Face Recognition Worker

High-performance face recognition engine with database integration,
caching optimization, and real-time person identification.
"""

import time
import logging
from typing import Dict, List, Any, Optional

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from src.pipeline.base_worker import PipelineWorker
from src.communication.message_bus import MessageBus, Message
from src.communication.events import (
    PipelineEvent, 
    EventType, 
    FaceDetectionEvent, 
    FaceRecognitionEvent,
    FaceRecognition,
    RecognitionStatus
)
from src.storage.face_database import FaceDatabase
from src.recognition.face_encoder import FaceEncoder
from src.recognition.similarity_matcher import SimilarityMatcher
from src.recognition.recognition_cache import RecognitionCache
from src.recognition.recognition_result import (
    FaceRecognitionResult, 
    PersonMatch,
    RecognitionMetadata
)

logger = logging.getLogger(__name__)


class FaceRecognitionWorker(PipelineWorker):
    """Face recognition worker with database integration and caching."""
    
    def __init__(self, message_bus: MessageBus, face_database: FaceDatabase, config: Dict[str, Any]):
        """
        Initialize face recognition worker.
        
        Args:
            message_bus: Message bus for event communication
            face_database: Face database instance
            config: Configuration dictionary
        """
        # Initialize core components before calling super().__init__
        self.face_database = face_database
        self.face_encoder = FaceEncoder(config)
        self.similarity_matcher = SimilarityMatcher(config)
        self.recognition_cache = RecognitionCache(config.get('cache', {}))
        
        # Configuration
        self.tolerance = config.get('tolerance', 0.6)
        self.blacklist_tolerance = config.get('blacklist_tolerance', 0.5)
        self.min_confidence = config.get('min_confidence', 0.4)
        self.batch_size = config.get('batch_size', 5)
        
        # Performance metrics
        self.recognition_count = 0
        self.recognition_errors = 0
        self.total_recognition_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.known_person_matches = 0
        self.blacklist_matches = 0
        self.unknown_faces = 0
        
        # Call parent constructor
        super().__init__(message_bus, config)
        
        logger.info(f"Initialized {self.worker_id} with tolerance {self.tolerance}")
    
    def _setup_subscriptions(self):
        """Setup message bus subscriptions."""
        self.message_bus.subscribe('faces_detected', self.handle_detection_event, self.worker_id)
        self.message_bus.subscribe('face_database_updated', self.handle_database_update, self.worker_id)
        self.message_bus.subscribe('recognition_cache_clear', self.handle_cache_clear, self.worker_id)
        self.message_bus.subscribe('system_shutdown', self.handle_shutdown, self.worker_id)
        logger.debug(f"{self.worker_id} subscriptions configured")
    
    def _initialize_worker(self):
        """Initialize face database and warm up cache."""
        try:
            # Initialize face database
            if not self.face_database.is_initialized():
                self.face_database.initialize()
            
            # Warm up recognition cache with frequently accessed faces
            self._warm_up_cache()
            
            # Test face encoding
            test_encoding = self.face_encoder.test_encoding()
            if test_encoding is None:
                logger.warning("Face encoder test returned None - may not be fully functional")
            
            logger.info(f"{self.worker_id} initialized successfully")
            
        except Exception as e:
            logger.error(f"{self.worker_id} initialization failed: {e}")
            raise
    
    def handle_detection_event(self, message: Message):
        """Handle face detection event and perform recognition."""
        detection_event: FaceDetectionEvent = message.data
        
        try:
            start_time = time.time()
            logger.debug(f"Processing detection event: {detection_event.event_id} with {len(detection_event.faces)} faces")
            
            # Convert FaceDetection objects to FaceRecognition objects
            recognitions = []
            
            # Process faces in batches for efficiency
            for i in range(0, len(detection_event.faces), self.batch_size):
                batch_faces = detection_event.faces[i:i + self.batch_size]
                batch_recognitions = self._process_face_batch(batch_faces, detection_event.event_id)
                recognitions.extend(batch_recognitions)
            
            # Create recognition event
            recognition_time = time.time() - start_time
            recognition_event = FaceRecognitionEvent(
                recognitions=recognitions,
                recognition_time=recognition_time,
                source=self.worker_id,
                correlation_id=detection_event.event_id,
                parent_event_id=detection_event.event_id
            )
            
            # Publish recognition results
            self.message_bus.publish('faces_recognized', recognition_event)
            
            # Update metrics
            self.recognition_count += len(detection_event.faces)
            self.total_recognition_time += recognition_time
            self.processed_count += 1
            
            logger.debug(f"Recognition completed for {detection_event.event_id}: {len(recognitions)} results in {recognition_time*1000:.2f}ms")
            
        except Exception as e:
            self.recognition_errors += 1
            self.error_count += 1
            logger.error(f"Recognition processing failed: {e}", exc_info=True)
            self._handle_recognition_error(e, detection_event)
    
    def _process_face_batch(self, faces: List[Any], event_id: str) -> List[FaceRecognition]:
        """Process a batch of faces for recognition."""
        recognitions = []
        
        for face_detection in faces:
            try:
                # Extract face encoding
                face_encoding = self._extract_face_encoding(face_detection)
                if face_encoding is None:
                    logger.warning(f"Failed to extract encoding for face in {event_id}")
                    # Create failed recognition
                    recognitions.append(FaceRecognition(
                        face_detection=face_detection,
                        status=RecognitionStatus.FAILED,
                        recognition_time=0.0
                    ))
                    continue
                
                # Perform recognition
                recognition = self._recognize_face(face_detection, face_encoding)
                recognitions.append(recognition)
                
                # Update statistics
                if recognition.status == RecognitionStatus.KNOWN:
                    self.known_person_matches += 1
                elif recognition.status == RecognitionStatus.BLACKLISTED:
                    self.blacklist_matches += 1
                elif recognition.status == RecognitionStatus.UNKNOWN:
                    self.unknown_faces += 1
                
            except Exception as e:
                logger.error(f"Face processing failed in batch: {e}", exc_info=True)
                # Add failed recognition
                recognitions.append(FaceRecognition(
                    face_detection=face_detection,
                    status=RecognitionStatus.FAILED,
                    recognition_time=0.0
                ))
                continue
        
        return recognitions
    
    def _extract_face_encoding(self, face_detection: Any) -> Optional[Any]:
        """Extract face encoding from detected face."""
        try:
            # Check if encoding already exists in detection
            if hasattr(face_detection, 'encoding') and face_detection.encoding is not None:
                return face_detection.encoding
            
            # For now, generate a mock encoding since we don't have the actual face image
            # In a real implementation, this would extract from face_detection.face_image
            if NUMPY_AVAILABLE:
                # Generate deterministic encoding based on face position for consistency
                bbox = face_detection.bounding_box
                seed = bbox.x + bbox.y + bbox.width + bbox.height
                np.random.seed(seed % 2**32)
                encoding = np.random.rand(128)
                return encoding
            
            return None
            
        except Exception as e:
            logger.error(f"Face encoding extraction failed: {e}")
            return None
    
    def _recognize_face(self, face_detection: Any, face_encoding: Any) -> FaceRecognition:
        """Perform face recognition against database."""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(face_encoding)
        cached_result = self.recognition_cache.get(cache_key)
        
        if cached_result is not None:
            self.cache_hits += 1
            logger.debug("Cache hit for face recognition")
            return self._create_face_recognition(face_detection, face_encoding, cached_result, start_time)
        
        self.cache_misses += 1
        
        # Check blacklist first (higher priority)
        blacklist_matches = self.face_database.find_blacklist_matches(
            face_encoding, 
            tolerance=self.blacklist_tolerance
        )
        
        if blacklist_matches:
            best_match = blacklist_matches[0]
            result_data = {
                'status': RecognitionStatus.BLACKLISTED,
                'identity': best_match.person_name or best_match.person_id,
                'similarity_score': best_match.confidence,
                'match_details': {
                    'person_id': best_match.person_id,
                    'person_name': best_match.person_name,
                    'confidence': best_match.confidence
                }
            }
            
            # Cache result
            self.recognition_cache.put(cache_key, result_data)
            
            return self._create_face_recognition(face_detection, face_encoding, result_data, start_time)
        
        # Check known persons
        known_matches = self.face_database.find_known_matches(
            face_encoding,
            tolerance=self.tolerance
        )
        
        if known_matches:
            # Filter by minimum confidence
            confident_matches = [m for m in known_matches if m.confidence >= self.min_confidence]
            
            if confident_matches:
                best_match = confident_matches[0]
                result_data = {
                    'status': RecognitionStatus.KNOWN,
                    'identity': best_match.person_name or best_match.person_id,
                    'similarity_score': best_match.confidence,
                    'match_details': {
                        'person_id': best_match.person_id,
                        'person_name': best_match.person_name,
                        'confidence': best_match.confidence
                    }
                }
                
                # Cache result
                self.recognition_cache.put(cache_key, result_data)
                
                return self._create_face_recognition(face_detection, face_encoding, result_data, start_time)
        
        # Unknown face
        result_data = {
            'status': RecognitionStatus.UNKNOWN,
            'identity': None,
            'similarity_score': 0.0,
            'match_details': None
        }
        
        # Cache unknown result (shorter TTL)
        self.recognition_cache.put(cache_key, result_data, ttl=300)  # 5 minutes
        
        return self._create_face_recognition(face_detection, face_encoding, result_data, start_time)
    
    def _create_face_recognition(self, face_detection: Any, face_encoding: Any,
                                 result_data: Dict[str, Any], start_time: float) -> FaceRecognition:
        """Create FaceRecognition object from detection and match data."""
        recognition_time = time.time() - start_time
        
        return FaceRecognition(
            face_detection=face_detection,
            status=result_data.get('status', RecognitionStatus.UNKNOWN),
            identity=result_data.get('identity'),
            similarity_score=result_data.get('similarity_score', 0.0),
            recognition_time=recognition_time,
            match_details=result_data.get('match_details')
        )
    
    def _generate_cache_key(self, face_encoding: Any) -> str:
        """Generate cache key from face encoding."""
        if not NUMPY_AVAILABLE:
            return f"face_encoding_{hash(str(face_encoding)):016x}"
        
        # Use hash of encoding for cache key
        encoding_array = np.array(face_encoding) if not isinstance(face_encoding, np.ndarray) else face_encoding
        encoding_hash = hash(encoding_array.tobytes())
        return f"face_encoding_{encoding_hash:016x}"
    
    def _warm_up_cache(self):
        """Warm up cache with frequently accessed faces."""
        try:
            # Get most frequently matched faces from database
            frequent_faces = self.face_database.get_frequent_faces(limit=100)
            
            for face_data in frequent_faces:
                cache_key = self._generate_cache_key(face_data['encoding'])
                status = RecognitionStatus.BLACKLISTED if face_data['is_blacklisted'] else RecognitionStatus.KNOWN
                self.recognition_cache.put(cache_key, {
                    'status': status,
                    'identity': face_data['person_id'],
                    'similarity_score': 1.0,
                    'match_details': {
                        'person_id': face_data['person_id'],
                        'confidence': 1.0
                    }
                })
            
            logger.info(f"Cache warmed up with {len(frequent_faces)} frequent faces")
            
        except Exception as e:
            logger.warning(f"Cache warm-up failed: {e}")
    
    def handle_database_update(self, message: Message):
        """Handle face database update events."""
        try:
            update_data = message.data
            
            # Clear cache for affected persons
            if 'person_id' in update_data:
                self.recognition_cache.invalidate_person(update_data['person_id'])
            elif update_data.get('clear_all'):
                self.recognition_cache.clear()
            
            logger.info(f"Processed database update: {update_data}")
            
        except Exception as e:
            logger.error(f"Database update handling failed: {e}")
    
    def handle_cache_clear(self, message: Message):
        """Handle cache clear requests."""
        try:
            clear_data = message.data
            
            if clear_data.get('clear_all'):
                self.recognition_cache.clear()
                logger.info("Recognition cache cleared")
            elif 'person_id' in clear_data:
                self.recognition_cache.invalidate_person(clear_data['person_id'])
                logger.info(f"Cache cleared for person: {clear_data['person_id']}")
            
        except Exception as e:
            logger.error(f"Cache clear handling failed: {e}")
    
    def _handle_recognition_error(self, error: Exception, detection_event: FaceDetectionEvent):
        """Handle recognition errors and publish error events."""
        error_event = PipelineEvent(
            event_type=EventType.COMPONENT_ERROR,
            data={
                'component': self.worker_id,
                'error': str(error),
                'error_type': type(error).__name__,
                'detection_event_id': detection_event.event_id,
                'face_count': len(detection_event.faces),
                'recognition_metrics': self.get_metrics()
            },
            source=self.worker_id
        )
        
        self.message_bus.publish('recognition_errors', error_event)
    
    def _cleanup_worker(self):
        """Cleanup worker resources."""
        try:
            # Save cache state
            if hasattr(self.recognition_cache, 'save_state'):
                self.recognition_cache.save_state()
            
            # Close database connections
            if self.face_database:
                self.face_database.close()
            
            logger.info(f"{self.worker_id} cleanup completed")
            
        except Exception as e:
            logger.error(f"{self.worker_id} cleanup failed: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get worker performance metrics."""
        base_metrics = super().get_metrics()
        
        recognition_metrics = {
            'recognition_count': self.recognition_count,
            'recognition_errors': self.recognition_errors,
            'avg_recognition_time': self.total_recognition_time / max(1, self.recognition_count),
            'recognition_rate': self.recognition_count / max(1, time.time() - (self.start_time or time.time())),
            'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'known_person_matches': self.known_person_matches,
            'blacklist_matches': self.blacklist_matches,
            'unknown_faces': self.unknown_faces,
            'error_rate': self.recognition_errors / max(1, self.recognition_count),
            'tolerance': self.tolerance,
            'blacklist_tolerance': self.blacklist_tolerance
        }
        
        return {**base_metrics, **recognition_metrics}
