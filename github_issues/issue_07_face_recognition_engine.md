# GitHub Issue: Implement Face Recognition Engine with Database Integration

## 📋 Overview

### Phase Information
- **Phase**: 2 - Core Pipeline Workers
- **PR Number**: #7
- **Complexity**: Medium-High
- **Estimated Duration**: 4-5 days
- **Dependencies**: Face detection worker, Face database, Storage layer
- **Priority**: High (Core AI/ML functionality)

### Goals
Implement a high-performance face recognition engine with face encoding, similarity matching, database integration, and caching optimization for real-time person identification in doorbell security applications.

## 🎯 Requirements

### Functional Requirements

#### Core Functionality
- **Face Encoding Extraction**: Generate face encodings from detected faces using face_recognition library
- **Similarity Matching**: Compare face encodings against known faces and blacklist databases
- **Database Integration**: Efficient storage and retrieval of face encodings with metadata
- **Caching System**: LRU cache for frequently accessed face encodings
- **Batch Processing**: Support for processing multiple faces per frame efficiently
- **Threshold Configuration**: Configurable similarity thresholds for recognition decisions

#### Recognition Processing
- **Detection Event Processing**: Subscribe to `faces_detected` events from face detection worker
- **Recognition Results**: Publish `faces_recognized` events with identification results
- **Known Person Identification**: Match against known person database
- **Blacklist Detection**: Identify faces on blacklist with immediate alerts
- **Unknown Person Handling**: Create records for unrecognized faces for later identification

### Non-Functional Requirements

#### Performance Targets
- **Recognition Latency**: <200ms per face on Raspberry Pi 4
- **Database Query Time**: <50ms for face lookup operations
- **Cache Hit Rate**: >80% for known faces in typical usage
- **Memory Usage**: <100MB for face database and cache
- **Throughput**: >10 faces/second recognition processing
- **Startup Time**: <3 seconds for database initialization and cache warming

#### Reliability Requirements
- **Data Consistency**: Ensure face database integrity and consistency
- **Error Recovery**: Graceful handling of database errors and corruption
- **Performance Monitoring**: Real-time metrics for recognition accuracy and performance
- **Cache Coherence**: Maintain cache consistency with database updates
- **Backup Integration**: Support for face database backup and restore

## 🔧 Implementation Specifications

### Files to Create/Modify

#### New Files
```
src/pipeline/face_recognizer.py     # Main face recognition worker
src/storage/face_database.py        # Face encoding database manager
src/recognition/face_encoder.py     # Face encoding extraction utility
src/recognition/similarity_matcher.py # Face similarity matching engine
src/recognition/recognition_cache.py # Face recognition caching system
src/recognition/recognition_result.py # Recognition result data structures
tests/test_face_recognizer.py       # Comprehensive unit tests
tests/test_face_database.py         # Database tests
tests/performance/recognition_bench.py # Performance benchmarking
tests/fixtures/face_test_data.py    # Test face data and fixtures
```

#### Modified Files
```
config/pipeline_config.py           # Add face recognition configuration
requirements.txt                    # Add face_recognition dependencies
```

### Architecture Patterns

#### Face Recognition Worker Pattern
```python
class FaceRecognitionWorker(PipelineWorker):
    """Face recognition worker with database integration and caching."""
    
    def __init__(self, message_bus: MessageBus, face_database: FaceDatabase, config: Dict[str, Any]):
        super().__init__(message_bus, config)
        
        # Core components
        self.face_database = face_database
        self.face_encoder = FaceEncoder(config)
        self.similarity_matcher = SimilarityMatcher(config)
        self.recognition_cache = RecognitionCache(config)
        
        # Configuration
        self.tolerance = config.get('tolerance', 0.6)
        self.cache_size = config.get('cache_size', 1000)
        self.batch_size = config.get('batch_size', 5)
        
    def _setup_subscriptions(self):
        self.message_bus.subscribe('faces_detected', self.handle_detection_event, self.worker_id)
        self.message_bus.subscribe('face_database_updated', self.handle_database_update, self.worker_id)
```

#### Face Database Pattern
```python
class FaceDatabase:
    \"\"\"SQLite-based face encoding database with efficient similarity search.\"\"\"
    
    def __init__(self, db_path: str, config: Dict[str, Any]):
        self.db_path = db_path
        self.config = config
        self._init_database()
        
    def add_face(self, person_id: str, encoding: np.ndarray, metadata: Dict[str, Any]) -> str:
        \"\"\"Add face encoding to database with metadata.\"\"\"
        
    def find_matches(self, encoding: np.ndarray, tolerance: float = 0.6) -> List[FaceMatch]:
        \"\"\"Find matching faces in database using similarity search.\"\"\"
        
    def get_person_encodings(self, person_id: str) -> List[np.ndarray]:
        \"\"\"Get all face encodings for a specific person.\"\"\"
```

### Core Implementation

#### Face Recognition Worker
```python
#!/usr/bin/env python3
\"\"\"
Face Recognition Worker

High-performance face recognition engine with database integration,
caching optimization, and real-time person identification.
\"\"\"

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from src.pipeline.base_worker import PipelineWorker
from src.communication.message_bus import MessageBus, Message
from src.communication.events import PipelineEvent, EventType, FaceDetectionEvent, FaceRecognitionEvent
from src.storage.face_database import FaceDatabase
from src.recognition.face_encoder import FaceEncoder
from src.recognition.similarity_matcher import SimilarityMatcher
from src.recognition.recognition_cache import RecognitionCache
from src.recognition.recognition_result import FaceRecognitionResult, PersonMatch

logger = logging.getLogger(__name__)


class FaceRecognitionWorker(PipelineWorker):
    \"\"\"Face recognition worker with database integration and caching.\"\"\"
    
    def __init__(self, message_bus: MessageBus, face_database: FaceDatabase, config: Dict[str, Any]):
        super().__init__(message_bus, config)
        
        # Core components
        self.face_database = face_database
        self.face_encoder = FaceEncoder(config)
        self.similarity_matcher = SimilarityMatcher(config)
        self.recognition_cache = RecognitionCache(config)
        
        # Configuration
        self.tolerance = config.get('tolerance', 0.6)
        self.blacklist_tolerance = config.get('blacklist_tolerance', 0.5)  # Stricter for blacklist
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
        
        logger.info(f\"Initialized {self.worker_id} with tolerance {self.tolerance}\")
    
    def _setup_subscriptions(self):
        \"\"\"Setup message bus subscriptions.\"\"\"
        self.message_bus.subscribe('faces_detected', self.handle_detection_event, self.worker_id)
        self.message_bus.subscribe('face_database_updated', self.handle_database_update, self.worker_id)
        self.message_bus.subscribe('recognition_cache_clear', self.handle_cache_clear, self.worker_id)
        self.message_bus.subscribe('system_shutdown', self.handle_shutdown, self.worker_id)
        logger.debug(f\"{self.worker_id} subscriptions configured\")
    
    def _initialize_worker(self):
        \"\"\"Initialize face database and warm up cache.\"\"\"
        try:
            # Initialize face database
            if not self.face_database.is_initialized():
                self.face_database.initialize()
            
            # Warm up recognition cache with frequently accessed faces
            self._warm_up_cache()
            
            # Test face encoding
            test_encoding = self.face_encoder.test_encoding()
            if test_encoding is None:
                raise RuntimeError(\"Face encoder initialization failed\")
            
            logger.info(f\"{self.worker_id} initialized successfully\")
            
        except Exception as e:
            logger.error(f\"{self.worker_id} initialization failed: {e}\")
            raise
    
    def handle_detection_event(self, message: Message):
        \"\"\"Handle face detection event and perform recognition.\"\"\"
        detection_event = message.data
        
        try:
            start_time = time.time()
            logger.debug(f\"Processing detection event: {detection_event.event_id} with {len(detection_event.faces)} faces\")
            
            recognition_results = []
            
            # Process faces in batches for efficiency
            for i in range(0, len(detection_event.faces), self.batch_size):
                batch_faces = detection_event.faces[i:i + self.batch_size]
                batch_results = self._process_face_batch(batch_faces, detection_event.event_id)
                recognition_results.extend(batch_results)
            
            # Create recognition event
            recognition_event = FaceRecognitionEvent(
                event_id=detection_event.event_id,
                frame_event_id=detection_event.frame_event_id,
                detection_event_id=detection_event.event_id,
                recognition_results=recognition_results,
                recognition_metadata={
                    'total_faces': len(detection_event.faces),
                    'known_matches': sum(1 for r in recognition_results if r.is_known),
                    'blacklist_matches': sum(1 for r in recognition_results if r.is_blacklisted),
                    'unknown_faces': sum(1 for r in recognition_results if not r.is_known),
                    'processing_time_ms': (time.time() - start_time) * 1000,
                    'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
                    'recognition_timestamp': time.time()
                }
            )
            
            # Publish recognition results
            self.message_bus.publish('faces_recognized', recognition_event)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.recognition_count += len(detection_event.faces)
            self.total_recognition_time += processing_time
            self.processed_count += 1
            
            logger.debug(f\"Recognition completed for {detection_event.event_id}: {len(recognition_results)} results\")
            
        except Exception as e:
            self.recognition_errors += 1
            self.error_count += 1
            logger.error(f\"Recognition processing failed: {e}\")
            self._handle_recognition_error(e, detection_event)
    
    def _process_face_batch(self, faces: List['FaceDetectionResult'], event_id: str) -> List[FaceRecognitionResult]:
        \"\"\"Process a batch of faces for recognition.\"\"\"
        results = []
        
        for face_detection in faces:
            try:
                # Extract face encoding
                face_encoding = self._extract_face_encoding(face_detection)
                if face_encoding is None:
                    logger.warning(f\"Failed to extract encoding for face in {event_id}\")
                    continue
                
                # Perform recognition
                recognition_result = self._recognize_face(face_detection, face_encoding)
                results.append(recognition_result)
                
                # Update statistics
                if recognition_result.is_known:
                    self.known_person_matches += 1
                if recognition_result.is_blacklisted:
                    self.blacklist_matches += 1
                if not recognition_result.is_known:
                    self.unknown_faces += 1
                
            except Exception as e:
                logger.error(f\"Face processing failed in batch: {e}\")
                continue
        
        return results
    
    def _extract_face_encoding(self, face_detection: 'FaceDetectionResult') -> Optional[np.ndarray]:
        \"\"\"Extract face encoding from detected face.\"\"\"
        try:
            # Use face image from detection result
            face_image = face_detection.face_image
            
            # Extract encoding using face_recognition library
            encoding = self.face_encoder.encode_face(face_image)
            
            return encoding
            
        except Exception as e:
            logger.error(f\"Face encoding extraction failed: {e}\")
            return None
    
    def _recognize_face(self, face_detection: 'FaceDetectionResult', face_encoding: np.ndarray) -> FaceRecognitionResult:
        \"\"\"Perform face recognition against database.\"\"\"
        # Check cache first
        cache_key = self._generate_cache_key(face_encoding)
        cached_result = self.recognition_cache.get(cache_key)
        
        if cached_result is not None:
            self.cache_hits += 1
            logger.debug(\"Cache hit for face recognition\")
            return self._create_recognition_result(face_detection, face_encoding, cached_result)
        
        self.cache_misses += 1
        
        # Check blacklist first (higher priority)
        blacklist_matches = self.face_database.find_blacklist_matches(
            face_encoding, 
            tolerance=self.blacklist_tolerance
        )
        
        if blacklist_matches:
            best_blacklist_match = blacklist_matches[0]
            result_data = {
                'is_blacklisted': True,
                'person_matches': [best_blacklist_match],
                'confidence': best_blacklist_match.confidence
            }
            
            # Cache result
            self.recognition_cache.put(cache_key, result_data)
            
            return self._create_recognition_result(face_detection, face_encoding, result_data)
        
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
                    'is_known': True,
                    'person_matches': confident_matches[:3],  # Top 3 matches
                    'confidence': best_match.confidence
                }
                
                # Cache result
                self.recognition_cache.put(cache_key, result_data)
                
                return self._create_recognition_result(face_detection, face_encoding, result_data)
        
        # Unknown face
        result_data = {
            'is_known': False,
            'is_blacklisted': False,
            'person_matches': [],
            'confidence': 0.0
        }
        
        # Cache unknown result (shorter TTL)
        self.recognition_cache.put(cache_key, result_data, ttl=300)  # 5 minutes
        
        return self._create_recognition_result(face_detection, face_encoding, result_data)
    
    def _create_recognition_result(self, face_detection: 'FaceDetectionResult', 
                                 face_encoding: np.ndarray, result_data: Dict[str, Any]) -> FaceRecognitionResult:
        \"\"\"Create face recognition result from detection and match data.\"\"\"
        return FaceRecognitionResult(
            face_detection=face_detection,
            face_encoding=face_encoding,
            is_known=result_data.get('is_known', False),
            is_blacklisted=result_data.get('is_blacklisted', False),
            person_matches=result_data.get('person_matches', []),
            confidence=result_data.get('confidence', 0.0),
            recognition_timestamp=time.time(),
            metadata={
                'tolerance_used': self.blacklist_tolerance if result_data.get('is_blacklisted') else self.tolerance,
                'cache_hit': cache_key in self.recognition_cache if hasattr(self, '_last_cache_key') else False,
                'processing_method': 'cached' if result_data.get('cached') else 'database'
            }
        )
    
    def _generate_cache_key(self, face_encoding: np.ndarray) -> str:
        \"\"\"Generate cache key from face encoding.\"\"\"
        # Use hash of encoding for cache key
        encoding_hash = hash(face_encoding.tobytes())
        return f\"face_encoding_{encoding_hash:016x}\"
    
    def _warm_up_cache(self):
        \"\"\"Warm up cache with frequently accessed faces.\"\"\"
        try:
            # Get most frequently matched faces from database
            frequent_faces = self.face_database.get_frequent_faces(limit=100)
            
            for face_data in frequent_faces:
                cache_key = self._generate_cache_key(face_data['encoding'])
                self.recognition_cache.put(cache_key, {
                    'person_id': face_data['person_id'],
                    'is_known': True,
                    'is_blacklisted': face_data.get('is_blacklisted', False),
                    'confidence': 1.0  # Perfect match for known encoding
                })
            
            logger.info(f\"Cache warmed up with {len(frequent_faces)} frequent faces\")
            
        except Exception as e:
            logger.warning(f\"Cache warm-up failed: {e}\")
    
    def handle_database_update(self, message: Message):
        \"\"\"Handle face database update events.\"\"\"
        try:
            update_data = message.data
            
            # Clear cache for affected persons
            if 'person_id' in update_data:
                self.recognition_cache.invalidate_person(update_data['person_id'])
            elif update_data.get('clear_all'):
                self.recognition_cache.clear()
            
            logger.info(f\"Processed database update: {update_data}\")
            
        except Exception as e:
            logger.error(f\"Database update handling failed: {e}\")
    
    def handle_cache_clear(self, message: Message):
        \"\"\"Handle cache clear requests.\"\"\"
        try:
            clear_data = message.data
            
            if clear_data.get('clear_all'):
                self.recognition_cache.clear()
                logger.info(\"Recognition cache cleared\")
            elif 'person_id' in clear_data:
                self.recognition_cache.invalidate_person(clear_data['person_id'])
                logger.info(f\"Cache cleared for person: {clear_data['person_id']}\")
            
        except Exception as e:
            logger.error(f\"Cache clear handling failed: {e}\")
    
    def _handle_recognition_error(self, error: Exception, detection_event: FaceDetectionEvent):
        \"\"\"Handle recognition errors and publish error events.\"\"\"
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
        \"\"\"Cleanup worker resources.\"\"\"
        try:
            # Save cache state
            if hasattr(self.recognition_cache, 'save_state'):
                self.recognition_cache.save_state()
            
            # Close database connections
            if self.face_database:
                self.face_database.close()
            
            logger.info(f\"{self.worker_id} cleanup completed\")
            
        except Exception as e:
            logger.error(f\"{self.worker_id} cleanup failed: {e}\")
    
    def get_metrics(self) -> Dict[str, Any]:
        \"\"\"Get worker performance metrics.\"\"\"
        base_metrics = super().get_metrics()
        
        recognition_metrics = {
            'recognition_count': self.recognition_count,
            'recognition_errors': self.recognition_errors,
            'avg_recognition_time': self.total_recognition_time / max(1, self.recognition_count),
            'recognition_rate': self.recognition_count / max(1, time.time() - self.start_time),
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
```

### Configuration

#### Face Recognition Configuration
```yaml
# config/pipeline_config.py - Face Recognition Section
face_recognition:
  enabled: true
  tolerance: 0.6
  blacklist_tolerance: 0.5
  min_confidence: 0.4
  batch_size: 5
  
  # Face encoding settings
  encoding:
    model: \"large\"  # or \"small\"
    jitters: 1
    num_upsamplings: 1
    
  # Database configuration
  database:
    path: \"data/faces.db\"
    backup_enabled: true
    backup_interval_hours: 24
    max_faces_per_person: 10
    
  # Cache configuration
  cache:
    enabled: true
    size: 1000
    ttl_seconds: 3600
    warm_up_enabled: true
    warm_up_limit: 100
    
  # Performance tuning
  performance:
    max_concurrent_recognitions: 10
    database_connection_pool: 5
    encoding_timeout_seconds: 5.0
    
  # Similarity thresholds
  thresholds:
    known_person: 0.6
    blacklist: 0.5
    high_confidence: 0.7
    low_confidence: 0.4
```

## 🧪 Testing Requirements

### Unit Tests (>90% Coverage)

#### Core Functionality Tests
```python
class TestFaceRecognitionWorker:
    \"\"\"Comprehensive test suite for face recognition worker.\"\"\"
    
    def test_worker_initialization(self):
        \"\"\"Test worker initializes with correct configuration.\"\"\"
        
    def test_face_encoding_extraction(self):
        \"\"\"Test face encoding extraction from detection results.\"\"\"
        
    def test_known_person_recognition(self):
        \"\"\"Test recognition of known persons.\"\"\"
        
    def test_blacklist_detection(self):
        \"\"\"Test blacklist face detection.\"\"\"
        
    def test_unknown_face_handling(self):
        \"\"\"Test handling of unknown faces.\"\"\"
        
    def test_batch_processing(self):
        \"\"\"Test batch processing of multiple faces.\"\"\"
        
    def test_cache_functionality(self):
        \"\"\"Test recognition cache hit/miss behavior.\"\"\"
        
    def test_database_integration(self):
        \"\"\"Test face database operations.\"\"\"
        
    def test_similarity_thresholds(self):
        \"\"\"Test different similarity thresholds.\"\"\"
        
    def test_error_handling(self):
        \"\"\"Test error scenarios and recovery.\"\"\"
```

#### Performance Tests
```python
class TestFaceRecognitionPerformance:
    \"\"\"Performance testing for face recognition.\"\"\"
    
    def test_recognition_latency(self):
        \"\"\"Test recognition latency for various scenarios.\"\"\"
        
    def test_database_query_performance(self):
        \"\"\"Test database query performance with large datasets.\"\"\"
        
    def test_cache_performance(self):
        \"\"\"Test cache hit rates and performance impact.\"\"\"
        
    def test_memory_usage(self):
        \"\"\"Test memory usage with large face databases.\"\"\"
        
    def test_concurrent_recognition(self):
        \"\"\"Test performance with concurrent recognition requests.\"\"\"
```

### Integration Tests

#### Database Integration
```python
def test_face_database_operations():
    \"\"\"Test face database CRUD operations.\"\"\"
    
def test_recognition_accuracy():
    \"\"\"Test recognition accuracy with known test dataset.\"\"\"
    
def test_pipeline_integration():
    \"\"\"Test integration with face detection worker.\"\"\"
```

### Test Coverage Requirements
- **Minimum Coverage**: 90%
- **Critical Path Coverage**: 100% (recognition logic, database operations, cache management)
- **Error Handling Coverage**: 100%
- **Database Operations Coverage**: 100%

## ✅ Acceptance Criteria

### Definition of Done
- [ ] **Face Recognition Engine**: Complete recognition workflow implemented
- [ ] **Database Integration**: Efficient face database operations
- [ ] **Caching System**: LRU cache with high hit rates
- [ ] **Performance Targets**: All benchmarks met on target hardware
- [ ] **Blacklist Detection**: Immediate identification of blacklisted faces
- [ ] **Unknown Face Handling**: Proper processing of unrecognized faces
- [ ] **Error Recovery**: Graceful handling of all error scenarios
- [ ] **Integration Testing**: Seamless pipeline integration

### Quality Gates
- [ ] **Recognition Accuracy**: >95% accuracy on test dataset
- [ ] **Database Consistency**: Zero data corruption or inconsistency
- [ ] **Cache Coherence**: Cache always consistent with database
- [ ] **Memory Management**: No memory leaks during extended operation
- [ ] **Performance Stability**: Consistent performance under load

### Performance Benchmarks
- [ ] **<200ms Recognition Latency**: Per face on Raspberry Pi 4
- [ ] **<50ms Database Query Time**: For face lookup operations
- [ ] **>80% Cache Hit Rate**: For known faces in typical usage
- [ ] **<100MB Memory Usage**: For face database and cache
- [ ] **>10 Faces/Second**: Recognition processing throughput
- [ ] **<3 Second Startup**: Database and cache initialization
- [ ] **>95% Accuracy**: Recognition accuracy on test dataset

## 🏷️ Labels

`enhancement`, `pipeline`, `ai-ml`, `database`, `phase-2`, `priority-high`, `complexity-medium-high`

## 📝 Implementation Notes

### Development Approach
1. **Start with Core Recognition**: Implement basic face encoding and matching
2. **Add Database Layer**: Implement efficient face database operations
3. **Integrate Caching**: Add LRU cache for performance optimization
4. **Performance Optimization**: Optimize for target hardware
5. **Error Handling**: Add comprehensive error handling and recovery
6. **Accuracy Testing**: Test recognition accuracy with diverse datasets

### Risk Mitigation
- **Database Performance**: Index optimization and query performance testing
- **Memory Usage**: Regular memory profiling and optimization
- **Recognition Accuracy**: Extensive testing with diverse face datasets
- **Cache Consistency**: Careful cache invalidation strategies

### Success Metrics
- All acceptance criteria met
- Recognition accuracy meets or exceeds target
- Performance benchmarks achieved consistently
- Zero data loss or corruption in production

---

## 🤖 **For Coding Agents: Auto-Close Setup**

### **Branch Naming Convention**
When implementing this issue, create your branch using one of these patterns:
- `issue-7/face-recognition-engine`
- `7-face-recognition-engine` 
- `issue-7/implement-face-recognition`

### **PR Creation**
The GitHub Action will automatically append `Closes #7` to your PR description when you follow the branch naming convention above. This ensures the issue closes automatically when your PR is merged to the default branch.

### **Manual Alternative**
If you prefer manual control, include one of these in your PR description:
```
Closes #7
Fixes #7
Resolves #7
```

---

**This issue implements the face recognition engine as the core intelligence component of the Frigate-inspired pipeline architecture, providing accurate person identification with comprehensive database management and real-time performance optimization.**