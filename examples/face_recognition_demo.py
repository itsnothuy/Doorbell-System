#!/usr/bin/env python3
"""
Face Recognition Integration Example

Demonstrates how to use the face recognition engine components together.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("Warning: NumPy not available, using mock implementations")
    NUMPY_AVAILABLE = False
    np = None

from src.storage.face_database import FaceDatabase
from src.recognition.face_encoder import FaceEncoder
from src.recognition.similarity_matcher import SimilarityMatcher
from src.recognition.recognition_cache import RecognitionCache
from src.recognition.recognition_result import PersonMatch


def setup_example_database(db_path: str):
    """Set up example database with sample persons."""
    print("\n1. Setting up face database...")
    
    db = FaceDatabase(db_path, {
        'max_faces_per_person': 10,
        'backup_enabled': False
    })
    db.initialize()
    
    # Add example persons
    persons = [
        ("john_doe", "John Doe", False),
        ("jane_smith", "Jane Smith", False),
        ("alice_wonder", "Alice Wonder", False),
        ("suspicious_person", "Suspicious Person", True),  # Blacklisted
    ]
    
    for person_id, person_name, is_blacklisted in persons:
        db.add_person(person_id, person_name, is_blacklisted)
        print(f"  Added person: {person_name} (blacklisted: {is_blacklisted})")
        
        # Add face encodings for each person
        if NUMPY_AVAILABLE:
            for i in range(3):
                # Generate deterministic encoding for demo
                np.random.seed(hash(person_id) + i)
                encoding = np.random.rand(128)
                db.add_face_encoding(person_id, encoding, quality_score=0.8 + i * 0.05)
        else:
            print(f"    (Skipping encodings - NumPy not available)")
    
    stats = db.get_stats()
    print(f"\nDatabase stats:")
    print(f"  Persons: {stats['person_count']}")
    print(f"  Blacklisted: {stats['blacklisted_count']}")
    print(f"  Encodings: {stats['encoding_count']}")
    
    return db


def demonstrate_recognition_workflow(db: FaceDatabase):
    """Demonstrate complete recognition workflow."""
    print("\n2. Setting up recognition components...")
    
    # Setup encoder
    encoder = FaceEncoder({
        'encoding_model': 'small',
        'face_jitter': 1,
        'number_of_times_to_upsample': 1
    })
    
    # Setup matcher
    matcher = SimilarityMatcher({
        'similarity_metric': 'euclidean',
        'tolerance': 0.6
    })
    
    # Setup cache
    cache = RecognitionCache({
        'enabled': True,
        'cache_size': 100,
        'ttl_seconds': 3600
    })
    
    print("  ✓ Face encoder initialized")
    print("  ✓ Similarity matcher initialized")
    print("  ✓ Recognition cache initialized")
    
    # Simulate recognition scenarios
    print("\n3. Testing recognition scenarios...")
    
    scenarios = [
        ("Known person (John Doe)", "john_doe"),
        ("Known person (Jane Smith)", "jane_smith"),
        ("Blacklisted person", "suspicious_person"),
        ("Unknown person", None),
    ]
    
    for scenario_name, person_id in scenarios:
        print(f"\n  Scenario: {scenario_name}")
        
        if not NUMPY_AVAILABLE:
            print(f"    (Skipped - NumPy not available)")
            continue
        
        # Generate test encoding
        if person_id:
            # Use same seed as database for known persons
            np.random.seed(hash(person_id))
            test_encoding = np.random.rand(128)
        else:
            # Random encoding for unknown
            test_encoding = np.random.rand(128)
        
        # Check cache first
        start_time = time.time()
        cache_key = f"test_{person_id or 'unknown'}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            print(f"    ✓ Cache hit! Recognition time: {(time.time() - start_time)*1000:.2f}ms")
            print(f"      Identity: {cached_result.get('person_id', 'Unknown')}")
        else:
            # Query database
            blacklist_matches = db.find_blacklist_matches(test_encoding, tolerance=0.5)
            
            if blacklist_matches:
                match = blacklist_matches[0]
                print(f"    ⚠️ BLACKLISTED PERSON DETECTED!")
                print(f"      Identity: {match.person_name}")
                print(f"      Confidence: {match.confidence:.2%}")
                print(f"      Recognition time: {(time.time() - start_time)*1000:.2f}ms")
                
                # Cache result
                cache.put(cache_key, {
                    'person_id': match.person_id,
                    'confidence': match.confidence,
                    'is_blacklisted': True
                })
            else:
                known_matches = db.find_known_matches(test_encoding, tolerance=0.6)
                
                if known_matches:
                    match = known_matches[0]
                    print(f"    ✓ Known person recognized")
                    print(f"      Identity: {match.person_name}")
                    print(f"      Confidence: {match.confidence:.2%}")
                    print(f"      Recognition time: {(time.time() - start_time)*1000:.2f}ms")
                    
                    # Cache result
                    cache.put(cache_key, {
                        'person_id': match.person_id,
                        'confidence': match.confidence
                    })
                else:
                    print(f"    ℹ️ Unknown person")
                    print(f"      Recognition time: {(time.time() - start_time)*1000:.2f}ms")
                    
                    # Cache with short TTL
                    cache.put(cache_key, {
                        'person_id': None,
                        'confidence': 0.0
                    }, ttl=300)
    
    # Show cache stats
    print("\n4. Cache performance:")
    cache_stats = cache.get_stats()
    print(f"  Size: {cache_stats['size']}/{cache_stats['max_size']}")
    print(f"  Hits: {cache_stats['hits']}")
    print(f"  Misses: {cache_stats['misses']}")
    print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
    
    # Show database stats
    print("\n5. Database performance:")
    db_stats = db.get_stats()
    print(f"  Query count: {db_stats['query_count']}")
    print(f"  Avg query time: {db_stats['avg_query_time']*1000:.2f}ms")


def demonstrate_batch_recognition(db: FaceDatabase):
    """Demonstrate batch recognition processing."""
    print("\n6. Testing batch recognition...")
    
    if not NUMPY_AVAILABLE:
        print("  (Skipped - NumPy not available)")
        return
    
    # Generate multiple test encodings
    test_encodings = []
    for i in range(5):
        np.random.seed(i)
        encoding = np.random.rand(128)
        test_encodings.append(encoding)
    
    # Process batch
    start_time = time.time()
    batch_results = []
    
    for idx, encoding in enumerate(test_encodings):
        matches = db.find_known_matches(encoding, tolerance=0.6)
        batch_results.append({
            'index': idx,
            'matches': len(matches),
            'best_match': matches[0] if matches else None
        })
    
    elapsed = time.time() - start_time
    
    print(f"  Processed {len(test_encodings)} faces in {elapsed*1000:.2f}ms")
    print(f"  Average: {(elapsed/len(test_encodings))*1000:.2f}ms per face")
    
    for result in batch_results:
        if result['best_match']:
            print(f"    Face {result['index']}: {result['best_match'].person_name} ({result['best_match'].confidence:.2%})")
        else:
            print(f"    Face {result['index']}: Unknown")


def main():
    """Run face recognition demonstration."""
    print("="*60)
    print("Face Recognition Engine - Integration Example")
    print("="*60)
    
    # Setup database
    db_path = "/tmp/example_faces.db"
    db = setup_example_database(db_path)
    
    # Run demonstrations
    demonstrate_recognition_workflow(db)
    demonstrate_batch_recognition(db)
    
    # Cleanup
    db.close()
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
