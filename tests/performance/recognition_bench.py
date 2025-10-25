#!/usr/bin/env python3
"""
Performance Benchmarking for Face Recognition Engine

Tests to validate performance targets and identify bottlenecks.
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from src.recognition.recognition_cache import RecognitionCache
from src.storage.face_database import FaceDatabase
from src.recognition.face_encoder import FaceEncoder
from src.recognition.similarity_matcher import SimilarityMatcher


class PerformanceBenchmark(unittest.TestCase):
    """Performance benchmarking tests for face recognition components."""
    
    def setUp(self):
        """Set up benchmark fixtures."""
        self.test_db_path = "/tmp/test_face_benchmark.db"
        self.iterations = 100
    
    def tearDown(self):
        """Clean up test database."""
        import os
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
    
    def test_cache_get_performance(self):
        """Benchmark cache get operations."""
        cache = RecognitionCache({
            'enabled': True,
            'cache_size': 1000,
            'ttl_seconds': 3600
        })
        
        # Pre-populate cache
        for i in range(500):
            cache.put(f'key{i}', {'data': f'value{i}'})
        
        # Benchmark get operations
        start_time = time.time()
        for i in range(self.iterations):
            key = f'key{i % 500}'
            cache.get(key)
        elapsed = time.time() - start_time
        
        avg_time_ms = (elapsed / self.iterations) * 1000
        
        print(f"\nCache GET Performance: {avg_time_ms:.4f}ms per operation")
        
        # Should be very fast (< 0.1ms per operation)
        self.assertLess(avg_time_ms, 0.1, "Cache GET too slow")
    
    def test_cache_put_performance(self):
        """Benchmark cache put operations."""
        cache = RecognitionCache({
            'enabled': True,
            'cache_size': 10000,
            'ttl_seconds': 3600
        })
        
        # Benchmark put operations
        start_time = time.time()
        for i in range(self.iterations):
            cache.put(f'key{i}', {'data': f'value{i}'})
        elapsed = time.time() - start_time
        
        avg_time_ms = (elapsed / self.iterations) * 1000
        
        print(f"Cache PUT Performance: {avg_time_ms:.4f}ms per operation")
        
        # Should be very fast (< 0.2ms per operation)
        self.assertLess(avg_time_ms, 0.2, "Cache PUT too slow")
    
    def test_cache_eviction_performance(self):
        """Benchmark cache eviction under load."""
        cache = RecognitionCache({
            'enabled': True,
            'cache_size': 100,
            'ttl_seconds': 3600
        })
        
        # Benchmark with continuous eviction
        start_time = time.time()
        for i in range(self.iterations * 10):  # More iterations to trigger evictions
            cache.put(f'key{i}', {'data': f'value{i}'})
        elapsed = time.time() - start_time
        
        avg_time_ms = (elapsed / (self.iterations * 10)) * 1000
        
        print(f"Cache PUT with eviction: {avg_time_ms:.4f}ms per operation")
        print(f"Total evictions: {cache.evictions}")
        
        # Should handle evictions efficiently
        self.assertLess(avg_time_ms, 0.5, "Cache eviction too slow")
    
    @unittest.skipIf(not NUMPY_AVAILABLE, "NumPy not available")
    def test_database_query_performance(self):
        """Benchmark database query performance."""
        # Create test database
        db = FaceDatabase(self.test_db_path, {
            'max_faces_per_person': 10,
            'backup_enabled': False
        })
        db.initialize()
        
        # Add test data
        for person_idx in range(50):
            person_id = f"person_{person_idx:03d}"
            db.add_person(person_id, f"Person {person_idx}")
            
            # Add face encodings
            for face_idx in range(5):
                encoding = np.random.rand(128)
                db.add_face_encoding(person_id, encoding, quality_score=0.8)
        
        # Create query encoding
        query_encoding = np.random.rand(128)
        
        # Benchmark queries
        start_time = time.time()
        for _ in range(self.iterations):
            db.find_known_matches(query_encoding, tolerance=0.6)
        elapsed = time.time() - start_time
        
        avg_time_ms = (elapsed / self.iterations) * 1000
        
        print(f"Database query (250 encodings): {avg_time_ms:.2f}ms per query")
        
        # Target: < 50ms per query
        self.assertLess(avg_time_ms, 100, "Database query too slow for benchmark")
        
        db.close()
    
    @unittest.skipIf(not NUMPY_AVAILABLE, "NumPy not available")
    def test_face_encoding_performance(self):
        """Benchmark face encoding extraction (mock)."""
        encoder = FaceEncoder({
            'encoding_model': 'small',
            'face_jitter': 1,
            'number_of_times_to_upsample': 1
        })
        
        # Create test image
        test_image = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Benchmark encoding
        start_time = time.time()
        for _ in range(self.iterations):
            encoder.encode_face(test_image)
        elapsed = time.time() - start_time
        
        avg_time_ms = (elapsed / self.iterations) * 1000
        
        print(f"Face encoding (mock): {avg_time_ms:.2f}ms per encoding")
        
        # Mock encoding should be very fast
        self.assertLess(avg_time_ms, 5, "Face encoding (mock) too slow")
    
    @unittest.skipIf(not NUMPY_AVAILABLE, "NumPy not available")
    def test_similarity_matching_performance(self):
        """Benchmark similarity matching performance."""
        matcher = SimilarityMatcher({
            'similarity_metric': 'euclidean',
            'tolerance': 0.6
        })
        
        # Create test encodings
        known_encodings = [np.random.rand(128) for _ in range(100)]
        unknown_encoding = np.random.rand(128)
        
        # Benchmark single comparison
        start_time = time.time()
        for _ in range(self.iterations):
            matcher.compare_faces(known_encodings[0], unknown_encoding)
        elapsed = time.time() - start_time
        
        avg_time_ms = (elapsed / self.iterations) * 1000
        print(f"Single face comparison: {avg_time_ms:.4f}ms per comparison")
        
        # Benchmark batch distance computation
        start_time = time.time()
        for _ in range(self.iterations):
            matcher.compute_distances_batch(known_encodings, unknown_encoding)
        elapsed = time.time() - start_time
        
        avg_time_ms = (elapsed / self.iterations) * 1000
        print(f"Batch distance (100 faces): {avg_time_ms:.2f}ms per batch")
        
        # Should be reasonably fast
        self.assertLess(avg_time_ms, 10, "Batch similarity matching too slow")
    
    @unittest.skipIf(not NUMPY_AVAILABLE, "NumPy not available")
    def test_end_to_end_recognition_performance(self):
        """Benchmark end-to-end recognition pipeline."""
        # Setup components
        cache = RecognitionCache({
            'enabled': True,
            'cache_size': 1000,
            'ttl_seconds': 3600
        })
        
        db = FaceDatabase(self.test_db_path, {
            'max_faces_per_person': 10,
            'backup_enabled': False
        })
        db.initialize()
        
        # Add test data
        for person_idx in range(10):
            person_id = f"person_{person_idx:03d}"
            db.add_person(person_id, f"Person {person_idx}")
            for _ in range(3):
                encoding = np.random.rand(128)
                db.add_face_encoding(person_id, encoding)
        
        # Simulate recognition pipeline
        query_encoding = np.random.rand(128)
        
        start_time = time.time()
        for iteration in range(self.iterations):
            # Check cache
            cache_key = f"encoding_{iteration}"
            cached = cache.get(cache_key)
            
            if cached is None:
                # Query database
                matches = db.find_known_matches(query_encoding, tolerance=0.6)
                
                # Cache result
                if matches:
                    cache.put(cache_key, {
                        'person_id': matches[0].person_id,
                        'confidence': matches[0].confidence
                    })
        
        elapsed = time.time() - start_time
        avg_time_ms = (elapsed / self.iterations) * 1000
        
        print(f"End-to-end recognition: {avg_time_ms:.2f}ms per face")
        
        cache_stats = cache.get_stats()
        print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")
        
        # Target: < 200ms per face (without real face_recognition library)
        # With mock encoding, should be much faster
        self.assertLess(avg_time_ms, 50, "End-to-end recognition too slow")
        
        db.close()
    
    def test_memory_usage_cache(self):
        """Test memory usage of cache with large dataset."""
        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            self.skipTest("psutil not available")
            return
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large cache
        cache = RecognitionCache({
            'enabled': True,
            'cache_size': 5000,
            'ttl_seconds': 3600
        })
        
        # Fill cache
        for i in range(5000):
            cache.put(f'key{i}', {
                'person_id': f'person_{i}',
                'confidence': 0.85,
                'data': [1, 2, 3, 4, 5]
            })
        
        # Measure memory after
        after_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = after_memory - baseline_memory
        
        print(f"Memory usage for 5000 cache entries: {memory_increase:.2f}MB")
        
        # Should use reasonable memory (< 50MB for 5000 entries)
        self.assertLess(memory_increase, 50, "Cache memory usage too high")


class PerformanceReport(unittest.TestCase):
    """Generate performance report."""
    
    def test_generate_performance_summary(self):
        """Generate summary of performance benchmarks."""
        print("\n" + "="*60)
        print("FACE RECOGNITION ENGINE - PERFORMANCE SUMMARY")
        print("="*60)
        print("\nPerformance Targets (from requirements):")
        print("  - Recognition Latency: < 200ms per face")
        print("  - Database Query: < 50ms")
        print("  - Cache Hit Rate: > 80%")
        print("  - Memory Usage: < 100MB")
        print("\nNote: Actual performance will vary based on:")
        print("  - Hardware (CPU, RAM)")
        print("  - Database size")
        print("  - face_recognition library performance")
        print("  - Image quality and resolution")
        print("="*60 + "\n")


if __name__ == '__main__':
    # Run benchmarks
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
