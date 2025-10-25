#!/usr/bin/env python3
"""
Test suite for Recognition Cache

Tests for LRU cache functionality, TTL, and performance metrics.
"""

import sys
import time
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.recognition.recognition_cache import RecognitionCache


class TestRecognitionCache(unittest.TestCase):
    """Test suite for RecognitionCache class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache_config = {
            'enabled': True,
            'cache_size': 5,  # Small size for testing
            'ttl_seconds': 2  # Short TTL for testing
        }
        self.cache = RecognitionCache(self.cache_config)
    
    def test_cache_initialization(self):
        """Test cache initialization with config."""
        self.assertTrue(self.cache.enabled)
        self.assertEqual(self.cache.max_size, 5)
        self.assertEqual(self.cache.default_ttl, 2)
        self.assertEqual(len(self.cache.cache), 0)
    
    def test_cache_put_and_get(self):
        """Test basic put and get operations."""
        test_data = {'person_id': 'person_001', 'confidence': 0.85}
        
        self.cache.put('test_key', test_data)
        result = self.cache.get('test_key')
        
        self.assertIsNotNone(result)
        self.assertEqual(result['person_id'], 'person_001')
        self.assertEqual(result['confidence'], 0.85)
    
    def test_cache_miss(self):
        """Test cache miss for non-existent key."""
        result = self.cache.get('non_existent_key')
        
        self.assertIsNone(result)
        self.assertEqual(self.cache.misses, 1)
    
    def test_cache_hit_tracking(self):
        """Test cache hit/miss tracking."""
        self.cache.put('key1', 'value1')
        
        # Hit
        self.cache.get('key1')
        self.assertEqual(self.cache.hits, 1)
        self.assertEqual(self.cache.misses, 0)
        
        # Miss
        self.cache.get('key2')
        self.assertEqual(self.cache.hits, 1)
        self.assertEqual(self.cache.misses, 1)
    
    def test_cache_ttl_expiration(self):
        """Test TTL expiration of cache entries."""
        self.cache.put('expiring_key', 'expiring_value', ttl=0.5)
        
        # Should be available immediately
        result = self.cache.get('expiring_key')
        self.assertIsNotNone(result)
        
        # Wait for expiration
        time.sleep(0.6)
        
        # Should be expired
        result = self.cache.get('expiring_key')
        self.assertIsNone(result)
        self.assertEqual(self.cache.expired_entries, 1)
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        # Fill cache to capacity
        for i in range(5):
            self.cache.put(f'key{i}', f'value{i}')
        
        self.assertEqual(len(self.cache.cache), 5)
        
        # Add one more - should evict least recently used
        self.cache.put('key5', 'value5')
        
        self.assertEqual(len(self.cache.cache), 5)
        self.assertEqual(self.cache.evictions, 1)
        
        # key0 should be evicted
        result = self.cache.get('key0')
        self.assertIsNone(result)
    
    def test_cache_lru_order(self):
        """Test LRU ordering updates on access."""
        # Add items
        self.cache.put('key1', 'value1')
        self.cache.put('key2', 'value2')
        self.cache.put('key3', 'value3')
        
        # Access key1 to make it recently used
        self.cache.get('key1')
        
        # Fill cache
        self.cache.put('key4', 'value4')
        self.cache.put('key5', 'value5')
        
        # Add one more - should evict key2 (oldest unused)
        self.cache.put('key6', 'value6')
        
        # key1 should still be present
        result = self.cache.get('key1')
        self.assertIsNotNone(result)
        
        # key2 should be evicted
        result = self.cache.get('key2')
        self.assertIsNone(result)
    
    def test_cache_update_existing(self):
        """Test updating existing cache entry."""
        self.cache.put('key1', 'original_value')
        self.cache.put('key1', 'updated_value')
        
        result = self.cache.get('key1')
        self.assertEqual(result, 'updated_value')
        
        # Should not count as eviction
        self.assertEqual(self.cache.evictions, 0)
    
    def test_cache_invalidate(self):
        """Test cache invalidation."""
        self.cache.put('key1', 'value1')
        
        # Verify it's there
        self.assertIsNotNone(self.cache.get('key1'))
        
        # Invalidate
        result = self.cache.invalidate('key1')
        self.assertTrue(result)
        
        # Should be gone
        self.assertIsNone(self.cache.get('key1'))
        
        # Invalidating non-existent key
        result = self.cache.invalidate('non_existent')
        self.assertFalse(result)
    
    def test_cache_invalidate_person(self):
        """Test invalidating all entries for a person."""
        # Add entries for different persons
        self.cache.put('face1', {'person_id': 'person_001', 'conf': 0.8})
        self.cache.put('face2', {'person_id': 'person_001', 'conf': 0.9})
        self.cache.put('face3', {'person_id': 'person_002', 'conf': 0.7})
        
        # Invalidate person_001
        count = self.cache.invalidate_person('person_001')
        
        self.assertEqual(count, 2)
        self.assertIsNone(self.cache.get('face1'))
        self.assertIsNone(self.cache.get('face2'))
        self.assertIsNotNone(self.cache.get('face3'))
    
    def test_cache_clear(self):
        """Test clearing entire cache."""
        # Add several entries
        for i in range(3):
            self.cache.put(f'key{i}', f'value{i}')
        
        self.assertEqual(len(self.cache.cache), 3)
        
        # Clear cache
        self.cache.clear()
        
        self.assertEqual(len(self.cache.cache), 0)
        for i in range(3):
            self.assertIsNone(self.cache.get(f'key{i}'))
    
    def test_cache_cleanup_expired(self):
        """Test cleanup of expired entries."""
        # Add entries with short TTL
        for i in range(3):
            self.cache.put(f'key{i}', f'value{i}', ttl=0.5)
        
        # Add entry with long TTL
        self.cache.put('persistent_key', 'persistent_value', ttl=10)
        
        # Wait for short TTL to expire
        time.sleep(0.6)
        
        # Run cleanup
        cleaned = self.cache.cleanup_expired()
        
        self.assertEqual(cleaned, 3)
        self.assertIsNone(self.cache.get('key0'))
        self.assertIsNotNone(self.cache.get('persistent_key'))
    
    def test_cache_stats(self):
        """Test cache statistics."""
        # Perform various operations
        self.cache.put('key1', 'value1')
        self.cache.put('key2', 'value2')
        self.cache.get('key1')  # Hit
        self.cache.get('key3')  # Miss
        
        stats = self.cache.get_stats()
        
        self.assertEqual(stats['size'], 2)
        self.assertEqual(stats['max_size'], 5)
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertEqual(stats['hit_rate'], 0.5)
        self.assertTrue(stats['enabled'])
    
    def test_cache_disabled(self):
        """Test cache behavior when disabled."""
        disabled_cache = RecognitionCache({'enabled': False})
        
        disabled_cache.put('key1', 'value1')
        result = disabled_cache.get('key1')
        
        self.assertIsNone(result)
        self.assertEqual(len(disabled_cache.cache), 0)
    
    def test_cache_custom_ttl(self):
        """Test cache with custom TTL per entry."""
        # Short TTL
        self.cache.put('short_key', 'short_value', ttl=0.5)
        
        # Long TTL
        self.cache.put('long_key', 'long_value', ttl=5)
        
        # Wait for short TTL
        time.sleep(0.6)
        
        # Short should be expired
        self.assertIsNone(self.cache.get('short_key'))
        
        # Long should still be valid
        self.assertIsNotNone(self.cache.get('long_key'))
    
    def test_cache_save_state(self):
        """Test cache state saving (placeholder)."""
        self.cache.put('key1', 'value1')
        
        # Should not raise exception
        try:
            self.cache.save_state()
        except Exception as e:
            self.fail(f"save_state raised exception: {e}")


class TestRecognitionCachePerformance(unittest.TestCase):
    """Performance tests for RecognitionCache."""
    
    def test_cache_performance_large_dataset(self):
        """Test cache performance with larger dataset."""
        cache = RecognitionCache({
            'enabled': True,
            'cache_size': 1000,
            'ttl_seconds': 3600
        })
        
        # Add many entries
        start_time = time.time()
        for i in range(500):
            cache.put(f'key{i}', {'data': f'value{i}'})
        put_time = time.time() - start_time
        
        # Access entries
        start_time = time.time()
        for i in range(500):
            cache.get(f'key{i}')
        get_time = time.time() - start_time
        
        # Should be reasonably fast
        self.assertLess(put_time, 1.0, "Put operations too slow")
        self.assertLess(get_time, 0.5, "Get operations too slow")
        
        # Verify hit rate
        stats = cache.get_stats()
        self.assertEqual(stats['hit_rate'], 1.0)
    
    def test_cache_eviction_performance(self):
        """Test performance of cache eviction."""
        cache = RecognitionCache({
            'enabled': True,
            'cache_size': 100,
            'ttl_seconds': 3600
        })
        
        # Add entries beyond capacity
        start_time = time.time()
        for i in range(200):
            cache.put(f'key{i}', f'value{i}')
        elapsed = time.time() - start_time
        
        # Should handle evictions efficiently
        self.assertLess(elapsed, 1.0)
        self.assertEqual(len(cache.cache), 100)
        self.assertEqual(cache.evictions, 100)


if __name__ == '__main__':
    unittest.main()
