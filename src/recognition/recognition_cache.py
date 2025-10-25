#!/usr/bin/env python3
"""
Recognition Cache

LRU cache for face recognition results to improve performance.
"""

import time
import logging
from typing import Dict, Any, Optional
from collections import OrderedDict

logger = logging.getLogger(__name__)


class RecognitionCache:
    """LRU cache for face recognition results."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize recognition cache.
        
        Args:
            config: Configuration dictionary with cache settings
        """
        self.config = config
        self.max_size = config.get('cache_size', 1000)
        self.default_ttl = config.get('ttl_seconds', 3600)
        self.enabled = config.get('enabled', True)
        
        # Cache storage: key -> (value, timestamp, ttl)
        self.cache: OrderedDict[str, tuple] = OrderedDict()
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expired_entries = 0
        
        logger.info(f"RecognitionCache initialized with size={self.max_size}, ttl={self.default_ttl}s")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if not self.enabled:
            return None
        
        if key not in self.cache:
            self.misses += 1
            return None
        
        # Get cached entry
        value, timestamp, ttl = self.cache[key]
        
        # Check if expired
        if time.time() - timestamp > ttl:
            # Remove expired entry
            del self.cache[key]
            self.expired_entries += 1
            self.misses += 1
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        self.hits += 1
        
        return value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
        """
        if not self.enabled:
            return
        
        if ttl is None:
            ttl = self.default_ttl
        
        # Check if we need to evict
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Remove least recently used item
            self.cache.popitem(last=False)
            self.evictions += 1
        
        # Add or update entry
        self.cache[key] = (value, time.time(), ttl)
        
        # Move to end if updating existing key
        if len(self.cache) > 1:
            self.cache.move_to_end(key)
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cache entry.
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if entry was found and removed, False otherwise
        """
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def invalidate_person(self, person_id: str) -> int:
        """
        Invalidate all cache entries for a person.
        
        Args:
            person_id: Person ID to invalidate
            
        Returns:
            Number of entries invalidated
        """
        keys_to_remove = []
        
        # Find all keys related to this person
        for key, (value, _, _) in self.cache.items():
            if isinstance(value, dict) and value.get('person_id') == person_id:
                keys_to_remove.append(key)
        
        # Remove found keys
        for key in keys_to_remove:
            del self.cache[key]
        
        return len(keys_to_remove)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        logger.info("Recognition cache cleared")
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.
        
        Returns:
            Number of expired entries removed
        """
        current_time = time.time()
        keys_to_remove = []
        
        for key, (_, timestamp, ttl) in self.cache.items():
            if current_time - timestamp > ttl:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
        
        if keys_to_remove:
            self.expired_entries += len(keys_to_remove)
            logger.debug(f"Removed {len(keys_to_remove)} expired cache entries")
        
        return len(keys_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions,
            'expired_entries': self.expired_entries,
            'enabled': self.enabled
        }
    
    def save_state(self) -> None:
        """Save cache state (placeholder for future persistence)."""
        # This could be extended to save cache to disk
        logger.debug(f"Cache state: {len(self.cache)} entries")
