"""
Advanced caching module for DARF framework.

This module provides a thread-safe caching implementation with support for TTL,
automatic cleanup, and performance metrics. It implements the Component interface
for standardized lifecycle management.
"""

import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple, Union, Callable, List, TypeVar, Generic, Type
import json
import pickle
from datetime import datetime
import asyncio
import functools

from src.interfaces.component import Component
from src.errors import DARFError, ConfigurationError
from src.config.config_manager import config_manager
from src.types.common_types import Result

logger = logging.getLogger("DARF.Cache")

T = TypeVar('T')

class CacheError(DARFError):
    """Base exception for cache-related errors."""
    pass


class CacheSerializationError(CacheError):
    """Raised when serialization or deserialization fails."""
    pass


class CacheItem(Generic[T]):
    """Represents a single item in the cache with metadata."""
    
    def __init__(self, value: T, expiry_time: Optional[float] = None):
        """
        Initialize a cache item.
        
        Args:
            value: The cached value
            expiry_time: Time when the item expires (None for no expiry)
        """
        self.value = value
        self.expiry_time = expiry_time
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.access_count = 0
    
    def is_expired(self, now: Optional[float] = None) -> bool:
        """
        Check if the item is expired.
        
        Args:
            now: Current time (uses time.time() if not provided)
            
        Returns:
            Whether the item is expired
        """
        if self.expiry_time is None:
            return False
        current_time = now if now is not None else time.time()
        return current_time > self.expiry_time
    
    def access(self) -> None:
        """Update access metadata when the item is retrieved."""
        self.last_accessed = time.time()
        self.access_count += 1


class Serializer:
    """Base class for cache serializers."""
    
    @staticmethod
    def serialize(value: Any) -> bytes:
        """
        Serialize a value to bytes.
        
        Args:
            value: Value to serialize
            
        Returns:
            Serialized value as bytes
            
        Raises:
            CacheSerializationError: If serialization fails
        """
        try:
            return pickle.dumps(value)
        except Exception as e:
            raise CacheSerializationError(f"Failed to serialize value: {e}", "cache")
    
    @staticmethod
    def deserialize(data: bytes) -> Any:
        """
        Deserialize bytes to a value.
        
        Args:
            data: Serialized data
            
        Returns:
            Deserialized value
            
        Raises:
            CacheSerializationError: If deserialization fails
        """
        try:
            return pickle.loads(data)
        except Exception as e:
            raise CacheSerializationError(f"Failed to deserialize value: {e}", "cache")


class JsonSerializer(Serializer):
    """JSON serializer for the cache."""
    
    @staticmethod
    def serialize(value: Any) -> bytes:
        """Serialize a value to JSON bytes."""
        try:
            return json.dumps(value).encode('utf-8')
        except Exception as e:
            raise CacheSerializationError(f"Failed to serialize value to JSON: {e}", "cache")
    
    @staticmethod
    def deserialize(data: bytes) -> Any:
        """Deserialize JSON bytes to a value."""
        try:
            return json.loads(data.decode('utf-8'))
        except Exception as e:
            raise CacheSerializationError(f"Failed to deserialize JSON: {e}", "cache")


class AdvancedCache(Component):
    """
    Thread-safe cache implementation with TTL support and comprehensive metrics.
    
    Implements the Component interface for lifecycle management and provides
    a rich feature set including:
    - TTL support with automatic cleanup
    - Thread safety for concurrent access
    - Size limits with configurable eviction policies
    - Performance metrics and statistics
    - Multiple serialization options for different data types
    """
    
    def __init__(self, component_id: str = "cache", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cache with configuration.
        
        Args:
            component_id: Unique identifier for this component
            config: Configuration dictionary for this component
        """
        super().__init__(component_id, config)
        
        # Load configuration
        config_dict = config or {}
        if not config:
            # Try to load from global config
            cache_config = config_manager.get_section("cache", {})
            config_dict = cache_config
        
        # Configure cache parameters
        self.cache_type = config_dict.get("cache_type", "memory")
        self.default_ttl = config_dict.get("default_ttl", 300)  # Default 5 minutes
        self.max_size = config_dict.get("max_size", 1000)  # Default max 1000 items
        self.cleanup_interval = config_dict.get("cleanup_interval", 60)  # Cleanup every minute
        
        # Configure serializer
        serializer_type = config_dict.get("serializer", "pickle")
        self.serializer = self._get_serializer(serializer_type)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Cache storage - {key: CacheItem}
        self.cache = {}
        
        # Cache metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0
        self.total_response_time = 0
        self.request_count = 0
        
        # Async cleanup task
        self._cleanup_task = None
        self._running = False
        
        self.logger.info(f"Initialized {self.cache_type} cache with TTL={self.default_ttl}s")
    
    def _get_serializer(self, serializer_type: str) -> Type[Serializer]:
        """
        Get the serializer class for the specified type.
        
        Args:
            serializer_type: Type of serializer ('json' or 'pickle')
            
        Returns:
            Serializer class
        """
        if serializer_type.lower() == 'json':
            return JsonSerializer
        return Serializer
    
    async def start(self) -> Result[bool]:
        """
        Start the cache component.
        
        Begins automatic cleanup of expired items.
        
        Returns:
            Result containing success status or error
        """
        try:
            self._running = True
            
            # Start background cleanup task
            if self.cleanup_interval > 0:
                self.logger.info(f"Starting cache cleanup task (interval: {self.cleanup_interval}s)")
                
                async def cleanup_loop():
                    while self._running:
                        try:
                            count = self.cleanup()
                            if count > 0:
                                self.logger.debug(f"Cleaned up {count} expired cache items")
                        except Exception as e:
                            self.logger.error(f"Error in cache cleanup: {e}")
                        
                        await asyncio.sleep(self.cleanup_interval)
                
                self._cleanup_task = asyncio.create_task(cleanup_loop())
            
            return Result.success(True)
        except Exception as e:
            self.logger.error(f"Failed to start cache component: {e}")
            return Result.failure(CacheError(f"Failed to start cache: {e}", "cache"))
    
    async def stop(self) -> Result[bool]:
        """
        Stop the cache component.
        
        Stops automatic cleanup and clears the cache.
        
        Returns:
            Result containing success status or error
        """
        try:
            self._running = False
            
            # Cancel cleanup task if running
            if self._cleanup_task is not None:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
                self._cleanup_task = None
            
            # Clear the cache
            with self._lock:
                self.cache.clear()
            
            self.logger.info("Cache component stopped and cleared")
            return Result.success(True)
        except Exception as e:
            self.logger.error(f"Error stopping cache component: {e}")
            return Result.failure(CacheError(f"Failed to stop cache: {e}", "cache"))
    
    def get(self, cache_type: str, key: Any) -> Any:
        """
        Get a value from the cache.
        
        Args:
            cache_type: Category of the cached item
            key: Cache key
        
        Returns:
            Cached value or None if not found or expired
        """
        cache_key = f"{cache_type}:{str(key)}"
        start_time = time.time()
        
        with self._lock:
            # Check if key exists
            if cache_key not in self.cache:
                self.misses += 1
                return None
            
            # Get the cache item
            cache_item = self.cache[cache_key]
            
            # Check if expired
            if cache_item.is_expired():
                # Remove expired item
                del self.cache[cache_key]
                self.misses += 1
                self.expirations += 1
                return None
            
            # Update access metadata
            cache_item.access()
            
            # Valid cache hit
            self.hits += 1
            
            # Record response time
            end_time = time.time()
            self.total_response_time += (end_time - start_time)
            self.request_count += 1
            
            return cache_item.value
    
    def set(self, cache_type: str, key: Any, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            cache_type: Category of the cached item
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for no expiry)
        """
        cache_key = f"{cache_type}:{str(key)}"
        
        # Apply TTL (use default if not specified)
        if ttl is None:
            ttl = self.default_ttl
        
        # Calculate expiry time (None if ttl is negative)
        expiry_time = None if ttl < 0 else time.time() + ttl
        
        with self._lock:
            # Check if we need to clean up to make space
            if len(self.cache) >= self.max_size:
                self._cleanup_oldest(10)  # Remove 10% of oldest entries
            
            # Store value with expiry time
            self.cache[cache_key] = CacheItem(value, expiry_time)
    
    def get_or_set(self, cache_type: str, key: Any, value_func: Callable[[], T], ttl: Optional[int] = None) -> T:
        """
        Get a value from the cache, or set it if not found.
        
        Args:
            cache_type: Category of the cached item
            key: Cache key
            value_func: Function to call to get the value if not in cache
            ttl: Time-to-live in seconds (None for no expiry)
            
        Returns:
            Cached or computed value
        """
        # Try to get from cache first
        result = self.get(cache_type, key)
        if result is not None:
            return result
        
        # Not in cache, compute the value
        computed_value = value_func()
        
        # Store in cache
        self.set(cache_type, key, computed_value, ttl)
        
        return computed_value
    
    def delete(self, cache_type: str, key: Any) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            cache_type: Category of the cached item
            key: Cache key
            
        Returns:
            True if the key was deleted, False if it didn't exist
        """
        cache_key = f"{cache_type}:{str(key)}"
        
        with self._lock:
            if cache_key in self.cache:
                del self.cache[cache_key]
                return True
            return False
    
    def clear(self, cache_type: Optional[str] = None) -> int:
        """
        Clear all cache entries or entries of a specific type.
        
        Args:
            cache_type: If provided, only clear entries of this type
            
        Returns:
            Number of entries cleared
        """
        with self._lock:
            if cache_type is None:
                count = len(self.cache)
                self.cache.clear()
                return count
            
            # Only clear entries of the specified type
            prefix = f"{cache_type}:"
            keys_to_delete = [k for k in self.cache.keys() if k.startswith(prefix)]
            for k in keys_to_delete:
                del self.cache[k]
            
            return len(keys_to_delete)
    
    def cleanup(self) -> int:
        """
        Remove all expired items from the cache.
        
        Returns:
            Number of items removed
        """
        now = time.time()
        count = 0
        
        with self._lock:
            keys_to_delete = []
            
            # Find expired items
            for key, item in self.cache.items():
                if item.is_expired(now):
                    keys_to_delete.append(key)
            
            # Delete expired items
            for key in keys_to_delete:
                del self.cache[key]
            
            count = len(keys_to_delete)
            self.expirations += count
        
        return count
    
    def _cleanup_oldest(self, percent: int = 10) -> None:
        """
        Remove oldest entries when cache is full.
        
        Args:
            percent: Percentage of cache to clear (default: 10%)
        """
        with self._lock:
            cache_size = len(self.cache)
            num_to_remove = max(1, int(cache_size * percent / 100))
            
            # Sort items by last accessed time (oldest first)
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: x[1].last_accessed
            )
            
            # Remove oldest entries
            for i in range(min(num_to_remove, len(sorted_items))):
                key, _ = sorted_items[i]
                del self.cache[key]
            
            self.evictions += num_to_remove
            self.logger.debug(f"Removed {num_to_remove} oldest cache entries (cache size: {cache_size})")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            # Count expired items without removing them
            now = time.time()
            expired_count = sum(1 for item in self.cache.values() if item.is_expired(now))
            
            stats = {
                "type": self.cache_type,
                "size": len(self.cache),
                "expired_count": expired_count,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "expirations": self.expirations,
                "hit_ratio": self.hits / max(self.hits + self.misses, 1),
                "max_size": self.max_size,
                "default_ttl": self.default_ttl,
                "serializer": self.serializer.__name__,
            }
            
            # Add average response time if available
            if self.request_count > 0:
                stats["avg_response_time_ms"] = (self.total_response_time / self.request_count) * 1000
            
            return stats
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get component status information.
        
        Returns:
            Dictionary with component status
        """
        status = super().get_status()
        stats = self.get_stats()
        
        status.update({
            "cache_stats": stats,
            "cleanup_task_running": self._cleanup_task is not None and not self._cleanup_task.done()
        })
        
        return status
    
    def serialize(self, value: Any) -> bytes:
        """
        Serialize a value using the configured serializer.
        
        Args:
            value: Value to serialize
            
        Returns:
            Serialized value as bytes
        """
        return self.serializer.serialize(value)
    
    def deserialize(self, data: bytes) -> Any:
        """
        Deserialize bytes using the configured serializer.
        
        Args:
            data: Serialized data
            
        Returns:
            Deserialized value
        """
        return self.serializer.deserialize(data)


# Decorator for caching function results
def cached(cache_type: str, key_func: Optional[Callable] = None, ttl: Optional[int] = None):
    """
    Decorator for caching function results.
    
    Args:
        cache_type: Category for the cached items
        key_func: Function to generate the cache key (defaults to using function args)
        ttl: Time-to-live in seconds (None for default TTL)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the cache instance
            cache = None
            from src.utils.dependency_injection import container
            try:
                cache = container.get(AdvancedCache)
            except KeyError:
                # Fallback to global instance
                cache = global_cache
                
            if cache is None:
                # No cache available, just execute the function
                return func(*args, **kwargs)
            
            # Generate cache key
            if key_func is not None:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key is function name + args + kwargs
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)
            
            # Try to get from cache
            result = cache.get(cache_type, cache_key)
            if result is not None:
                return result
            
            # Not in cache, execute the function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_type, cache_key, result, ttl)
            
            return result
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get the cache instance
            cache = None
            from src.utils.dependency_injection import container
            try:
                cache = container.get(AdvancedCache)
            except KeyError:
                # Fallback to global instance
                cache = global_cache
                
            if cache is None:
                # No cache available, just execute the function
                return await func(*args, **kwargs)
            
            # Generate cache key
            if key_func is not None:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key is function name + args + kwargs
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)
            
            # Try to get from cache
            result = cache.get(cache_type, cache_key)
            if result is not None:
                return result
            
            # Not in cache, execute the function
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_type, cache_key, result, ttl)
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator


# Create a global cache instance with default settings
global_cache = AdvancedCache("global_cache", {
    "cache_type": "memory",
    "default_ttl": 300,
    "max_size": 1000,
    "cleanup_interval": 60,
    "serializer": "pickle"
})
