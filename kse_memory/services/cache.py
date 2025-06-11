"""
Cache service for KSE Memory SDK.
"""

import asyncio
import json
import logging
from typing import Any, Optional
from datetime import datetime, timedelta

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..core.interfaces import CacheInterface
from ..core.config import CacheConfig
from ..exceptions import CacheError


logger = logging.getLogger(__name__)


class CacheService(CacheInterface):
    """
    Cache service for KSE Memory SDK.
    
    Supports Redis and in-memory caching for improved performance.
    """
    
    def __init__(self, config: CacheConfig):
        """
        Initialize cache service.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self._redis_client = None
        self._memory_cache = {}
        self._memory_expiry = {}
        self._connected = False
        
        logger.info(f"Cache service initialized with backend: {config.backend}")
    
    async def _connect(self):
        """Connect to cache backend."""
        if self._connected:
            return
        
        if not self.config.enabled:
            logger.info("Cache is disabled")
            self._connected = True
            return
        
        if self.config.backend == "redis":
            if not REDIS_AVAILABLE:
                logger.warning("Redis not available, falling back to memory cache")
                self.config.backend = "memory"
            else:
                try:
                    # Create Redis connection
                    self._redis_client = redis.Redis(
                        host=self.config.host,
                        port=self.config.port,
                        password=self.config.password,
                        db=self.config.db,
                        decode_responses=True,
                        socket_timeout=5,
                        socket_connect_timeout=5,
                    )
                    
                    # Test connection
                    await self._redis_client.ping()
                    logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")
                    
                except Exception as e:
                    logger.warning(f"Failed to connect to Redis: {str(e)}, falling back to memory cache")
                    self.config.backend = "memory"
                    self._redis_client = None
        
        if self.config.backend == "memory":
            logger.info("Using in-memory cache")
        
        self._connected = True
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found, None otherwise
        """
        if not self.config.enabled:
            return None
        
        await self._connect()
        
        try:
            if self.config.backend == "redis" and self._redis_client:
                # Get from Redis
                value = await self._redis_client.get(key)
                if value:
                    return json.loads(value)
                return None
            
            else:
                # Get from memory cache
                if key in self._memory_cache:
                    # Check expiry
                    if key in self._memory_expiry:
                        if datetime.utcnow() > self._memory_expiry[key]:
                            # Expired
                            del self._memory_cache[key]
                            del self._memory_expiry[key]
                            return None
                    
                    return self._memory_cache[key]
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get cache key '{key}': {str(e)}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
            
        Returns:
            True if successful
        """
        if not self.config.enabled:
            return True
        
        await self._connect()
        
        try:
            ttl = ttl or self.config.ttl
            
            if self.config.backend == "redis" and self._redis_client:
                # Set in Redis
                serialized_value = json.dumps(value, default=str)
                if ttl:
                    await self._redis_client.setex(key, ttl, serialized_value)
                else:
                    await self._redis_client.set(key, serialized_value)
                return True
            
            else:
                # Set in memory cache
                self._memory_cache[key] = value
                
                if ttl:
                    expiry_time = datetime.utcnow() + timedelta(seconds=ttl)
                    self._memory_expiry[key] = expiry_time
                
                # Clean up expired entries periodically
                await self._cleanup_memory_cache()
                return True
                
        except Exception as e:
            logger.error(f"Failed to set cache key '{key}': {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful
        """
        if not self.config.enabled:
            return True
        
        await self._connect()
        
        try:
            if self.config.backend == "redis" and self._redis_client:
                # Delete from Redis
                result = await self._redis_client.delete(key)
                return result > 0
            
            else:
                # Delete from memory cache
                if key in self._memory_cache:
                    del self._memory_cache[key]
                
                if key in self._memory_expiry:
                    del self._memory_expiry[key]
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete cache key '{key}': {str(e)}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists
        """
        if not self.config.enabled:
            return False
        
        await self._connect()
        
        try:
            if self.config.backend == "redis" and self._redis_client:
                # Check Redis
                result = await self._redis_client.exists(key)
                return result > 0
            
            else:
                # Check memory cache
                if key in self._memory_cache:
                    # Check expiry
                    if key in self._memory_expiry:
                        if datetime.utcnow() > self._memory_expiry[key]:
                            # Expired
                            del self._memory_cache[key]
                            del self._memory_expiry[key]
                            return False
                    
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to check cache key '{key}': {str(e)}")
            return False
    
    async def clear(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            True if successful
        """
        if not self.config.enabled:
            return True
        
        await self._connect()
        
        try:
            if self.config.backend == "redis" and self._redis_client:
                # Clear Redis database
                await self._redis_client.flushdb()
                return True
            
            else:
                # Clear memory cache
                self._memory_cache.clear()
                self._memory_expiry.clear()
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            return False
    
    async def _cleanup_memory_cache(self):
        """Clean up expired entries from memory cache."""
        try:
            current_time = datetime.utcnow()
            expired_keys = []
            
            for key, expiry_time in self._memory_expiry.items():
                if current_time > expiry_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                if key in self._memory_cache:
                    del self._memory_cache[key]
                del self._memory_expiry[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
        except Exception as e:
            logger.error(f"Failed to cleanup memory cache: {str(e)}")
    
    async def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "backend": self.config.backend,
            "enabled": self.config.enabled,
            "connected": self._connected,
        }
        
        try:
            if self.config.backend == "redis" and self._redis_client:
                # Get Redis stats
                info = await self._redis_client.info()
                stats.update({
                    "redis_version": info.get("redis_version"),
                    "used_memory": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                    "total_commands_processed": info.get("total_commands_processed"),
                })
            
            else:
                # Get memory cache stats
                await self._cleanup_memory_cache()  # Clean up first
                stats.update({
                    "memory_entries": len(self._memory_cache),
                    "memory_expiry_entries": len(self._memory_expiry),
                })
                
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            stats["error"] = str(e)
        
        return stats
    
    async def disconnect(self):
        """Disconnect from cache backend."""
        try:
            if self._redis_client:
                await self._redis_client.close()
                self._redis_client = None
            
            self._memory_cache.clear()
            self._memory_expiry.clear()
            self._connected = False
            
            logger.info("Cache service disconnected")
            
        except Exception as e:
            logger.error(f"Error disconnecting cache service: {str(e)}")


class MemoryCache(CacheInterface):
    """
    Simple in-memory cache implementation.
    
    Useful for testing or when Redis is not available.
    """
    
    def __init__(self, default_ttl: int = 3600):
        """
        Initialize memory cache.
        
        Args:
            default_ttl: Default time to live in seconds
        """
        self.default_ttl = default_ttl
        self._cache = {}
        self._expiry = {}
        
        logger.info("Memory cache initialized")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        if key in self._cache:
            # Check expiry
            if key in self._expiry:
                if datetime.utcnow() > self._expiry[key]:
                    # Expired
                    del self._cache[key]
                    del self._expiry[key]
                    return None
            
            return self._cache[key]
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        self._cache[key] = value
        
        ttl = ttl or self.default_ttl
        if ttl:
            expiry_time = datetime.utcnow() + timedelta(seconds=ttl)
            self._expiry[key] = expiry_time
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        if key in self._cache:
            del self._cache[key]
        
        if key in self._expiry:
            del self._expiry[key]
        
        return True
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        if key in self._cache:
            # Check expiry
            if key in self._expiry:
                if datetime.utcnow() > self._expiry[key]:
                    # Expired
                    del self._cache[key]
                    del self._expiry[key]
                    return False
            
            return True
        
        return False
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        self._cache.clear()
        self._expiry.clear()
        return True