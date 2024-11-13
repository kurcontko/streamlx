# src/core/cache.py
from typing import Optional, Any, Callable
import time
from functools import wraps


class TTLCache:
    """Simple cache implementation with time-to-live."""
    def __init__(self, ttl: int):
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.timestamps[key]
        return None

    def set(self, key: str, value: Any):
        """Set value in cache with current timestamp."""
        self.cache[key] = value
        self.timestamps[key] = time.time()

def cache_with_ttl(ttl: int):
    """Decorator for caching function results with TTL."""
    cache = TTLCache(ttl)
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            result = cache.get(key)
            if result is None:
                result = func(*args, **kwargs)
                cache.set(key, result)
            return result
        return wrapper
    return decorator