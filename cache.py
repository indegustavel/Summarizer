"""
Sistema de cache para otimização de performance.
"""
import hashlib
import json
import time
from typing import Optional, Any, Dict
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class MemoryCache:
    """Cache em memória simples."""
    
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        """
        Inicializa o cache em memória.
        
        Args:
            max_size: Tamanho máximo do cache
            ttl: Time to live em segundos
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Gera chave única baseada nos argumentos."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """Verifica se uma entrada expirou."""
        if key not in self._cache:
            return True
        
        entry_time = self._cache[key].get('timestamp', 0)
        return time.time() - entry_time > self.ttl
    
    def _evict_oldest(self):
        """Remove a entrada mais antiga."""
        if not self._access_times:
            return
        
        oldest_key = min(self._access_times, key=self._access_times.get)
        self._cache.pop(oldest_key, None)
        self._access_times.pop(oldest_key, None)
    
    def get(self, key: str) -> Optional[Any]:
        """Recupera valor do cache."""
        if key in self._cache and not self._is_expired(key):
            self._access_times[key] = time.time()
            logger.debug(f"Cache hit para chave: {key}")
            return self._cache[key]['value']
        
        # Remove entrada expirada
        if key in self._cache:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)
        
        logger.debug(f"Cache miss para chave: {key}")
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Armazena valor no cache."""
        # Remove entradas expiradas
        expired_keys = [k for k in self._cache.keys() if self._is_expired(k)]
        for k in expired_keys:
            self._cache.pop(k, None)
            self._access_times.pop(k, None)
        
        # Evict se necessário
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        self._cache[key] = {
            'value': value,
            'timestamp': time.time()
        }
        self._access_times[key] = time.time()
        logger.debug(f"Valor armazenado no cache com chave: {key}")
    
    def clear(self) -> None:
        """Limpa todo o cache."""
        self._cache.clear()
        self._access_times.clear()
        logger.info("Cache limpo")
    
    def stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache."""
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'ttl': self.ttl,
            'entries': list(self._cache.keys())
        }


# Cache global
cache = MemoryCache(max_size=50, ttl=1800)  # 30 minutos TTL


def cached(ttl: Optional[int] = None):
    """
    Decorator para cache de funções.
    
    Args:
        ttl: Time to live específico para esta função
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Gerar chave única
            key = f"{func.__name__}:{hashlib.md5(str(args).encode() + str(sorted(kwargs.items())).encode()).hexdigest()}"
            
            # Tentar recuperar do cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Executar função e cachear resultado
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        return wrapper
    return decorator


class CacheManager:
    """Gerenciador de cache com funcionalidades avançadas."""
    
    def __init__(self):
        self.cache = cache
    
    def get_summary_cache_key(self, text: str, method: str, max_length: int, min_length: int) -> str:
        """Gera chave de cache para sumarização."""
        content = f"{text[:100]}_{method}_{max_length}_{min_length}"
        return f"summary:{hashlib.md5(content.encode()).hexdigest()}"
    
    def get_cached_summary(self, text: str, method: str, max_length: int, min_length: int) -> Optional[str]:
        """Recupera sumário do cache."""
        key = self.get_summary_cache_key(text, method, max_length, min_length)
        return self.cache.get(key)
    
    def cache_summary(self, text: str, method: str, max_length: int, min_length: int, summary: str) -> None:
        """Armazena sumário no cache."""
        key = self.get_summary_cache_key(text, method, max_length, min_length)
        self.cache.set(key, summary)
    
    def invalidate_summary_cache(self, text_pattern: Optional[str] = None) -> None:
        """Invalida cache de sumários."""
        if text_pattern:
            # Invalidar apenas entradas que contenham o padrão
            keys_to_remove = []
            for key in self.cache._cache.keys():
                if text_pattern in str(self.cache._cache[key].get('value', '')):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self.cache._cache.pop(key, None)
                self.cache._access_times.pop(key, None)
        else:
            # Limpar todo o cache
            self.cache.clear()


# Instância global do gerenciador de cache
cache_manager = CacheManager()
