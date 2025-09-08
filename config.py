"""
Configurações da API de Sumarização de Textos.
"""
import os
from typing import List, Optional
from pydantic import validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configurações da aplicação usando Pydantic Settings."""
    
    # Configurações da API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    
    # Configurações de Segurança
    secret_key: str = "your-secret-key-change-in-production"
    allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Token do Hugging Face
    huggingface_token: Optional[str] = None
    
    # Configurações de Sumarização
    default_max_length: int = 150
    default_min_length: int = 30
    max_text_length: int = 50000  # Limite de segurança para texto de entrada
    abstractive_threshold: int = 500
    extractive_threshold: int = 1000
    ambiguous_buffer: int = 100
    
    # Configurações de Performance
    model_cache_size: int = 1
    request_timeout: int = 300  # 5 minutos
    max_concurrent_requests: int = 10
    
    # Configurações de Cache
    redis_url: Optional[str] = None
    cache_ttl: int = 3600  # 1 hora
    
    # Configurações de Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configurações do Modelo
    model_name: str = "csebuetnlp/mT5_multilingual_XLSum"
    max_input_length: int = 512
    default_sentences_count: int = 3
    
    @validator('huggingface_token')
    def validate_huggingface_token(cls, v):
        """Valida se o token do Hugging Face foi fornecido."""
        if not v:
            # Token padrão para desenvolvimento
            return "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"
        return v
    
    @validator('allowed_origins', pre=True)
    def parse_allowed_origins(cls, v):
        """Parse allowed origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Instância global das configurações
settings = Settings()
