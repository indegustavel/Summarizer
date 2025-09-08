"""
Módulo de modelos com lazy loading e gerenciamento de recursos.
"""
import logging
from typing import Optional, Dict, Any
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Gerenciador de modelos com lazy loading."""
    
    def __init__(self):
        self._summarizer_pipeline: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None
        self._is_loaded = False
        self._loading_lock = False
    
    def _load_model(self) -> None:
        """Carrega o modelo de sumarização."""
        if self._is_loaded or self._loading_lock:
            return
        
        self._loading_lock = True
        try:
            logger.info(f"Carregando modelo: {settings.model_name}")
            
            # Carregar tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.model_name,
                use_auth_token=settings.huggingface_token
            )
            
            # Carregar modelo
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                settings.model_name,
                use_auth_token=settings.huggingface_token
            )
            
            # Criar pipeline
            self._summarizer_pipeline = pipeline(
                "summarization",
                model=self._model,
                tokenizer=self._tokenizer,
                device=-1  # CPU por padrão
            )
            
            self._is_loaded = True
            logger.info("Modelo carregado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise
        finally:
            self._loading_lock = False
    
    def get_summarizer_pipeline(self):
        """Retorna o pipeline de sumarização, carregando se necessário."""
        if not self._is_loaded:
            self._load_model()
        return self._summarizer_pipeline
    
    def get_tokenizer(self):
        """Retorna o tokenizer, carregando se necessário."""
        if not self._is_loaded:
            self._load_model()
        return self._tokenizer
    
    def get_model(self):
        """Retorna o modelo, carregando se necessário."""
        if not self._is_loaded:
            self._load_model()
        return self._model
    
    def is_loaded(self) -> bool:
        """Verifica se o modelo está carregado."""
        return self._is_loaded
    
    def unload_model(self) -> None:
        """Descarrega o modelo para liberar memória."""
        if self._is_loaded:
            logger.info("Descarregando modelo")
            self._summarizer_pipeline = None
            self._tokenizer = None
            self._model = None
            self._is_loaded = False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo."""
        return {
            "model_name": settings.model_name,
            "is_loaded": self._is_loaded,
            "max_input_length": settings.max_input_length,
            "default_sentences_count": settings.default_sentences_count
        }


# Instância global do gerenciador de modelos
model_manager = ModelManager()
