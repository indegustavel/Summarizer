# main.py
import logging
import asyncio
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import time
from contextlib import asynccontextmanager

from config import settings
from security import security_validator
from models import model_manager
from cache import cache_manager
from summarizer import summarize_extractive, summarize_abstractive, summarize_auto

# Configuração de logging
logging.basicConfig(level=getattr(logging, settings.log_level), format=settings.log_format)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação."""
    # Startup
    logger.info("Iniciando API de Sumarização")
    logger.info(f"Configurações: {settings.model_name}, Cache TTL: {settings.cache_ttl}s")
    
    # Verificar se o modelo pode ser carregado
    try:
        if settings.huggingface_token:
            logger.info("Token do Hugging Face configurado")
        else:
            logger.warning("Token do Hugging Face não configurado")
    except Exception as e:
        logger.error(f"Erro na inicialização: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Finalizando API de Sumarização")
    model_manager.unload_model()
    cache_manager.cache.clear()

app = FastAPI(
    title="API de Sumarização de Textos",
    description="""
    API robusta para sumarização de textos em português usando métodos avançados de IA.

    ## Funcionalidades
    - **Sumarização Extrativa**: Seleciona as sentenças mais importantes do texto original
    - **Sumarização Abstrativa**: Gera novos textos concisos usando modelo de linguagem T5
    - **Controle de Comprimento**: Parâmetros max_length e min_length para personalizar o tamanho do resumo
    - **Processamento de Textos Longos**: Chunking inteligente para textos grandes
    - **Validação de Parâmetros**: Verificação robusta de entrada
    - **Logging Detalhado**: Monitoramento completo das operações
    - **Cache Inteligente**: Sistema de cache para otimização de performance
    - **Processamento Paralelo**: Chunks processados em paralelo para textos longos
    - **Segurança Avançada**: Validação e sanitização de entrada

    ## Métodos Disponíveis
    - `extractive`: Método tradicional baseado em extração de sentenças
    - `abstractive`: Método moderno usando IA generativa com controle de qualidade
    - `auto`: Seleção automática do método baseado na análise do texto

    ## Limites
    - Texto máximo: {max_text_length} caracteres
    - max_length: Até 1000 caracteres
    - min_length: Mínimo 10 caracteres
    """.format(max_text_length=settings.max_text_length),
    version="3.0.0",
    lifespan=lifespan
)

# Middleware de segurança
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # Em produção, especificar hosts confiáveis
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de entrada
class TextInput(BaseModel):
    text: str = Field(..., description="Texto a ser sumarizado", min_length=1)
    method: str = Field(default="auto", description="Método de sumarização")
    max_length: int = Field(default=settings.default_max_length, description="Comprimento máximo do resumo", ge=10, le=1000)
    min_length: int = Field(default=settings.default_min_length, description="Comprimento mínimo do resumo", ge=10, le=500)

    @validator('method')
    def validate_method(cls, v):
        """Valida o método de sumarização."""
        return security_validator.validate_method(v)

    @validator('text')
    def validate_text(cls, v):
        """Valida o texto de entrada."""
        return security_validator.validate_text_input(v)

    @validator('max_length', 'min_length')
    def validate_lengths(cls, v, values):
        """Valida os comprimentos."""
        if 'max_length' in values and 'min_length' in values:
            if values['max_length'] <= values['min_length']:
                raise ValueError('max_length deve ser maior que min_length')
        return v

    class Config:
        """Configuração do modelo Pydantic."""
        json_schema_extra = {
            "example": {
                "text": "Este é um exemplo de texto longo que será resumido pela API. A API pode processar textos de diferentes tamanhos e gerar resumos concisos usando métodos extrativos ou abstrativos.",
                "method": "auto",
                "max_length": 200,
                "min_length": 50
            }
        }

# Modelo de saída
class SummaryOutput(BaseModel):
    summary: str = Field(..., description="Texto resumido gerado")
    method_used: Optional[str] = Field(None, description="Método efetivamente utilizado")
    processing_time: Optional[float] = Field(None, description="Tempo de processamento em segundos")
    cached: Optional[bool] = Field(None, description="Indica se o resultado veio do cache")
    quality_score: Optional[float] = Field(None, description="Score de qualidade do resumo (0-1)")

# Modelo de erro
class ErrorResponse(BaseModel):
    error: str = Field(..., description="Tipo do erro")
    detail: str = Field(..., description="Detalhes do erro")
    timestamp: str = Field(..., description="Timestamp do erro")

@app.post("/summarize", response_model=SummaryOutput)
async def get_summary(payload: TextInput, request: Request):
    """
    Recebe um texto e retorna seu resumo usando métodos extrativo ou abstrativo.

    - **text**: O texto a ser sumarizado (obrigatório).
    - **method**: Método de sumarização - 'extractive', 'abstractive' ou 'auto' (padrão).
    - **max_length**: Comprimento máximo do resumo em caracteres (padrão: 150, máximo: 1000).
    - **min_length**: Comprimento mínimo do resumo em caracteres (padrão: 30, mínimo: 10).

    O método extrativo seleciona as sentenças mais importantes do texto original.
    O método abstrativo gera um novo texto que resume o conteúdo de forma concisa.
    """
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    
    logger.info(f"Requisição de sumarização recebida de {client_ip}. Método: {payload.method}, Texto length: {len(payload.text)}")

    try:
        # Timeout para a operação
        result = await asyncio.wait_for(
            _process_summarization(payload),
            timeout=settings.request_timeout
        )
        
        processing_time = time.time() - start_time
        result['processing_time'] = round(processing_time, 2)
        
        logger.info(f"Sumarização concluída com sucesso em {processing_time:.2f}s. Resumo length: {len(result['summary'])}")
        return SummaryOutput(**result)

    except asyncio.TimeoutError:
        logger.error(f"Timeout na sumarização após {settings.request_timeout}s")
        raise HTTPException(
            status_code=408, 
            detail=f"Timeout na sumarização. Tente com um texto menor ou aumente o timeout."
        )
    except ValueError as e:
        logger.warning(f"Erro de validação: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Erro durante a sumarização após {processing_time:.2f}s: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Erro interno do servidor durante a sumarização."
        )


async def _process_summarization(payload: TextInput) -> Dict[str, Any]:
    """
    Processa a sumarização de forma assíncrona.
    
    Args:
        payload: Dados da requisição
        
    Returns:
        Resultado da sumarização
    """
    # Executar em thread separada para não bloquear o event loop
    loop = asyncio.get_event_loop()
    
    if payload.method == "extractive":
        logger.info("Iniciando sumarização extrativa")
        summary = await loop.run_in_executor(
            None, 
            summarize_extractive, 
            payload.text, 
            payload.max_length, 
            payload.min_length
        )
        return {
            'summary': summary,
            'method_used': 'extractive',
            'cached': False
        }
        
    elif payload.method == "abstractive":
        logger.info("Iniciando sumarização abstrativa")
        summary = await loop.run_in_executor(
            None, 
            summarize_abstractive, 
            payload.text, 
            payload.max_length, 
            payload.min_length
        )
        return {
            'summary': summary,
            'method_used': 'abstractive',
            'cached': False
        }
        
    elif payload.method == "auto":
        logger.info("Iniciando sumarização automática")
        result = await loop.run_in_executor(
            None, 
            summarize_auto, 
            payload.text, 
            payload.max_length, 
            payload.min_length
        )
        logger.info(f"Método selecionado automaticamente: {result['method_selected']}")
        return {
            'summary': result['summary'],
            'method_used': result['method_selected'],
            'cached': result.get('cached', False),
            'quality_score': result.get('quality', {}).get('overall_score')
        }
    
    else:
        raise ValueError(f"Método inválido: {payload.method}")

@app.get("/")
async def root():
    """Endpoint raiz que retorna informações sobre a API."""
    logger.info("Acesso ao endpoint raiz")
    return {
        "message": "Bem-vindo à API de Sumarização de Textos!",
        "docs": "/docs",
        "methods": ["extractive", "abstractive", "auto"],
        "version": "3.0.0",
        "features": [
            "Cache inteligente",
            "Processamento paralelo",
            "Validação de segurança",
            "Timeout configurável",
            "Lazy loading de modelos"
        ]
    }

@app.get("/health")
async def health_check():
    """Endpoint de verificação de saúde da API."""
    logger.info("Verificação de saúde solicitada")
    
    # Verificar status do modelo
    model_status = "loaded" if model_manager.is_loaded() else "not_loaded"
    
    # Verificar cache
    cache_stats = cache_manager.cache.stats()
    
    return {
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "version": "3.0.0",
        "model": {
            "name": settings.model_name,
            "status": model_status
        },
        "cache": {
            "size": cache_stats['size'],
            "max_size": cache_stats['max_size'],
            "ttl": cache_stats['ttl']
        },
        "config": {
            "max_text_length": settings.max_text_length,
            "request_timeout": settings.request_timeout,
            "default_max_length": settings.default_max_length,
            "default_min_length": settings.default_min_length
        }
    }

@app.get("/cache/stats")
async def cache_stats():
    """Retorna estatísticas do cache."""
    logger.info("Estatísticas do cache solicitadas")
    return cache_manager.cache.stats()

@app.delete("/cache/clear")
async def clear_cache():
    """Limpa o cache."""
    logger.info("Limpeza do cache solicitada")
    cache_manager.cache.clear()
    return {"message": "Cache limpo com sucesso"}

@app.get("/model/info")
async def model_info():
    """Retorna informações sobre o modelo."""
    logger.info("Informações do modelo solicitadas")
    return model_manager.get_model_info()

# Handler global de exceções
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler personalizado para exceções HTTP."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP Error",
            detail=exc.detail,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S") + ".000000Z"
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handler para exceções gerais."""
    logger.error(f"Erro não tratado: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="Erro interno do servidor",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S") + ".000000Z"
        ).dict()
    )
