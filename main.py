# main.py
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Importe as funções que você criou
from summarizer import summarize_extractive, summarize_abstractive

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

    ## Métodos Disponíveis
    - `extractive`: Método tradicional baseado em extração de sentenças
    - `abstractive`: Método moderno usando IA generativa com controle de qualidade

    ## Limites
    - Texto máximo: Ilimitado (processado em chunks)
    - max_length: Até 1000 caracteres
    - min_length: Mínimo 10 caracteres
    """,
    version="2.0.0"
)

# Modelo de entrada
class TextInput(BaseModel):
    text: str
    method: str = "extractive"  # Método de sumarização
    max_length: int = 150       # Comprimento máximo do resumo em caracteres
    min_length: int = 30        # Comprimento mínimo do resumo em caracteres

    class Config:
        """Configuração do modelo Pydantic."""
        json_schema_extra = {
            "example": {
                "text": "Este é um exemplo de texto longo que será resumido pela API. A API pode processar textos de diferentes tamanhos e gerar resumos concisos usando métodos extrativos ou abstrativos.",
                "method": "abstractive",
                "max_length": 200,
                "min_length": 50
            }
        }

# Modelo de saída
class SummaryOutput(BaseModel):
    summary: str

@app.post("/summarize", response_model=SummaryOutput)
async def get_summary(payload: TextInput):
    """
    Recebe um texto e retorna seu resumo usando métodos extrativo ou abstrativo.

    - **text**: O texto a ser sumarizado (obrigatório).
    - **method**: Método de sumarização - 'extractive' (padrão) ou 'abstractive'.
    - **max_length**: Comprimento máximo do resumo em caracteres (padrão: 150, máximo: 1000).
    - **min_length**: Comprimento mínimo do resumo em caracteres (padrão: 30, mínimo: 10).

    O método extrativo seleciona as sentenças mais importantes do texto original.
    O método abstrativo gera um novo texto que resume o conteúdo de forma concisa.
    """
    logger.info(f"Recebida requisição de sumarização. Método: {payload.method}, Texto length: {len(payload.text)}")

    # Validação de entrada
    if not payload.text or not payload.text.strip():
        logger.warning("Tentativa de sumarização com texto vazio")
        raise HTTPException(status_code=400, detail="O campo de texto não pode ser vazio.")

    # Validar parâmetros
    if payload.max_length <= payload.min_length:
        logger.warning(f"Parâmetros inválidos: max_length ({payload.max_length}) <= min_length ({payload.min_length})")
        raise HTTPException(status_code=400, detail="max_length deve ser maior que min_length")
    if payload.max_length > 1000:
        logger.warning(f"max_length muito alto: {payload.max_length}")
        raise HTTPException(status_code=400, detail="max_length não pode exceder 1000 caracteres")
    if payload.min_length < 10:
        logger.warning(f"min_length muito baixo: {payload.min_length}")
        raise HTTPException(status_code=400, detail="min_length deve ser pelo menos 10 caracteres")

    try:
        if payload.method == "extractive":
            logger.info("Iniciando sumarização extrativa")
            summary = summarize_extractive(payload.text, payload.max_length, payload.min_length)
        elif payload.method == "abstractive":
            logger.info("Iniciando sumarização abstrativa")
            summary = summarize_abstractive(payload.text, payload.max_length, payload.min_length)
        else:
            logger.warning(f"Método inválido solicitado: {payload.method}")
            raise HTTPException(status_code=400, detail="Método inválido. Escolha 'extractive' ou 'abstractive'.")

        logger.info(f"Sumarização concluída com sucesso. Resumo length: {len(summary)}")
        return SummaryOutput(summary=summary)

    except Exception as e:
        logger.error(f"Erro durante a sumarização: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor: {str(e)}")

@app.get("/")
async def root():
    """Endpoint raiz que retorna informações sobre a API."""
    logger.info("Acesso ao endpoint raiz")
    return {
        "message": "Bem-vindo à API de Sumarização de Textos!",
        "docs": "/docs",
        "methods": ["extractive", "abstractive"],
        "version": "2.0.0"
    }

@app.get("/health")
async def health_check():
    """Endpoint de verificação de saúde da API."""
    logger.info("Verificação de saúde solicitada")
    return {
        "status": "healthy",
        "timestamp": "2025-09-04T20:26:42.253Z",
        "version": "2.0.0",
        "model": "csebuetnlp/mT5_multilingual_XLSum"
    }
