# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Importe as funções que você criou
from summarizer import summarize_extractive

app = FastAPI(
    title="API de Sumarização de Textos",
    description="Uma API para sumarização de textos usando abordagens extrativa.",
    version="1.0.0"
)

# Modelo de entrada
class TextInput(BaseModel):
    text: str
    method: str = "extractive" # 'extractive' ou 'abstractive'

# Modelo de saída
class SummaryOutput(BaseModel):
    summary: str

@app.post("/summarize", response_model=SummaryOutput)
async def get_summary(payload: TextInput):
    """
    Recebe um texto e retorna seu resumo.
    - **text**: O texto a ser sumarizado.
    - **method**: 'extractive' (padrão) ou 'abstractive'.
    """
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="O campo de texto não pode ser vazio.")

    # ... (código anterior do endpoint)
    if payload.method == "extractive":
        summary = summarize_extractive(payload.text)
    else:
        raise HTTPException(status_code=400, detail="Método inválido. Atualmente, apenas 'extractive' é suportado.")
    # ... (resto do código)

    return SummaryOutput(summary=summary)

@app.get("/")
async def root():
    return {"message": "Bem-vindo à API de Sumarização! Acesse /docs para a documentação."}
