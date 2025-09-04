# API de Sumarização de Textos

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Uma API robusta e eficiente para sumarização de textos em português, desenvolvida com FastAPI e modelos de inteligência artificial avançados.

## 📋 Visão Geral

Esta API oferece dois métodos principais de sumarização:

- **Sumarização Extrativa**: Seleciona as sentenças mais importantes do texto original usando algoritmos de processamento de linguagem natural
- **Sumarização Abstrativa**: Gera novos textos concisos usando modelos de linguagem T5 treinados especificamente para tarefas de sumarização

A API é capaz de processar textos de qualquer tamanho, utilizando técnicas de chunking inteligente para textos longos, e oferece controle total sobre o comprimento dos resumos gerados.

## ✨ Funcionalidades Principais

### Métodos de Sumarização
- **Extrativo**: Baseado em seleção de sentenças-chave usando algoritmo LSA (Latent Semantic Analysis)
- **Abstrativo**: Geração de texto usando modelo mT5 multilingual otimizado para português

### Controle de Parâmetros
- `max_length`: Comprimento máximo do resumo (padrão: 150 caracteres, máximo: 1000)
- `min_length`: Comprimento mínimo do resumo (padrão: 30 caracteres, mínimo: 10)
- Validação automática de parâmetros com mensagens de erro detalhadas

### Processamento Avançado
- **Chunking Inteligente**: Divisão automática de textos longos em partes menores
- **Logging Detalhado**: Monitoramento completo de todas as operações
- **Tratamento de Erros**: Captura e logging de exceções com informações úteis
- **Validação de Entrada**: Verificação robusta de todos os parâmetros

## 🚀 Instalação e Configuração

### Pré-requisitos

- **Python**: 3.8 ou superior
- **Git**: Para clonar o repositório
- **Token do Hugging Face**: Necessário para acessar modelos (gratuito)

### Instalação por Sistema Operacional

#### Windows

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/api-sumarizacao-textos.git
cd api-sumarizacao-textos

# 2. Crie um ambiente virtual
python -m venv .venv1

# 3. Ative o ambiente virtual
.venv1\Scripts\activate

# 4. Instale as dependências
pip install -r requirements.txt

# 5. Configure o token do Hugging Face
# Crie um arquivo chamado API_HuggingFace no diretório raiz
# Cole seu token do Hugging Face (obtenha em: https://huggingface.co/settings/tokens)
echo "hf_seu_token_aqui" > API_HuggingFace
```

#### Linux/macOS

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/api-sumarizacao-textos.git
cd api-sumarizacao-textos

# 2. Crie um ambiente virtual
python3 -m venv .venv1

# 3. Ative o ambiente virtual
source .venv1/bin/activate

# 4. Instale as dependências
pip install -r requirements.txt

# 5. Configure o token do Hugging Face
echo "hf_seu_token_aqui" > API_HuggingFace
```

### Dependências Principais

```txt
fastapi>=0.100.0
uvicorn>=0.20.0
pydantic>=2.0.0
transformers>=4.20.0
torch>=2.0.0
sumy>=0.11.0
nltk>=3.8.0
protobuf>=4.21.0
huggingface-hub>=0.15.0
```

### Executando Localmente

```bash
# Ative o ambiente virtual
.venv1\Scripts\activate  # Windows
# ou
source .venv1/bin/activate  # Linux/macOS

# Execute a API
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

A API estará disponível em:
- **Documentação Interativa**: http://127.0.0.1:8000/docs
- **Documentação Alternativa**: http://127.0.0.1:8000/redoc
- **Endpoint Raiz**: http://127.0.0.1:8000/
- **Verificação de Saúde**: http://127.0.0.1:8000/health

## 📖 Exemplos de Uso

### Verificar Status da API

```bash
curl http://127.0.0.1:8000/
```

**Resposta:**
```json
{
  "message": "Bem-vindo à API de Sumarização de Textos!",
  "docs": "/docs",
  "methods": ["extractive", "abstractive"],
  "version": "2.0.0"
}
```

### Verificar Saúde do Sistema

```bash
curl http://127.0.0.1:8000/health
```

**Resposta:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-04T20:44:11.628Z",
  "version": "2.0.0",
  "model": "csebuetnlp/mT5_multilingual_XLSum"
}
```

### Sumarização Extrativa

```bash
curl -X POST "http://127.0.0.1:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "A inteligência artificial está revolucionando diversos setores da sociedade. Desde o diagnóstico médico até a análise financeira, os algoritmos de IA demonstram capacidades impressionantes. No entanto, é fundamental garantir que seu desenvolvimento seja ético e responsável.",
    "method": "extractive",
    "max_length": 200,
    "min_length": 50
  }'
```

### Sumarização Abstrativa

```bash
curl -X POST "http://127.0.0.1:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "A inteligência artificial está revolucionando diversos setores da sociedade. Desde o diagnóstico médico até a análise financeira, os algoritmos de IA demonstram capacidades impressionantes. No entanto, é fundamental garantir que seu desenvolvimento seja ético e responsável.",
    "method": "abstractive",
    "max_length": 150,
    "min_length": 30
  }'
```

### Exemplo com Python

```python
import requests

# Configuração da requisição
url = "http://127.0.0.1:8000/summarize"
payload = {
    "text": "Texto longo que você deseja resumir...",
    "method": "abstractive",
    "max_length": 200,
    "min_length": 50
}

# Fazendo a requisição
response = requests.post(url, json=payload)
result = response.json()

print("Resumo:", result["summary"])
```

## 📚 Documentação da API

### Modelos de Dados

#### TextInput
Modelo para entrada de dados de sumarização.

```python
class TextInput(BaseModel):
    text: str                           # Texto a ser sumarizado (obrigatório)
    method: str = "extractive"          # Método: 'extractive' ou 'abstractive'
    max_length: int = 150              # Comprimento máximo em caracteres
    min_length: int = 30               # Comprimento mínimo em caracteres
```

**Exemplo:**
```json
{
  "text": "A inteligência artificial está transformando o mundo...",
  "method": "abstractive",
  "max_length": 200,
  "min_length": 50
}
```

#### SummaryOutput
Modelo para saída de dados de sumarização.

```python
class SummaryOutput(BaseModel):
    summary: str  # Texto resumido gerado
```

**Exemplo:**
```json
{
  "summary": "A IA está revolucionando diversos setores, desde medicina até finanças, mas seu desenvolvimento deve ser ético."
}
```

### Endpoints

#### GET /
Retorna informações básicas sobre a API.

- **URL**: `/`
- **Método**: GET
- **Resposta**: Informações da API

#### GET /health
Verifica o status de saúde da API.

- **URL**: `/health`
- **Método**: GET
- **Resposta**: Status do sistema

#### POST /summarize
Realiza a sumarização do texto fornecido.

- **URL**: `/summarize`
- **Método**: POST
- **Corpo**: Objeto TextInput
- **Resposta**: Objeto SummaryOutput

**Códigos de Status:**
- `200`: Sucesso
- `400`: Erro de validação (parâmetros inválidos)
- `500`: Erro interno do servidor

## 🔧 Detalhes Técnicos

### Método Extrativo

1. **Pré-processamento**: Tokenização e análise linguística usando NLTK
2. **Análise de Sentenças**: Extração de features das sentenças (comprimento, posição, palavras-chave)
3. **Pontuação LSA**: Aplicação do algoritmo Latent Semantic Analysis para identificar sentenças mais relevantes
4. **Seleção**: Escolha das N sentenças com maior pontuação
5. **Pós-processamento**: Ajuste do comprimento baseado nos parâmetros `max_length` e `min_length`

**Vantagens:**
- Preserva o texto original
- Mais rápido e eficiente
- Menos propenso a erros factuais

### Método Abstrativo

1. **Pré-processamento**: Adição do prefixo "summarize: " para orientar o modelo T5
2. **Tokenização**: Conversão do texto em tokens usando o tokenizer do mT5
3. **Chunking (se necessário)**: Divisão em partes menores se o texto exceder 512 tokens
4. **Geração**: Uso do pipeline de sumarização com parâmetros otimizados:
   - `temperature=0.3`: Controle de aleatoriedade
   - `top_p=0.9`: Nucleus sampling
   - `top_k=50`: Limitação de candidatos
   - `num_beams=4`: Busca por feixe para qualidade
   - `repetition_penalty=1.2`: Penalização de repetições
5. **Pós-processamento**: Validação do comprimento e ajuste se necessário

**Parâmetros Avançados:**
- **Beam Search**: Explora múltiplas possibilidades de geração para encontrar o melhor resumo
- **Repetition Penalty**: Evita frases repetidas no resumo
- **Length Control**: Garante que o resumo respeite os limites especificados

### Validações e Logging

**Validações Implementadas:**
- Verificação de texto vazio
- Validação de `max_length > min_length`
- Limites de `max_length` (máximo 1000)
- Limites de `min_length` (mínimo 10)

**Sistema de Logging:**
- Logs estruturados com timestamps
- Níveis: INFO, WARNING, ERROR
- Informações sobre processamento de chunks
- Valores de parâmetros utilizados
- Tempos de processamento
- Detalhes de erros ocorridos

### Processamento de Textos Longos

Para textos que excedem o limite do modelo (512 tokens):

1. **Divisão Inteligente**: Prioriza quebras por sentenças para manter coerência
2. **Fallback por Tokens**: Se a divisão por sentenças falhar, divide por tokens
3. **Distribuição de Comprimento**: Aloca o `max_length` entre os chunks
4. **Combinação**: Junta os resumos parciais
5. **Resumo Final**: Se necessário, gera um resumo do resumo combinado

### Diretrizes de Contribuição
- Siga o estilo de código PEP 8
- Adicione testes para novas funcionalidades
- Atualize a documentação conforme necessário
- Mantenha compatibilidade com versões anteriores

## 📞 Suporte

Para suporte ou dúvidas:
- Abra uma issue no GitHub
- Consulte a documentação em `/docs`
- Verifique os logs da aplicação para diagnóstico

## 🔄 Changelog

### Versão 2.0.0
- ✨ Reformulação completa do método abstrativo
- 🚀 Adição de controle de parâmetros `max_length` e `min_length`
- 📊 Implementação de logging detalhado
- 🔧 Chunking inteligente para textos longos
- 🛡️ Validação robusta de entrada
- 📚 Documentação aprimorada

### Versão 1.0.0
- ✅ Implementação básica com métodos extrativo e abstrativo
- ✅ API funcional com FastAPI
- ✅ Integração com modelos Hugging Face

---


**Desenvolvido com ❤️ usando FastAPI e Transformers**
