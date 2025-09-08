# 🚀 API de Sumarização de Textos

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)

Uma API robusta, segura e de alta performance para sumarização de textos em português, desenvolvida com FastAPI e modelos de inteligência artificial avançados.

## 📋 Visão Geral

Esta API oferece três métodos principais de sumarização:

- **Sumarização Extrativa**: Seleciona as sentenças mais importantes do texto original usando algoritmos de processamento de linguagem natural
- **Sumarização Abstrativa**: Gera novos textos concisos usando modelos de linguagem T5 treinados especificamente para tarefas de sumarização
- **Sumarização Automática**: Seleciona inteligentemente o melhor método baseado na análise do texto

A API é capaz de processar textos de qualquer tamanho, utilizando técnicas de chunking inteligente para textos longos, e oferece controle total sobre o comprimento dos resumos gerados.

## ✨ Funcionalidades Principais

### Métodos de Sumarização
- **Extrativo**: Baseado em seleção de sentenças-chave usando algoritmo LSA (Latent Semantic Analysis)
- **Abstrativo**: Geração de texto usando modelo mT5 multilingual otimizado para português
- **Automático**: Seleção inteligente do método baseado na análise do texto

### 🆕 Funcionalidades Avançadas
- **Cache Inteligente**: Sistema de cache em memória com TTL configurável
- **Processamento Paralelo**: Chunks processados simultaneamente para textos longos
- **Lazy Loading**: Modelos carregados apenas quando necessário
- **Validação de Segurança**: Proteção contra XSS, validação de entrada robusta
- **Timeouts Configuráveis**: Controle de tempo limite para operações
- **Monitoramento**: Endpoints de saúde, estatísticas de cache e performance

### Controle de Parâmetros
- `max_length`: Comprimento máximo do resumo (padrão: 150 caracteres, máximo: 1000)
- `min_length`: Comprimento mínimo do resumo (padrão: 30 caracteres, mínimo: 10)
- `max_text_length`: Limite de segurança para texto de entrada (padrão: 50.000 caracteres)
- Validação automática de parâmetros com mensagens de erro detalhadas

## 🚀 Instalação e Configuração

### Pré-requisitos

- **Python**: 3.8 ou superior
- **Git**: Para clonar o repositório (opcional)
- **Token do Hugging Face**: Necessário para acessar modelos (gratuito)

### Passo 1: Preparar o Ambiente

#### Windows
```bash
# 1. Navegue até a pasta do projeto
cd "C:\Users\Gustavo\Desktop\Sumarização"

# 2. Crie um ambiente virtual (se não existir)
python -m venv .venv1

# 3. Ative o ambiente virtual
.venv1\Scripts\activate
```

#### Linux/macOS
```bash
# 1. Navegue até a pasta do projeto
cd /caminho/para/seu/projeto

# 2. Crie um ambiente virtual (se não existir)
python3 -m venv .venv1

# 3. Ative o ambiente virtual
source .venv1/bin/activate
```

### Passo 2: Instalar Dependências

```bash
# Instalar todas as dependências necessárias
pip install -r requirements.txt
```

### Passo 3: Configurar Token do Hugging Face

1. **Obter Token**:
   - Acesse: https://huggingface.co/settings/tokens
   - Crie uma conta gratuita se necessário
   - Gere um novo token

2. **Configurar Token** (Opcional):
   - O token já está configurado no código para desenvolvimento
   - Para produção, configure a variável de ambiente `HUGGINGFACE_TOKEN`

### Passo 4: Iniciar a API

```bash
# Iniciar o servidor
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🌐 Como Usar a API

### Acessar a Documentação

Após iniciar a API, acesse no seu navegador:

- **Documentação Interativa**: http://localhost:8000/docs
- **Documentação Alternativa**: http://localhost:8000/redoc
- **Página Inicial**: http://localhost:8000/

### Endpoints Disponíveis

#### 1. **POST /summarize** - Sumarização de Texto
Endpoint principal para sumarizar textos.

**Exemplo de Requisição:**
```json
{
  "text": "Este é um texto de exemplo para testar a sumarização. A API pode processar textos longos e gerar resumos concisos usando diferentes métodos de inteligência artificial.",
  "method": "auto",
  "max_length": 100,
  "min_length": 30
}
```

**Exemplo de Resposta:**
```json
{
  "summary": "A API processa textos usando IA para gerar resumos concisos.",
  "method_used": "abstractive",
  "processing_time": 2.34,
  "cached": false,
  "quality_score": 0.85
}
```

#### 2. **GET /health** - Verificação de Saúde
Verifica o status da API e componentes.

**Resposta:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-05T18:04:33.283Z",
  "version": "3.0.0",
  "model": {
    "name": "csebuetnlp/mT5_multilingual_XLSum",
    "status": "loaded"
  },
  "cache": {
    "size": 5,
    "max_size": 50,
    "ttl": 3600
  }
}
```

#### 3. **GET /cache/stats** - Estatísticas do Cache
Retorna informações sobre o cache.

#### 4. **DELETE /cache/clear** - Limpar Cache
Remove todas as entradas do cache.

#### 5. **GET /model/info** - Informações do Modelo
Retorna detalhes sobre o modelo carregado.

### Métodos de Sumarização

#### **Extrativo** (`method: "extractive"`)
- Seleciona as sentenças mais importantes do texto original
- Mais rápido e preserva o texto original
- Ideal para textos estruturados

#### **Abstrativo** (`method: "abstractive"`)
- Gera novo texto usando IA
- Mais flexível e conciso
- Ideal para textos narrativos

#### **Automático** (`method: "auto"`)
- Seleciona automaticamente o melhor método
- Baseado na análise do texto (tamanho, complexidade)
- Recomendado para uso geral

## 💻 Exemplos de Uso

### 1. Usando cURL

```bash
# Sumarização automática
curl -X POST "http://localhost:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "A inteligência artificial está revolucionando diversos setores da sociedade. Desde o diagnóstico médico até a análise financeira, os algoritmos de IA demonstram capacidades impressionantes. No entanto, é fundamental garantir que seu desenvolvimento seja ético e responsável.",
    "method": "auto",
    "max_length": 150,
    "min_length": 50
  }'
```

### 2. Usando Python

```python
import requests

# Configuração da requisição
url = "http://localhost:8000/summarize"
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
print("Método usado:", result["method_used"])
print("Tempo de processamento:", result["processing_time"], "s")
```

### 3. Usando JavaScript (Fetch)

```javascript
const response = await fetch('http://localhost:8000/summarize', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'Seu texto aqui...',
    method: 'auto',
    max_length: 150,
    min_length: 30
  })
});

const result = await response.json();
console.log('Resumo:', result.summary);
```

## 📊 Monitoramento e Estatísticas

### Verificar Status da API
```bash
curl http://localhost:8000/health
```

### Ver Estatísticas do Cache
```bash
curl http://localhost:8000/cache/stats
```

### Limpar Cache
```bash
curl -X DELETE http://localhost:8000/cache/clear
```

## 🔧 Configurações Avançadas

### Parâmetros de Configuração

Você pode modificar as configurações editando o arquivo `config.py`:

```python
# Configurações de Sumarização
DEFAULT_MAX_LENGTH = 150        # Comprimento padrão do resumo
DEFAULT_MIN_LENGTH = 30         # Comprimento mínimo
MAX_TEXT_LENGTH = 50000         # Limite de texto de entrada

# Configurações de Performance
REQUEST_TIMEOUT = 300           # Timeout em segundos
CACHE_TTL = 3600               # TTL do cache em segundos

# Configurações do Modelo
MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"
MAX_INPUT_LENGTH = 512         # Limite de tokens do modelo
```

### Variáveis de Ambiente

Para produção, configure as seguintes variáveis:

```bash
# Token do Hugging Face
export HUGGINGFACE_TOKEN=seu_token_aqui

# Configurações da API
export API_HOST=0.0.0.0
export API_PORT=8000
export LOG_LEVEL=INFO
```

## 🚨 Solução de Problemas

### Problema: "Token do Hugging Face não encontrado"
**Solução**: O token já está configurado no código. Se necessário, configure a variável de ambiente `HUGGINGFACE_TOKEN`.

### Problema: "Erro de importação"
**Solução**: Verifique se todas as dependências foram instaladas:
```bash
pip install -r requirements.txt
```

### Problema: "Porta já em uso"
**Solução**: Use uma porta diferente:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### Problema: "Modelo não carrega"
**Solução**: 
1. Verifique sua conexão com a internet
2. Confirme se o token do Hugging Face é válido
3. Aguarde o carregamento inicial (pode demorar alguns minutos)

## 📁 Estrutura do Projeto

```
Sumarização/
├── main.py              # Aplicação principal FastAPI
├── summarizer.py        # Lógica de sumarização
├── config.py           # Configurações da aplicação
├── security.py         # Validação e segurança
├── cache.py            # Sistema de cache
├── models.py           # Gerenciamento de modelos
├── requirements.txt    # Dependências Python
└── README.md          # Este arquivo
```

## 🔄 Changelog

### Versão 3.0.0 (Atual)
- 🔒 **Segurança**: Validação avançada, sanitização de entrada, proteção XSS
- ⚡ **Performance**: Cache inteligente, lazy loading, processamento paralelo
- 🏗️ **Arquitetura**: Separação de responsabilidades, configuração externa
- 🛡️ **Robustez**: Timeouts configuráveis, retry logic, tratamento de erros
- 📊 **Monitoramento**: Endpoints de saúde, estatísticas, métricas

---

## 🎯 Resumo Rápido

1. **Ativar ambiente virtual**: `.venv1\Scripts\activate` (Windows)
2. **Instalar dependências**: `pip install -r requirements.txt`
3. **Iniciar API**: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`
4. **Acessar**: http://localhost:8000/docs
5. **Testar**: Use a interface interativa ou faça requisições POST para `/summarize`
