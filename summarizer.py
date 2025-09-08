# summarizer.py
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import re
import nltk
from collections import Counter
from huggingface_hub import login

from config import settings
from models import model_manager
from cache import cache_manager, cached
from security import security_validator

# Configuração de logging
logging.basicConfig(level=getattr(logging, settings.log_level), format=settings.log_format)
logger = logging.getLogger(__name__)

# Login no Hugging Face
if settings.huggingface_token:
    login(settings.huggingface_token)

LANGUAGE = "portuguese"

def preprocess_text(text: str) -> str:
    """
    Pré-processa o texto para melhorar a qualidade da sumarização.
    
    Args:
        text: Texto original
        
    Returns:
        Texto pré-processado
    """
    # Remover caracteres especiais excessivos
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
    
    # Normalizar espaços em branco
    text = re.sub(r'\s+', ' ', text)
    
    # Remover quebras de linha desnecessárias
    text = re.sub(r'\n+', ' ', text)
    
    # Garantir que sentenças terminem com pontuação
    if text and text[-1] not in '.!?':
        text += '.'
    
    return text.strip()

def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """
    Extrai palavras-chave do texto.
    
    Args:
        text: Texto para análise
        top_k: Número de palavras-chave a retornar
        
    Returns:
        Lista de palavras-chave
    """
    # Remover pontuação e converter para minúsculas
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filtrar palavras muito curtas
    words = [word for word in words if len(word) > 3]
    
    # Contar frequência
    word_freq = Counter(words)
    
    # Retornar as mais frequentes
    return [word for word, freq in word_freq.most_common(top_k)]

def calculate_sentence_importance(sentence: str, keywords: List[str], position: int, total_sentences: int) -> float:
    """
    Calcula a importância de uma sentença baseada em múltiplos fatores.
    
    Args:
        sentence: Sentença a ser avaliada
        keywords: Lista de palavras-chave
        position: Posição da sentença no texto
        total_sentences: Total de sentenças
        
    Returns:
        Score de importância (0-1)
    """
    score = 0.0
    
    # Fator 1: Presença de palavras-chave (40%)
    sentence_words = re.findall(r'\b\w+\b', sentence.lower())
    keyword_matches = sum(1 for word in sentence_words if word in keywords)
    if sentence_words:
        score += 0.4 * (keyword_matches / len(sentence_words))
    
    # Fator 2: Posição da sentença (20%)
    # Sentenças no início e fim são mais importantes
    if position < total_sentences * 0.1 or position > total_sentences * 0.9:
        score += 0.2
    elif position < total_sentences * 0.2 or position > total_sentences * 0.8:
        score += 0.1
    
    # Fator 3: Comprimento da sentença (20%)
    # Sentenças de tamanho médio são preferíveis
    sentence_length = len(sentence.split())
    if 10 <= sentence_length <= 25:
        score += 0.2
    elif 5 <= sentence_length <= 35:
        score += 0.1
    
    # Fator 4: Indicadores de importância (20%)
    importance_indicators = [
        'importante', 'significativo', 'principal', 'essencial', 'fundamental',
        'conclusão', 'resultado', 'descoberta', 'pesquisa', 'estudo',
        'portanto', 'assim', 'consequentemente', 'dessa forma'
    ]
    
    for indicator in importance_indicators:
        if indicator in sentence.lower():
            score += 0.05
    
    return min(score, 1.0)

@cached(ttl=3600)  # Cache por 1 hora
def analyze_text(text: str) -> Dict[str, float]:
    """
    Analisa o texto e retorna métricas para seleção do método de sumarização.

    Args:
        text: Texto a ser analisado

    Returns:
        Métricas incluindo word_count, sentence_count, complexity
    """
    words = text.split()
    word_count = len(words)

    sentences = text.split('.')
    sentence_count = len([s for s in sentences if s.strip()])

    # Complexidade estrutural: média de palavras por sentença
    avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0

    # Densidade de informações: proporção de palavras únicas
    unique_words = len(set(words))
    lexical_density = unique_words / word_count if word_count > 0 else 0

    # Complexidade média das palavras (comprimento médio)
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0

    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_words_per_sentence': avg_words_per_sentence,
        'lexical_density': lexical_density,
        'avg_word_length': avg_word_length
    }

@cached(ttl=1800)  # Cache por 30 minutos
def evaluate_quality(text: str, summary: str) -> Dict[str, float]:
    """
    Avalia a qualidade do resumo baseado em métricas avançadas.

    Args:
        text: Texto original
        summary: Resumo gerado

    Returns:
        Métricas de qualidade (coverage, coherence, informativeness, fluency)
    """
    # Pré-processar textos
    original_processed = preprocess_text(text)
    summary_processed = preprocess_text(summary)
    
    # 1. Cobertura de palavras-chave (30%)
    original_keywords = extract_keywords(original_processed, top_k=20)
    summary_keywords = extract_keywords(summary_processed, top_k=10)
    keyword_coverage = len(set(original_keywords).intersection(set(summary_keywords))) / len(original_keywords) if original_keywords else 0
    
    # 2. Cobertura de conteúdo (25%)
    original_words = set(re.findall(r'\b\w+\b', original_processed.lower()))
    summary_words = set(re.findall(r'\b\w+\b', summary_processed.lower()))
    content_coverage = len(original_words.intersection(summary_words)) / len(original_words) if original_words else 0
    
    # 3. Densidade de informação (20%)
    original_sentences = len(re.split(r'[.!?]+', original_processed))
    summary_sentences = len(re.split(r'[.!?]+', summary_processed))
    compression_ratio = len(summary_processed) / len(original_processed) if original_processed else 0
    informativeness = min(1.0, compression_ratio * 2)  # Resumos muito curtos perdem informação
    
    # 4. Fluência e coerência (15%)
    summary_analysis = analyze_text(summary_processed)
    fluency = min(1.0, summary_analysis['lexical_density'] * 0.7 + (1 / (1 + abs(summary_analysis['avg_words_per_sentence'] - 15) / 10)) * 0.3)
    
    # 5. Estrutura e pontuação (10%)
    punctuation_score = 0
    if summary_processed and summary_processed[-1] in '.!?':
        punctuation_score += 0.5
    if len(re.findall(r'[.!?]', summary_processed)) >= summary_sentences * 0.8:
        punctuation_score += 0.5
    
    # Calcular score geral
    overall_score = (
        keyword_coverage * 0.3 +
        content_coverage * 0.25 +
        informativeness * 0.2 +
        fluency * 0.15 +
        punctuation_score * 0.1
    )
    
    return {
        'coverage': content_coverage,
        'keyword_coverage': keyword_coverage,
        'informativeness': informativeness,
        'fluency': fluency,
        'punctuation_score': punctuation_score,
        'overall_score': overall_score,
        'compression_ratio': compression_ratio
    }

def summarize_auto(text: str, max_length: int = None, min_length: int = None) -> Dict[str, any]:
    """
    Método automatizado que seleciona dinamicamente o método de sumarização baseado na análise do texto.

    Args:
        text: Texto a ser resumido
        max_length: Comprimento máximo do resumo
        min_length: Comprimento mínimo do resumo

    Returns:
        Resumo e informações sobre o método selecionado e qualidade
    """
    # Usar valores padrão das configurações se não fornecidos
    max_length = max_length or settings.default_max_length
    min_length = min_length or settings.default_min_length
    
    # Validar entrada
    text = security_validator.validate_text_input(text)
    security_validator.validate_summarization_params(max_length, min_length)
    
    logger.info(f"Iniciando sumarização automática. Texto length: {len(text)}")

    # Verificar cache primeiro
    cached_summary = cache_manager.get_cached_summary(text, "auto", max_length, min_length)
    if cached_summary:
        logger.info("Resumo encontrado no cache")
        return {
            'summary': cached_summary,
            'method_selected': 'auto',
            'cached': True
        }

    # Analisar o texto
    analysis = analyze_text(text)
    word_count = analysis['word_count']

    logger.info(f"Análise do texto: {word_count} palavras, {analysis['sentence_count']} sentenças")

    # Selecionar método baseado em múltiplos fatores
    complexity_score = analysis['lexical_density'] * 0.4 + (analysis['avg_words_per_sentence'] / 20) * 0.3 + (analysis['avg_word_length'] / 8) * 0.3
    
    if word_count <= settings.abstractive_threshold and complexity_score > 0.6:
        selected_method = 'abstractive'
        logger.info("Selecionado método abstrativo (texto pequeno e complexo)")
    elif word_count >= settings.extractive_threshold or complexity_score < 0.4:
        selected_method = 'extractive'
        logger.info("Selecionado método extrativo (texto grande ou simples)")
    else:
        # Caso ambíguo: testar ambos e escolher o melhor
        logger.info("Caso ambíguo: testando ambos os métodos")
        selected_method = 'ambiguous'

    # Gerar resumo
    if selected_method == 'abstractive':
        summary = summarize_abstractive(text, max_length, min_length)
    elif selected_method == 'extractive':
        summary = summarize_extractive(text, max_length, min_length)
    else:
        # Caso ambíguo: gerar ambos e avaliar
        summary_abstractive = summarize_abstractive(text, max_length, min_length)
        summary_extractive = summarize_extractive(text, max_length, min_length)

        quality_abstractive = evaluate_quality(text, summary_abstractive)
        quality_extractive = evaluate_quality(text, summary_extractive)

        if quality_abstractive['overall_score'] > quality_extractive['overall_score']:
            summary = summary_abstractive
            selected_method = 'abstractive'
            logger.info("Fallback: escolhido abstrativo baseado na qualidade")
        else:
            summary = summary_extractive
            selected_method = 'extractive'
            logger.info("Fallback: escolhido extrativo baseado na qualidade")

    # Avaliar qualidade final
    final_quality = evaluate_quality(text, summary)

    # Cachear resultado
    cache_manager.cache_summary(text, "auto", max_length, min_length, summary)

    logger.info(f"Sumarização automática concluída. Método: {selected_method}, Qualidade: {final_quality['overall_score']:.2f}")

    return {
        'summary': summary,
        'method_selected': selected_method,
        'analysis': analysis,
        'quality': final_quality,
        'cached': False
    }

def summarize_extractive(text: str, max_length: int = None, min_length: int = None) -> str:
    """
    Gera um resumo extrativo do texto usando múltiplos algoritmos combinados.

    Args:
        text: Texto a ser resumido
        max_length: Comprimento máximo aproximado do resumo em caracteres
        min_length: Comprimento mínimo aproximado do resumo em caracteres

    Returns:
        Resumo extrativo do texto
    """
    # Usar valores padrão das configurações se não fornecidos
    max_length = max_length or settings.default_max_length
    min_length = min_length or settings.default_min_length
    
    # Validar entrada
    text = security_validator.validate_text_input(text)
    security_validator.validate_summarization_params(max_length, min_length)
    
    logger.info(f"Iniciando sumarização extrativa. Texto length: {len(text)}, max_length: {max_length}, min_length: {min_length}")

    # Verificar cache primeiro
    cached_summary = cache_manager.get_cached_summary(text, "extractive", max_length, min_length)
    if cached_summary:
        logger.info("Resumo extrativo encontrado no cache")
        return cached_summary

    try:
        # Pré-processar o texto
        processed_text = preprocess_text(text)
        
        # Extrair palavras-chave
        keywords = extract_keywords(processed_text, top_k=15)
        logger.info(f"Palavras-chave extraídas: {keywords[:5]}...")
        
        # Dividir em sentenças
        sentences = re.split(r'[.!?]+', processed_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            logger.warning("Texto muito curto para sumarização")
            return processed_text[:max_length]
        
        # Calcular importância de cada sentença
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            importance = calculate_sentence_importance(sentence, keywords, i, len(sentences))
            sentence_scores.append((sentence, importance, i))
        
        # Ordenar por importância
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Estimar número de sentenças baseado no comprimento desejado
        avg_chars_per_sentence = 120
        estimated_sentences = max(1, min(len(sentences), max_length // avg_chars_per_sentence))
        sentences_count = max(2, estimated_sentences)
        
        # Selecionar as melhores sentenças
        selected_sentences = sentence_scores[:sentences_count]
        
        # Ordenar por posição original para manter coerência
        selected_sentences.sort(key=lambda x: x[2])
        
        # Combinar sentenças selecionadas
        summary_sentences = [s[0] for s in selected_sentences]
        result = ". ".join(summary_sentences)
        
        # Garantir que termine com pontuação
        if result and result[-1] not in '.!?':
            result += "."
        
        # Ajustar comprimento se necessário
        if len(result) > max_length:
            # Truncar de forma inteligente
            words = result.split()
            truncated = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 <= max_length - 3:
                    truncated.append(word)
                    current_length += len(word) + 1
                else:
                    break
            
            result = " ".join(truncated) + "..."
        
        elif len(result) < min_length and len(sentence_scores) > sentences_count:
            # Adicionar mais sentenças se necessário
            additional_sentences = []
            for sentence, score, pos in sentence_scores[sentences_count:]:
                if len(result + ". " + sentence) <= max_length:
                    additional_sentences.append(sentence)
                    if len(". ".join(summary_sentences + additional_sentences)) >= min_length:
                        break
            
            if additional_sentences:
                # Ordenar por posição original
                all_sentences = summary_sentences + additional_sentences
                result = ". ".join(all_sentences)
                if result and result[-1] not in '.!?':
                    result += "."

        # Cachear resultado
        cache_manager.cache_summary(text, "extractive", max_length, min_length, result)

        logger.info(f"Resumo extrativo gerado com sucesso. Length: {len(result)}")
        return result

    except Exception as e:
        logger.error(f"Erro na sumarização extrativa: {str(e)}")
        # Fallback para método simples
        try:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            if sentences:
                return ". ".join(sentences[:3]) + "."
            return text[:max_length]
        except:
            return text[:max_length]

def _create_chunks(text: str, max_chunk_size: int) -> List[str]:
    """
    Divide texto em chunks inteligentes com sobreposição para melhor coerência.
    
    Args:
        text: Texto a ser dividido
        max_chunk_size: Tamanho máximo do chunk
        
    Returns:
        Lista de chunks
    """
    # Pré-processar o texto
    processed_text = preprocess_text(text)
    
    # Dividir em sentenças mais inteligentemente
    sentences = re.split(r'[.!?]+', processed_text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    if not sentences:
        return [processed_text[:max_chunk_size]]
    
    chunks = []
    current_chunk = ""
    overlap_sentences = 1  # Sobreposição de 1 sentença
    
    for i, sentence in enumerate(sentences):
        # Adicionar pontuação se necessário
        if sentence and sentence[-1] not in '.!?':
            sentence += "."
        
        # Verificar se cabe no chunk atual
        if len(current_chunk + " " + sentence) <= max_chunk_size:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            # Chunk está cheio, salvar e começar novo
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Começar novo chunk com sobreposição
            if i > 0 and overlap_sentences > 0:
                # Incluir última sentença do chunk anterior
                prev_sentences = current_chunk.split('. ')
                if len(prev_sentences) > 1:
                    overlap = prev_sentences[-1] + "."
                    current_chunk = overlap + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk = sentence
    
    # Adicionar último chunk se não estiver vazio
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Garantir que não temos chunks muito pequenos
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > 50:  # Chunk mínimo de 50 caracteres
            final_chunks.append(chunk)
        elif final_chunks:
            # Combinar com chunk anterior se muito pequeno
            final_chunks[-1] += " " + chunk
    
    return final_chunks if final_chunks else [processed_text[:max_chunk_size]]


def _process_chunk_parallel(chunk: str, chunk_max_length: int, chunk_min_length: int) -> str:
    """
    Processa um chunk individual para sumarização.
    
    Args:
        chunk: Chunk a ser processado
        chunk_max_length: Comprimento máximo do resumo do chunk
        chunk_min_length: Comprimento mínimo do resumo do chunk
        
    Returns:
        Resumo do chunk
    """
    try:
        pipeline = model_manager.get_summarizer_pipeline()
        input_text = f"summarize: {chunk}"
        
        summary = pipeline(
            input_text,
            max_length=chunk_max_length,
            min_length=chunk_min_length,
            do_sample=False,  # Desabilitado para beam search
            temperature=1.0,  # Não usado quando do_sample=False
            top_p=0.95,       # Não usado quando do_sample=False
            top_k=40,         # Não usado quando do_sample=False
            num_beams=6,      # Aumentado para melhor busca
            early_stopping=True,
            no_repeat_ngram_size=2,  # Reduzido para menos repetição
            length_penalty=1.2,      # Aumentado para resumos mais longos
            repetition_penalty=1.3,  # Aumentado para evitar repetições
            diversity_penalty=0.0,   # Desabilitado para evitar conflitos
            num_return_sequences=1
        )
        
        return summary[0]['summary_text'].strip()
    except Exception as e:
        logger.error(f"Erro ao processar chunk: {str(e)}")
        return chunk[:chunk_max_length]  # Fallback: truncar o chunk

def summarize_abstractive(text: str, max_length: int = None, min_length: int = None) -> str:
    """
    Gera um resumo abstrativo do texto usando modelo de linguagem T5.

    Args:
        text: Texto a ser resumido
        max_length: Comprimento máximo do resumo em tokens
        min_length: Comprimento mínimo do resumo em tokens

    Returns:
        Resumo abstrativo do texto
    """
    # Usar valores padrão das configurações se não fornecidos
    max_length = max_length or settings.default_max_length
    min_length = min_length or settings.default_min_length
    
    # Validar entrada
    text = security_validator.validate_text_input(text)
    security_validator.validate_summarization_params(max_length, min_length)
    
    logger.info(f"Iniciando sumarização abstrativa. Texto length: {len(text)}, max_length: {max_length}, min_length: {min_length}")

    # Verificar cache primeiro
    cached_summary = cache_manager.get_cached_summary(text, "abstractive", max_length, min_length)
    if cached_summary:
        logger.info("Resumo abstrativo encontrado no cache")
        return cached_summary

    try:
        # Obter tokenizer e pipeline
        tokenizer = model_manager.get_tokenizer()
        pipeline = model_manager.get_summarizer_pipeline()
        
        # Validar parâmetros
        if max_length > 500:
            logger.warning(f"max_length ({max_length}) muito alto, pode afetar performance")
        if min_length < 10:
            logger.warning(f"min_length ({min_length}) muito baixo, resumo pode ser muito curto")

        max_input_length = settings.max_input_length
        tokens = tokenizer.encode(text, add_special_tokens=True)

        if len(tokens) <= max_input_length:
            # Texto curto - processar diretamente
            logger.info("Processando texto curto diretamente")

            # Adicionar prefixo para T5 (importante para task de sumarização)
            input_text = f"summarize: {text}"

            summary = pipeline(
                input_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,  # Desabilitado para beam search
                temperature=1.0,  # Não usado quando do_sample=False
                top_p=0.95,       # Não usado quando do_sample=False
                top_k=40,         # Não usado quando do_sample=False
                num_beams=6,      # Aumentado para melhor busca
                early_stopping=True,
                no_repeat_ngram_size=2,  # Reduzido para menos repetição
                length_penalty=1.2,      # Aumentado para resumos mais longos
                repetition_penalty=1.3,  # Aumentado para evitar repetições
                diversity_penalty=0.0,   # Desabilitado para evitar conflitos
                num_return_sequences=1
            )

            result = summary[0]['summary_text'].strip()
            logger.info(f"Resumo abstrativo gerado. Length: {len(result)} caracteres")
            
            # Cachear resultado
            cache_manager.cache_summary(text, "abstractive", max_length, min_length, result)
            return result

        else:
            # Texto longo - dividir em chunks e processar em paralelo
            logger.info(f"Texto longo detectado ({len(tokens)} tokens). Dividindo em chunks.")

            # Criar chunks inteligentes
            chunks = _create_chunks(text, max_input_length - 50)  # Margem de segurança
            
            # Fallback para chunking por tokens se necessário
            if not chunks:
                logger.warning("Divisão por sentenças falhou, usando chunking por tokens")
                current_chunk = []
                current_length = 0
                for token in tokens:
                    current_chunk.append(token)
                    current_length += 1
                    if current_length >= max_input_length - 50:
                        chunks.append(tokenizer.decode(current_chunk, skip_special_tokens=True))
                        current_chunk = []
                        current_length = 0
                if current_chunk:
                    chunks.append(tokenizer.decode(current_chunk, skip_special_tokens=True))

            logger.info(f"Texto dividido em {len(chunks)} chunks")

            # Processar chunks em paralelo
            summaries = []
            chunk_max_length = max(50, max_length // len(chunks))  # Distribuir comprimento entre chunks
            chunk_min_length = max(10, min_length // len(chunks))

            # Usar ThreadPoolExecutor para processamento paralelo
            with ThreadPoolExecutor(max_workers=min(len(chunks), 4)) as executor:
                # Submeter todas as tarefas
                future_to_chunk = {
                    executor.submit(_process_chunk_parallel, chunk, chunk_max_length, chunk_min_length): i
                    for i, chunk in enumerate(chunks)
                }
                
                # Coletar resultados na ordem
                chunk_results = [None] * len(chunks)
                for future in as_completed(future_to_chunk):
                    chunk_index = future_to_chunk[future]
                    try:
                        result = future.result()
                        chunk_results[chunk_index] = result
                        logger.info(f"Chunk {chunk_index + 1}/{len(chunks)} processado")
                    except Exception as e:
                        logger.error(f"Erro ao processar chunk {chunk_index + 1}: {str(e)}")
                        chunk_results[chunk_index] = chunks[chunk_index][:chunk_max_length]  # Fallback
                
                summaries = [result for result in chunk_results if result]

            # Combinar resumos dos chunks
            combined_summary = " ".join(summaries)

            # Se o resumo combinado for muito longo, fazer um resumo final
            combined_tokens = tokenizer.encode(combined_summary, add_special_tokens=False)
            if len(combined_tokens) > max_length:
                logger.info("Fazendo resumo final do resumo combinado")

                final_input = f"summarize: {combined_summary}"
                final_summary = pipeline(
                    final_input,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,  # Desabilitado para beam search
                    temperature=1.0,  # Não usado quando do_sample=False
                    top_p=0.95,       # Não usado quando do_sample=False
                    top_k=40,         # Não usado quando do_sample=False
                    num_beams=6,      # Aumentado para melhor busca
                    early_stopping=True,
                    no_repeat_ngram_size=2,  # Reduzido para menos repetição
                    length_penalty=1.2,      # Aumentado para resumos mais longos
                    repetition_penalty=1.3,  # Aumentado para evitar repetições
                    diversity_penalty=0.0,   # Desabilitado para evitar conflitos
                    num_return_sequences=1
                )

                result = final_summary[0]['summary_text'].strip()
            else:
                result = combined_summary

            # Validação final do comprimento
            if len(result) > max_length * 2:  # Permitir alguma flexibilidade
                result = result[:max_length * 2].rsplit(' ', 1)[0] + "..."
            elif len(result) < min_length:
                logger.warning(f"Resumo final muito curto: {len(result)} caracteres")

            # Cachear resultado
            cache_manager.cache_summary(text, "abstractive", max_length, min_length, result)

            logger.info(f"Resumo abstrativo final gerado. Length: {len(result)} caracteres, chunks processados: {len(chunks)}")
            return result

    except Exception as e:
        logger.error(f"Erro na sumarização abstrativa: {str(e)}")
        raise