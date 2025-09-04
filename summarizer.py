# summarizer.py
import logging
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from huggingface_hub import login

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

with open("API_HuggingFace", "r") as f:
    token = f.read().strip()
login(token)

LANGUAGE = "portuguese"
DEFAULT_SENTENCES_COUNT = 3  # Número padrão de sentenças para resumo extrativo
DEFAULT_MAX_LENGTH = 150  # Comprimento máximo padrão do resumo
DEFAULT_MIN_LENGTH = 30   # Comprimento mínimo padrão do resumo

def summarize_extractive(text, max_length=DEFAULT_MAX_LENGTH, min_length=DEFAULT_MIN_LENGTH):
    """
    Gera um resumo extrativo do texto usando o algoritmo LSA.

    Args:
        text (str): Texto a ser resumido
        max_length (int): Comprimento máximo aproximado do resumo em caracteres
        min_length (int): Comprimento mínimo aproximado do resumo em caracteres

    Returns:
        str: Resumo extrativo do texto
    """
    logger.info(f"Iniciando sumarização extrativa. Texto length: {len(text)}, max_length: {max_length}, min_length: {min_length}")

    try:
        parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
        stemmer = Stemmer(LANGUAGE)
        summarizer = Summarizer(stemmer)
        summarizer.stop_words = get_stop_words(LANGUAGE)

        # Estimar número de sentenças baseado no comprimento desejado
        # Aproximadamente 100-150 caracteres por sentença em português
        avg_chars_per_sentence = 120
        estimated_sentences = max(1, min(10, max_length // avg_chars_per_sentence))
        sentences_count = max(DEFAULT_SENTENCES_COUNT, estimated_sentences)

        logger.info(f"Usando {sentences_count} sentenças para o resumo")

        summary_sentences = []
        for sentence in summarizer(parser.document, sentences_count):
            summary_sentences.append(str(sentence))

        result = " ".join(summary_sentences)

        # Ajustar comprimento se necessário
        if len(result) > max_length:
            result = result[:max_length].rsplit(' ', 1)[0] + "..."
        elif len(result) < min_length and len(summary_sentences) < sentences_count * 2:
            # Tentar adicionar mais sentenças se estiver muito curto
            additional_sentences = []
            for sentence in summarizer(parser.document, sentences_count + 2):
                if str(sentence) not in summary_sentences:
                    additional_sentences.append(str(sentence))
                    if len(" ".join(summary_sentences + additional_sentences)) >= min_length:
                        break
            if additional_sentences:
                result = " ".join(summary_sentences + additional_sentences[:2])

        logger.info(f"Resumo extrativo gerado com sucesso. Length: {len(result)}")
        return result

    except Exception as e:
        logger.error(f"Erro na sumarização extrativa: {str(e)}")
        raise

# Adicione isso ao summarizer.py
from transformers import pipeline

# Use um modelo pré-treinado em português
# Carregar o modelo pode demorar um pouco na primeira vez
summarizer_abstractive_pipeline = pipeline(
    "summarization",
    model="csebuetnlp/mT5_multilingual_XLSum"
)
tokenizer = summarizer_abstractive_pipeline.tokenizer

def summarize_abstractive(text, max_length=DEFAULT_MAX_LENGTH, min_length=DEFAULT_MIN_LENGTH):
    """
    Gera um resumo abstrativo do texto usando modelo de linguagem T5.

    Args:
        text (str): Texto a ser resumido
        max_length (int): Comprimento máximo do resumo em tokens
        min_length (int): Comprimento mínimo do resumo em tokens

    Returns:
        str: Resumo abstrativo do texto
    """
    logger.info(f"Iniciando sumarização abstrativa. Texto length: {len(text)}, max_length: {max_length}, min_length: {min_length}")

    try:
        # Validar parâmetros
        if max_length <= min_length:
            raise ValueError("max_length deve ser maior que min_length")
        if max_length > 500:
            logger.warning(f"max_length ({max_length}) muito alto, pode afetar performance")
        if min_length < 10:
            logger.warning(f"min_length ({min_length}) muito baixo, resumo pode ser muito curto")

        max_input_length = 512  # Limite do modelo mT5
        tokens = tokenizer.encode(text, add_special_tokens=True)

        if len(tokens) <= max_input_length:
            # Texto curto - processar diretamente
            logger.info("Processando texto curto diretamente")

            # Adicionar prefixo para T5 (importante para task de sumarização)
            input_text = f"summarize: {text}"

            summary = summarizer_abstractive_pipeline(
                input_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,  # Habilitar sampling para mais diversidade
                temperature=0.3,  # Baixa temperatura para consistência
                top_p=0.9,  # Nucleus sampling
                top_k=50,  # Top-k sampling
                num_beams=4,  # Beam search para qualidade
                early_stopping=True,
                no_repeat_ngram_size=3,  # Evitar repetições
                length_penalty=1.0,  # Penalidade de comprimento
                repetition_penalty=1.2  # Penalizar repetições
            )

            result = summary[0]['summary_text'].strip()
            logger.info(f"Resumo abstrativo gerado. Length: {len(result)} caracteres")
            return result

        else:
            # Texto longo - dividir em chunks
            logger.info(f"Texto longo detectado ({len(tokens)} tokens). Dividindo em chunks.")

            # Estratégia melhorada de chunking: dividir por sentenças quando possível
            chunks = []
            current_chunk_tokens = []
            current_length = 0

            # Tentar dividir por sentenças primeiro
            sentences = text.split('. ')
            current_chunk_text = ""

            for sentence in sentences:
                sentence_tokens = tokenizer.encode(sentence + ". ", add_special_tokens=False)
                if current_length + len(sentence_tokens) <= max_input_length - 50:  # Margem de segurança
                    current_chunk_text += sentence + ". "
                    current_length += len(sentence_tokens)
                else:
                    if current_chunk_text:
                        chunks.append(current_chunk_text.strip())
                    current_chunk_text = sentence + ". "
                    current_length = len(sentence_tokens)

            if current_chunk_text:
                chunks.append(current_chunk_text.strip())

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

            # Summarizar cada chunk
            summaries = []
            chunk_max_length = max(50, max_length // len(chunks))  # Distribuir comprimento entre chunks

            for i, chunk in enumerate(chunks):
                logger.info(f"Processando chunk {i+1}/{len(chunks)}")

                input_text = f"summarize: {chunk}"

                summary = summarizer_abstractive_pipeline(
                    input_text,
                    max_length=chunk_max_length,
                    min_length=max(10, min_length // len(chunks)),
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    top_k=50,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                    repetition_penalty=1.2
                )

                summaries.append(summary[0]['summary_text'].strip())

            # Combinar resumos dos chunks
            combined_summary = " ".join(summaries)

            # Se o resumo combinado for muito longo, fazer um resumo final
            combined_tokens = tokenizer.encode(combined_summary, add_special_tokens=False)
            if len(combined_tokens) > max_length:
                logger.info("Fazendo resumo final do resumo combinado")

                final_input = f"summarize: {combined_summary}"
                final_summary = summarizer_abstractive_pipeline(
                    final_input,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    top_k=50,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                    repetition_penalty=1.2
                )

                result = final_summary[0]['summary_text'].strip()
            else:
                result = combined_summary

            # Validação final do comprimento
            if len(result) > max_length * 2:  # Permitir alguma flexibilidade
                result = result[:max_length * 2].rsplit(' ', 1)[0] + "..."
            elif len(result) < min_length:
                logger.warning(f"Resumo final muito curto: {len(result)} caracteres")

            logger.info(f"Resumo abstrativo final gerado. Length: {len(result)} caracteres, chunks processados: {len(chunks)}")
            return result

    except Exception as e:
        logger.error(f"Erro na sumarização abstrativa: {str(e)}")
        raise