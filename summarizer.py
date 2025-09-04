# summarizer.py
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

LANGUAGE = "portuguese"
SENTENCES_COUNT = 3 # Define quantas sentenças você quer no resumo

def summarize_extractive(text):
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    summary_sentences = []
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        summary_sentences.append(str(sentence))

    return " ".join(summary_sentences)

# Adicione isso ao summarizer.py
from transformers import pipeline

# Use um modelo pré-treinado em português
# Carregar o modelo pode demorar um pouco na primeira vez
summarizer_abstractive_pipeline = pipeline(
    "summarization",
    model="unicamp-dl/ptt5-base-portuguese-summ"
)

def summarize_abstractive(text):
    # O modelo tem um limite de entrada, então truncamos para segurança
    # Para textos longos, a estratégia é dividir o texto em partes (chunks)
    summary = summarizer_abstractive_pipeline(
        text,
        max_length=150,
        min_length=30,
        do_sample=False
    )
    return summary[0]['summary_text']