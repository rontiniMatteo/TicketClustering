import pandas as pd
import numpy as np
import glob
import os
import re
import csv
import sys
import html
from tqdm import tqdm

# NLTK e spaCy rimangono per elaborazioni testuali
import nltk
from nltk.corpus import stopwords
import spacy
import spacy.cli

# Per TF-IDF
default_tfidf_params = { 'token_pattern': r"[^\s]+", 'min_df': 1 }
from sklearn.feature_extraction.text import TfidfVectorizer

# Import SBERT (modello multilingue che supporta l'italiano)
from sentence_transformers import SentenceTransformer

# torch rimane se serve per altre operazioni
import torch

# ---------------- Parametri ----------------
CSV_DIR         = r"D:\Magistrale\AI\output_chunks"   # percorso file CSV/TSV
# Usiamo un modello SBERT community multilingue che supporta italiano
SBERT_MODEL     = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE      = 32
SAMPLE_SIZE     = 5000      # None per processare tutti i ticket
TEMPORAL_WEIGHT = 0.0       # ignoriamo totalmente la componente temporale
TITLE_WEIGHT    = 1.2       # peso relativo da dare all'embedding SBERT dei titoli
TFIDF_WEIGHT    = 1.0       # peso da dare al componente TF-IDF

# Aumenta il limite per campi molto lunghi
try:
    csv.field_size_limit(2**31 - 1)
except OverflowError:
    csv.field_size_limit(sys.maxsize)

# 0️⃣ Setup stopwords italiane e spaCy
nltk.download('stopwords', quiet=True)
italian_stops = set(stopwords.words('italian'))
try:
    nlp = spacy.load('it_core_news_sm')
except OSError:
    spacy.cli.download('it_core_news_sm')
    nlp = spacy.load('it_core_news_sm')

# ---------------- Funzioni utili ----------------

def detect_delimiter(path):
    """Rileva automaticamente il separatore presente in un file CSV/TSV."""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        sample = f.read(2048)
    try:
        return csv.Sniffer().sniff(sample).delimiter
    except csv.Error:
        return ','

def safe_read_csv(path, **kwargs):
    """Legge un CSV provando sia 'utf-8' sia 'latin-1' in caso di errore."""
    try:
        return pd.read_csv(path, encoding='utf-8', **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin-1', **kwargs)

def clean_html(raw: str) -> str:
    """
    Rimuove i tag HTML, effettua html.unescape e converte in minuscolo
    per preparare il testo.
    """
    if pd.isna(raw):
        return ""
    text = html.unescape(raw)
    text = re.sub(r"<[^>]+>", " ", text)
    return ' '.join(text.split()).lower()

def custom_tokenizer(text: str) -> list:
    """
    Lemmatizza, filtra stop-words, token non alfabetici, lemmi < 3 char
    """
    doc = nlp(text)
    tokens = []
    for token in doc:
        lemma = token.lemma_.lower()
        if token.is_alpha and len(lemma) > 2 and lemma not in italian_stops and not token.is_stop:
            tokens.append(lemma)
    return tokens

def generate_ngrams(tokens: list) -> list:
    """
    Dato un elenco di tokens (lemmi), crea unigram, bigram e trigram.
    """
    ngrams = tokens.copy()
    L = len(tokens)
    if L >= 2:
        for i in range(L - 1):
            ngrams.append(tokens[i] + "_" + tokens[i+1])
    if L >= 3:
        for i in range(L - 2):
            ngrams.append(tokens[i] + "_" + tokens[i+1] + "_" + tokens[i+2])
    return ngrams

def char_ngrams(text: str, min_n=4, max_n=5) -> list:
    """
    Estrae n-gram di caratteri di lunghezza compresa tra min_n e max_n.
    """
    cleaned = text.replace(" ", "_")
    ngrams = []
    L = len(cleaned)
    for n in range(min_n, max_n + 1):
        for i in range(L - n + 1):
            ngrams.append(cleaned[i:i + n])
    return ngrams

def cyclic_feats(vals, max_val):
    """
    Calcola feature cicliche (sin, cos).
    """
    theta = 2 * np.pi * vals / max_val
    return np.sin(theta), np.cos(theta)

# ---------------- Lettura e preparazione dei dati ----------------

ticket_dfs = []
file_paths = [p for p in glob.glob(os.path.join(CSV_DIR, '*')) if os.path.isfile(p)]
print(f"DEBUG: trovati {len(file_paths)} file in '{CSV_DIR}'")

for path in tqdm(file_paths, desc="Lettura file", unit="file"):
    sep = detect_delimiter(path)
    df_head = safe_read_csv(path, sep=sep, engine='python', nrows=5)
    cols = df_head.columns.tolist()

    # Trova colonne data e testo
    date_cols = [c for c in cols if 'create' in c.lower()]
    use_cols  = [c for c in cols if any(x in c.lower() for x in ['segna','title','descr','create'])]

    df_part = safe_read_csv(path, sep=sep, engine='python', usecols=use_cols, parse_dates=date_cols)

    # Rinomino colonne
    col_map = {
        col: ('ticket_id'  if 'segna' in col.lower() else
              'title'      if 'title' in col.lower() else
              'descr'      if 'descr' in col.lower() else
              'created_at' if 'create' in col.lower() else col)
        for col in df_part.columns
    }
    df_part.rename(columns=col_map, inplace=True)

    # Assicuro che title e descr esistano
    df_part['title'] = df_part.get('title', '').fillna('').astype(str)
    df_part['descr'] = df_part.get('descr', '').fillna('').astype(str)

    # Pulizia HTML
    df_part['title_clean'] = df_part['title'].apply(clean_html)
    df_part['descr_clean'] = df_part['descr'].apply(clean_html)

    # Lemmatizzazione e filtri stop-words
    df_part['title_tokens'] = df_part['title_clean'].apply(custom_tokenizer)
    df_part['descr_tokens'] = df_part['descr_clean'].apply(custom_tokenizer)

    # Generazione lemma-ngrams e char-ngrams
    df_part['title_lemgrams'] = df_part['title_tokens'].apply(lambda toks: generate_ngrams(toks))
    df_part['descr_lemgrams'] = df_part['descr_tokens'].apply(lambda toks: generate_ngrams(toks))
    df_part['title_chargrams'] = df_part['title_clean'].apply(lambda s: char_ngrams(s))
    df_part['descr_chargrams'] = df_part['descr_clean'].apply(lambda s: char_ngrams(s))

    # Unisco entrambi i tipi di n-gram in un'unica stringa
    df_part['title_ngrams_combined'] = df_part.apply(lambda row: ' '.join(row['title_lemgrams'] + row['title_chargrams']), axis=1)
    df_part['descr_ngrams_combined'] = df_part.apply(lambda row: ' '.join(row['descr_lemgrams'] + row['descr_chargrams']), axis=1)

    ticket_dfs.append(df_part[['ticket_id','title_ngrams_combined','descr_ngrams_combined','created_at']])

# Concatena e rimuovi righe senza testo o data
_df = pd.concat(ticket_dfs, ignore_index=True)
_df.dropna(subset=['title_ngrams_combined','descr_ngrams_combined','created_at'], inplace=True)
print(f"Totale ticket dopo pulizia e n-gram combinati: {len(_df)}")

if SAMPLE_SIZE and len(_df) > SAMPLE_SIZE:
    _df = _df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print(f"Campionati {_df.shape[0]} ticket per test rapido")

# Feature temporali
_df['hour']    = _df['created_at'].dt.hour
_df['weekday'] = _df['created_at'].dt.weekday
_df['hour_sin'], _df['hour_cos']       = cyclic_feats(_df['hour'], 24)
_df['weekday_sin'], _df['weekday_cos'] = cyclic_feats(_df['weekday'], 7)

temporal = _df[['hour_sin','hour_cos','weekday_sin','weekday_cos']].to_numpy() * TEMPORAL_WEIGHT

# 4️⃣ Calcola TF-IDF sui n-gram combinati
tfidf_vec = TfidfVectorizer(**default_tfidf_params)
combined_texts = (_df['title_ngrams_combined'] + " " + _df['descr_ngrams_combined']).tolist()
print("Calcolo matrice TF-IDF su n-gram combinati...")
X_tfidf = tfidf_vec.fit_transform(combined_texts)
X_tfidf = X_tfidf.toarray()
# Normalizzo ogni vettore TF-IDF a L2 per metrica Cosine
X_tfidf = X_tfidf / np.linalg.norm(X_tfidf, axis=1, keepdims=True)

# 5️⃣ Carica modello SBERT
print(f"Caricamento modello SBERT {SBERT_MODEL}...")
model = SentenceTransformer(SBERT_MODEL)

# 6️⃣ Calcolo embedding SBERT su n-gram combinati (titolo + descrizione)
print("Calcolo embedding SBERT per tutti i ticket su n-gram combinati...")
texts_all = (_df['title_ngrams_combined'] + " " + _df['descr_ngrams_combined']).tolist()
X_sbert_all = model.encode(texts_all, batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True)
print(f"Shape embedding SBERT complessivo: {X_sbert_all.shape}")

# 7️⃣ Combina vettori SBERT e TF-IDF (già normalizzati) per tutti i ticket
X_combined_all = np.hstack([TITLE_WEIGHT * X_sbert_all, TFIDF_WEIGHT * X_tfidf])
print(f"Shape matrice feature X_combined_all: {X_combined_all.shape}")

# 8️⃣ Matrice finale X_all (text+tfidf + temporal)
X_all = np.hstack([X_combined_all, np.tile(np.zeros((1,4)), (X_combined_all.shape[0],1))])
print(f"Shape matrice feature X_all (finale): {X_all.shape}")

# 9️⃣ Salvo su disco
np.savez_compressed('features.npz', X=X_all)
print("SBERT + TF-IDF di tutti i ticket salvato in: features.npz")

# 10️⃣ Salvo il DataFrame normalizzato completo
_df.to_parquet('tickets_features_ngrams_tfidf_sbert.parquet', index=False)
print("DataFrame normalizzato con n-gram combinati + TF-IDF + SBERT salvato in: tickets_features_ngrams_tfidf_sbert.parquet")
