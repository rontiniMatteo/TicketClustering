import warnings
import logging
import os
import pandas as pd
import numpy as np
from bertopic import BERTopic
from umap.umap_ import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import spacy
import spacy.cli

# Silence warnings and logs
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.CRITICAL)

# 1️⃣ Download stopwords italiane e carica spaCy
nltk.download('stopwords', quiet=True)
italian_stops = set(stopwords.words('italian'))
try:
    nlp = spacy.load('it_core_news_sm')
except OSError:
    spacy.cli.download('it_core_news_sm')
    nlp = spacy.load('it_core_news_sm')

# Tokenizer: lemmatizza, filtra stopwords e parole non esplicative
def custom_tokenizer(text):
    doc = nlp(text)
    return [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and len(token.lemma_)>2 and token.lemma_.lower() not in italian_stops and not token.is_stop
    ]

# 2️⃣ Carica i ticket preprocessati
def load_texts(parquet_path='tickets_features.parquet'):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"File non trovato: {parquet_path}")
    df = pd.read_parquet(parquet_path, engine='pyarrow')
    if 'testo' not in df.columns:
        raise ValueError("Colonna 'testo' non trovata nel Parquet")
    df['testo'] = df['testo'].fillna(' ').astype(str)
    return df

# 3️⃣ Configura BERTopic come nel tuo esempio
umap_reducer = UMAP(
    n_components=10,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42
)

hdbscan_model = HDBSCAN(
    min_cluster_size=2,
    cluster_selection_method='eom',
    cluster_selection_epsilon=0.05,
    metric='euclidean',
    prediction_data=True
)

vectorizer = TfidfVectorizer(
    tokenizer=custom_tokenizer,
    ngram_range=(1,2),
    max_features=2000,
    min_df=1,
    max_df=0.9
)

topic_model = BERTopic(
    umap_model=umap_reducer,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    calculate_probabilities=False,
    verbose=False
)

# 4️⃣ Esegui clustering e salva risultati
if __name__ == '__main__':
    df_meta = load_texts()
    texts = df_meta['testo'].tolist()

    topics, probs = topic_model.fit_transform(texts)
    df_meta['cluster'] = topics

    # Esporta dataframe completo
    df_meta.to_parquet('tickets_with_clusters_bertopic.parquet', index=False)
    df_meta.to_csv('tickets_with_clusters_bertopic.csv', index=False)

    # Per ogni cluster, genera file separato e mostra parole chiave
    info = topic_model.get_topic_info()
    print(info[['Topic','Name','Count']])
    for topic_id in info['Topic']:
        if topic_id == -1:
            continue
        cluster_df = df_meta[df_meta['cluster']==topic_id]
        cluster_df.to_csv(f'cluster_bertopic_{topic_id}.csv', index=False)
        keywords = topic_model.get_topic(topic_id)
        print(f"\nTopic {topic_id} keywords:", [w for w,_ in keywords])
