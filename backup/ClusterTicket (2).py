import warnings
import logging
import os

# Silence all Python warnings and deprecations
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="umap.*")
warnings.filterwarnings("ignore", module="sklearn.*")
warnings.simplefilter("ignore")
warnings.warn = lambda *args, **kwargs: None
warnings.showwarning = lambda *args, **kwargs: None

# Disable all logging messages
logging.getLogger().setLevel(logging.CRITICAL)
logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd
from bertopic import BERTopic
from umap.umap_ import UMAP
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed
import nltk
from nltk.corpus import stopwords
import spacy
import spacy.cli

# 0️⃣ Setup stopwords italiane e spaCy
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
    tokens = []
    for token in doc:
        lemma = token.lemma_.lower()
        if token.is_alpha and len(lemma) > 2 and lemma not in italian_stops and not token.is_stop:
            tokens.append(lemma)
    return tokens

# 1️⃣ Griglie di iperparametri per UMAP e clustering
grids = {
    'umap': [
        {'n_neighbors': 15, 'min_dist': 0.1, 'n_components': 10},
        {'n_neighbors': 5,  'min_dist': 0.0, 'n_components': 5},
        {'n_neighbors': 30, 'min_dist': 0.5, 'n_components': 2},
    ],
    'dbscan': [
        {'eps': eps, 'min_samples': ms, 'metric': met}
        for eps in [0.1, 0.5, 1.0]
        for ms in [2,5,10]
        for met in ['euclidean','cosine']
    ],
    'hdbscan': [
        {'min_cluster_size': mcs, 'cluster_selection_method': csm,
         'cluster_selection_epsilon': cse, 'metric': met}
        for mcs in [2,5,10]
        for csm in ['eom','leaf']
        for cse in [0.01,0.05,0.1]
        for met in ['euclidean','cosine']
    ],
    'optics': [
        {'min_samples': ms, 'cluster_method': 'xi', 'xi': xi}
        for ms in [2,5,10]
        for xi in [0.01,0.05,0.1]
    ],
    'agglomerative': [
        {'n_clusters': nc, 'linkage': lg}
        for nc in [10,20,50]
        for lg in ['ward','average']
    ]
}

# 2️⃣ Estrarre i top-terms medi dai cluster
def extract_top_terms(labels, tfidf_matrix, feature_names, top_n=5):
    clusters = {}
    for label in np.unique(labels):
        if label < 0:
            continue
        idx = np.where(labels == label)[0]
        mean_vec = tfidf_matrix[idx].mean(axis=0).A1
        top_idx = np.argsort(mean_vec)[-top_n:][::-1]
        clusters[int(label)] = [feature_names[i] for i in top_idx]
    return clusters

# 3️⃣ Pipeline sperimentale con handling dinamico e salvataggio
def run_experiments(texts, tfidf_matrix, feature_names, grids, runs=10):
    n_samples = tfidf_matrix.shape[0]

    def single_run(run_id, umap_cfg, method, params):
        cfg = umap_cfg.copy()
        max_val = max(1, n_samples - 1)
        cfg['n_components'] = min(cfg.get('n_components', max_val), max_val)
        cfg['n_neighbors'] = min(cfg.get('n_neighbors', max_val), max_val)

        um = UMAP(init='random', random_state=run_id, **cfg)
        emb = um.fit_transform(tfidf_matrix)

        if method == 'dbscan':
            labels = DBSCAN(**params).fit_predict(emb)
        elif method == 'hdbscan':
            h_params = params.copy()
            metric = h_params.pop('metric')
            if metric == 'cosine':
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                emb_norm = emb / np.where(norms == 0, 1, norms)
                labels = hdbscan.HDBSCAN(**h_params, metric='euclidean').fit_predict(emb_norm)
            else:
                labels = hdbscan.HDBSCAN(**h_params, metric=metric).fit_predict(emb)
        elif method == 'optics':
            o_params = params.copy()
            o_params['min_samples'] = min(o_params['min_samples'], n_samples)
            labels = OPTICS(**o_params).fit_predict(emb)
        elif method == 'agglomerative':
            a_params = params.copy()
            a_params['n_clusters'] = min(a_params['n_clusters'], n_samples)
            key = 'metric' if 'metric' in AgglomerativeClustering.__init__.__code__.co_varnames else 'affinity'
            a_params[key] = 'euclidean' if a_params['linkage']=='ward' else 'cosine'
            labels = AgglomerativeClustering(**a_params).fit_predict(emb)
        else:
            raise ValueError(f"Metodo sconosciuto: {method}")

        valid_clusters = [l for l in np.unique(labels) if l >= 0]
        n_clusters = len(valid_clusters)
        noise_pts = int(np.sum(labels == -1))
        sil = silhouette_score(emb, labels) if 2 <= n_clusters <= (n_samples - 1) else np.nan
        top_terms = extract_top_terms(labels, tfidf_matrix, feature_names)

        return {
            'method': method,
            'run': run_id,
            'umap_cfg': cfg,
            'params': params,
            'n_clusters': n_clusters,
            'noise_points': noise_pts,
            'silhouette': sil,
            'top_terms': top_terms,
            'labels': labels.tolist()
        }

    jobs = [
        delayed(single_run)(run_id, u, m, p)
        for run_id in range(runs)
        for u in grids['umap']
        for m in ['dbscan', 'hdbscan', 'optics', 'agglomerative']
        for p in grids[m]
    ]
    results = Parallel(n_jobs=-1, verbose=5)(jobs)
    df = pd.DataFrame(results)

    # Salva i risultati su file CSV
    out_file = os.path.join(os.getcwd(), 'clustering_results.csv')
    df.to_csv(out_file, index=False)

    # Flatten per cluster in CSV
    top_terms_flat = []
    for i, res in enumerate(results):
        for cl, terms in res['top_terms'].items():
            top_terms_flat.append({
                'experiment': i,
                'method': res['method'],
                'run': res['run'],
                'cluster': cl,
                'terms': ', '.join(terms)
            })
    top_df = pd.DataFrame(top_terms_flat)
    top_csv = os.path.join(os.getcwd(), 'clustering_top_terms.csv')
    top_df.to_csv(top_csv, index=False)

    print(f"Risultati salvati in: {out_file}")
    print(f"Top-terms CSV salvato in: {top_csv}")

    return df

# 4️⃣ Test di integrazione e validazione
if __name__ == '__main__':
    sample_texts = ['gatto cane topo', 'cane gatto', 'auto moto bici']
    vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, min_df=1)
    tfidf_test = vectorizer.fit_transform(sample_texts)
    features = vectorizer.get_feature_names_out()
    test_grids = grids.copy()
    test_grids['umap'] = [{'n_neighbors':5,'min_dist':0.1,'n_components':2}]
    df_res = run_experiments(sample_texts, tfidf_test, features, test_grids, runs=2)
    print(df_res[['method','n_clusters']])
    assert not df_res.empty, "Result DataFrame dovrebbe contenere risultati"
    print("Test completati con successo.")

# 5️⃣ Macro-clustering (opzionale)
# Da implementare dopo selezione dei migliori esperimenti
