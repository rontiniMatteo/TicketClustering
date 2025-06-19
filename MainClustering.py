import os
import warnings

# Sopprimi i FutureWarning relativi a 'force_all_finite'
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*force_all_finite.*"
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import hdbscan
import joblib
from Load_Tickets import TicketEmbedder

class MainClustering:
    """
    Classe per il training del modello di clustering, calcolo metriche e salvataggio risultati.
    Il modello viene salvato in 'output_cluster/hdbscan_model.joblib'.
    """
    def __init__(self,
                 features_npz_path: str,
                 tickets_csv_name: str,
                 output_dir: str = "output_cluster",
                 micro_min_cluster_size: int = 5):
        self.features_npz_path = features_npz_path
        self.tickets_csv_name = tickets_csv_name
        self.output_dir = output_dir
        self.micro_min_cluster_size = micro_min_cluster_size
        os.makedirs(self.output_dir, exist_ok=True)
        self.df_tickets = None
        self.X = None
        self.df_macro_results = None
        self.df_micro_clusters = None
        self.clusterer = None
        self.embedder = TicketEmbedder()

    def load_data(self):
        """Carica metadata e embeddings, allinea righe."""
        csv_path = os.path.join(os.getcwd(), "output_chunks", self.tickets_csv_name)
        self.df_tickets = pd.read_csv(csv_path, sep=';', encoding='latin-1')
        if 'SGSEGNALID' not in self.df_tickets.columns:
            raise ValueError("Colonna 'SGSEGNALID' mancante.")
        print(f"[Loader] {len(self.df_tickets)} ticket da {csv_path}")
        data = np.load(self.features_npz_path)
        self.X = data['X']
        # rimuovi NaN
        self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"[Loader] Embedding shape: {self.X.shape}")
        # allineamento
        n_embed, n_tickets = self.X.shape[0], len(self.df_tickets)
        if n_embed != n_tickets:
            m = min(n_embed, n_tickets)
            print(f"[Loader] Allineo a {m} righe")
            self.X = self.X[:m]
            self.df_tickets = self.df_tickets.iloc[:m].reset_index(drop=True)

    def run_macro_benchmark(self):
        """Benchmark macro: calcola silhouette e noise per vari metodi."""
        if self.X is None:
            raise RuntimeError("Chiamare load_data() prima.")
        results = []
        # Definizione metodi
        methods = [
            ('KMeans', KMeans(n_clusters=3, random_state=42)),
            ('Agglomerative', AgglomerativeClustering(n_clusters=3, linkage='ward')),
            ('GMM', GaussianMixture(n_components=3, covariance_type='diag', random_state=42)),
            ('DBSCAN', DBSCAN(metric='euclidean', eps=0.5, min_samples=5)),
            ('HDBSCAN', hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=self.micro_min_cluster_size,
                                        cluster_selection_method='eom', prediction_data=False))
        ]
        for name, model in methods:
            labels = model.fit_predict(self.X)
            valid = set(labels) - {-1}
            noise = (labels == -1).sum() / len(labels) * 100.0
            sil = silhouette_score(self.X, labels) if len(valid) > 1 else np.nan
            params = getattr(model, 'n_clusters', getattr(model, 'eps',
                      f'min_cluster_size={self.micro_min_cluster_size}'))
            results.append({'Method': name,
                            'Params': params,
                            'Num_Clusters': len(valid),
                            'Noise_%': noise,
                            'Silhouette_euclid': sil})
        # creazione DataFrame
        self.df_macro_results = pd.DataFrame(results)
        self.df_macro_results.sort_values('Silhouette_euclid', ascending=False, inplace=True)
        # salva CSV
        out_csv = os.path.join(self.output_dir, "confronto_metodi_macro.csv")
        self.df_macro_results.to_csv(out_csv, sep=';', index=False)
        print(f"[Macro] CSV salvato in: {out_csv}")
        # plot
        mask = self.df_macro_results['Silhouette_euclid'].notna()
        plt.figure(figsize=(10, 4))
        x = self.df_macro_results.loc[mask, 'Method'] + ' - ' + self.df_macro_results.loc[mask, 'Params'].astype(str)
        y = self.df_macro_results.loc[mask, 'Silhouette_euclid']
        plt.bar(x, y, color='steelblue')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Silhouette (euclid)')
        plt.title('Confronto metodi clustering macro')
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "macro_cluster_comparison.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"[Macro] Plot salvato in: {plot_path}")

    def run_micro_clustering(self):
        """Micro-clustering HDBSCAN e salvataggio CSV."""
        if self.X is None:
            raise RuntimeError("Chiamare load_data() prima.")
        self.clusterer = hdbscan.HDBSCAN(metric='euclidean',
                                         min_cluster_size=self.micro_min_cluster_size,
                                         cluster_selection_method='eom',
                                         prediction_data=True).fit(self.X)
        self.df_tickets['label_micro'] = self.clusterer.labels_
        # salva micro clusters
        info = []
        for cid in sorted(set(self.clusterer.labels_)):
            if cid == -1: continue
            members = self.df_tickets.loc[self.df_tickets['label_micro']==cid, 'SGSEGNALID'].tolist()
            info.append({'cluster_id': cid, 'size': len(members), 'members': members})
        self.df_micro_clusters = pd.DataFrame(info)
        out_csv = os.path.join(self.output_dir, f"clusters_micro_hdbscan_{self.micro_min_cluster_size}.csv")
        self.df_micro_clusters.to_csv(out_csv, sep=';', index=False)
        print(f"[Micro] CSV salvato in: {out_csv}")

    def save_results(self):
        """Salva tickets con label_micro su CSV."""
        if self.df_tickets is not None and 'label_micro' in self.df_tickets:
            out_csv = os.path.join(self.output_dir, "tickets_with_micro_labels.csv")
            self.df_tickets.to_csv(out_csv, sep=';', index=False)
            print(f"[Save] Tickets salvati in: {out_csv}")

    def save_model(self, model_name: str = 'hdbscan_model.joblib'):
        """Serializza e salva il clusterer in output_dir."""
        if self.clusterer is None:
            raise RuntimeError("Eseguire run_micro_clustering() prima.")
        path = os.path.join(self.output_dir, model_name)
        joblib.dump(self.clusterer, path)
        print(f"[Model] Modello salvato in: {path}")

if __name__ == '__main__':
    mc = MainClustering(
        features_npz_path='features.npz',
        tickets_csv_name='fattoAmano.csv',
        output_dir='output_cluster',
        micro_min_cluster_size=5
    )
    mc.load_data()
    mc.run_macro_benchmark()
    mc.run_micro_clustering()
    mc.save_results()
    mc.save_model()
