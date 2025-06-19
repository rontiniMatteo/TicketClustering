import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

class TicketDistanceExplorer:
    """
    Classe per l'analisi esplorativa delle distanze fra embedding dei ticket.
    Permette di calcolare k-distance plots, istogrammi di distanze e proiezioni 2D.
    """
    def __init__(self, feature_path='features.npz', meta_path='tickets_features.parquet'):
        # Carica matrice delle feature
        data = np.load(feature_path)
        self.X = data['X']  # shape: (n_samples, dim)
        # Carica metadati (opzionale)
        try:
            self.meta = pd.read_parquet(meta_path, engine='pyarrow')
        except Exception:
            self.meta = None

    def k_distance_plot(self, k=5, show=True):
        """
        Calcola e mostra il k-distance plot per determinare un valore eps.
        """
        nbrs = NearestNeighbors(n_neighbors=k).fit(self.X)
        distances, _ = nbrs.kneighbors(self.X)
        # Prendi la distanza al k-esimo vicino per ciascun punto
        k_dist = np.sort(distances[:, k-1])
        plt.figure()
        plt.plot(k_dist)
        plt.title(f'k-distance plot (k={k})')
        plt.xlabel('Punti ordinati')
        plt.ylabel(f'Distanza al {k}-esimo vicino')
        if show:
            plt.show()
        return k_dist

    def distance_histogram(self, metric='euclidean', bins=50, sample_size=10000, show=True):
        """
        Calcola e mostra un istogramma delle distanze (a coppie) su un campione.
        """
        n = self.X.shape[0]
        # Campiona un sottoinsieme di indici
        idx = np.random.choice(n, min(n, sample_size), replace=False)
        Xs = self.X[idx]
        # Calcola distanze fra tutte le coppie
        dists = pairwise_distances(Xs, metric=metric)
        # Prendi la metà superiore (escludi diag e duplicati)
        iu = np.triu_indices_from(dists, k=1)
        vals = dists[iu]
        plt.figure()
        plt.hist(vals, bins=bins)
        plt.title(f'Istogramma distanze ({metric})')
        plt.xlabel('Distanza')
        plt.ylabel('Frequenza')
        if show:
            plt.show()
        return vals

    def pca_scatter(self, metadata=False, show=True):
        """
        Proietta gli embedding in 2D con PCA e mostra uno scatter.
        Se metadata=True e self.meta esiste, colora per qualche colonna.
        """
        pca = PCA(n_components=2)
        X2 = pca.fit_transform(self.X)
        plt.figure()
        if metadata and self.meta is not None:
            # esempio: colora per cluster se presente
            if 'cluster' in self.meta.columns:
                labels = self.meta['cluster'].values
                scatter = plt.scatter(X2[:,0], X2[:,1], c=labels, cmap='tab10', s=10)
                plt.legend(*scatter.legend_elements(), title='Cluster')
            else:
                plt.scatter(X2[:,0], X2[:,1], s=10)
        else:
            plt.scatter(X2[:,0], X2[:,1], s=10)
        plt.title('PCA 2D degli embedding')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        if show:
            plt.show()
        return X2

if __name__ == '__main__':
    explorer = TicketDistanceExplorer()
    # Esempi di utilizzo:
    explorer.k_distance_plot(k=5)
    explorer.distance_histogram(metric='cosine', bins=50)
    explorer.pca_scatter(metadata=True)
