# clustering.py

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import numpy as np

def run_kmeans(X, n_clusters, random_state=42, n_init=10):
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(X)
    return model, labels

def evaluate_kmeans(X, labels, true_labels=None):
    sil = silhouette_score(X, labels)
    ari = adjusted_rand_score(true_labels, labels) if true_labels is not None else None
    return sil, ari

def top_terms_per_cluster(cluster_centers, feature_names, n_terms=10):
    result = {}
    for i, center in enumerate(cluster_centers):
        top_indices = center.argsort()[-n_terms:][::-1]
        result[i] = [feature_names[idx] for idx in top_indices]
    return result

def cluster_stability(X, n_clusters, n_runs=10):
    base_labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X)
    ari_scores = []
    sil_scores = []
    for i in range(n_runs):
        labels = KMeans(n_clusters=n_clusters, random_state=i).fit_predict(X)
        ari_scores.append(adjusted_rand_score(base_labels, labels))
        sil_scores.append(silhouette_score(X, labels))
    return ari_scores, sil_scores
