# visualization.py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import umap
from wordcloud import WordCloud

def plot_elbow_curve(X, max_k=15):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    inertias = []
    silhouettes = []
    for k in range(2, max_k+1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(X, model.labels_))
    plt.figure(figsize=(10,4))
    plt.plot(range(2, max_k+1), inertias, marker='o', label='Inertia')
    plt.plot(range(2, max_k+1), silhouettes, marker='x', label='Silhouette')
    plt.legend()
    plt.show()

def run_tsne(X, n_components=2, random_state=42):
    tsne = TSNE(n_components=n_components, random_state=random_state)
    return tsne.fit_transform(X)

def run_umap(X, n_components=2, random_state=42):
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    return reducer.fit_transform(X)

def plot_wordcloud(word_freq, title=None):
    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()
