import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

def calculate_metrics(true_labels, predicted_labels, X):
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    ari = adjusted_rand_score(true_labels, predicted_labels)
    sil = silhouette_score(X, predicted_labels)
    return {"ARI": ari, "Silhouette": sil}

def plot_confusion(true_labels, predicted_labels, target_names, title):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=[f"C{i}" for i in np.unique(predicted_labels)],
                yticklabels=target_names, cmap='Blues')
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Cluster")
    plt.show()

def topic_diversity(topics, topk=10):
    words = set()
    total = 0
    for topic_words in topics.values():
        words.update(topic_words[:topk])
        total += topk
    return len(words) / total if total > 0 else 0

def compare_models(metrics_dict):
    df = pd.DataFrame(metrics_dict).T
    print(df)
    df.plot(kind="bar")
    plt.show()
