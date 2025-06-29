# topic_modeling.py

from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

def run_lda(count_matrix, n_topics, random_state=42, max_iter=10):
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        max_iter=max_iter,
        learning_method='online'
    )
    matrix = lda.fit_transform(count_matrix)
    labels = np.argmax(matrix, axis=1)
    return lda, matrix, labels

def evaluate_lda_perplexity(count_matrix, topic_range):
    scores = []
    for n_topics in topic_range:
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(count_matrix)
        scores.append(lda.perplexity(count_matrix))
    return scores

def extract_lda_topics(lda, feature_names, n_top_words=10):
    topics = {}
    for i, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        topics[i] = [feature_names[j] for j in top_indices]
    return topics
