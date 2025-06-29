import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src import (
    data_loader,
    preprocessing,
    vectorization,
    clustering,
    topic_modeling,
    visualization,
    evaluation,
)

# For reproducibility
np.random.seed(42)

def main():
    print("\n" + "="*50)
    print("Loading Data")
    print("="*50)

    # Load data
    df_train, df_test, categories = data_loader.load_20newsgroups(categories=None)

    # EDA plots
    visualization.plot_eda(df_train)

    print("\n" + "="*50)
    print("Preprocessing Texts")
    print("="*50)

    df_train["processed_text"] = df_train["text"].apply(preprocessing.preprocess_text)

    # Remove empty documents
    df_train = df_train[df_train["processed_text"].str.strip().str.len() > 0]

    print("\nSample processed text:")
    print(df_train["processed_text"].iloc[0][:500])

    print("\n" + "="*50)
    print("Vectorizing Text")
    print("="*50)

    tfidf_vectorizer, tfidf_matrix, feature_names = vectorization.vectorize_tfidf(df_train["processed_text"])
    count_vectorizer, count_matrix, count_feature_names = vectorization.vectorize_count(df_train["processed_text"])

    # Dense TF-IDF matrix
    tfidf_dense = tfidf_matrix.toarray()

    print("\n" + "="*50)
    print("Clustering with K-means")
    print("="*50)

    optimal_k, inertias, sil_scores = clustering.find_optimal_k(tfidf_matrix, max_k=20)
    n_clusters = len(categories)

    kmeans, kmeans_labels, top_terms_per_cluster = clustering.perform_kmeans(
        tfidf_matrix,
        n_clusters,
        feature_names,
        n_terms=15
    )

    df_train["kmeans_cluster"] = kmeans_labels

    print("\nTop terms per K-means cluster:")
    for cluster_id, terms in top_terms_per_cluster.items():
        print(f"Cluster {cluster_id}: {', '.join(terms)}")

    print("\n" + "="*50)
    print("Topic Modeling with LDA")
    print("="*50)

    optimal_topics, perplexities = topic_modeling.find_optimal_topics(count_matrix, range(5, 25, 2))

    lda, lda_matrix, lda_labels, lda_topics = topic_modeling.perform_lda(
        count_matrix,
        count_feature_names,
        n_topics=n_clusters
    )

    df_train["lda_topic"] = lda_labels

    print("\nTop words per LDA topic:")
    for topic_id, words in lda_topics.items():
        print(f"Topic {topic_id}: {', '.join(words)}")

    # Generate word clouds
    visualization.plot_wordclouds(lda, count_feature_names, n_topics)

    print("\n" + "="*50)
    print("Dimensionality Reduction for Visualization")
    print("="*50)

    # PCA
    pca = PCA(n_components=50, random_state=42)
    tfidf_pca = pca.fit_transform(tfidf_dense)

    # t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tfidf_tsne = tsne.fit_transform(tfidf_pca)

    # UMAP
    print("Running UMAP...")
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    tfidf_umap = umap_reducer.fit_transform(tfidf_dense)

    print("\n" + "="*50)
    print("Generating Visualizations")
    print("="*50)

    visualization.plot_tsne_umap(
        tfidf_tsne,
        tfidf_umap,
        df_train["target"],
        df_train["kmeans_cluster"],
        df_train["lda_topic"],
        categories
    )

    visualization.plot_interactive(
        tfidf_tsne,
        tfidf_umap,
        df_train["target"],
        df_train["kmeans_cluster"],
        categories
    )

    print("\n" + "="*50)
    print("Evaluating Clustering Models")
    print("="*50)

    kmeans_metrics = evaluation.calculate_metrics(df_train["target"], df_train["kmeans_cluster"], tfidf_matrix)

    lda_silhouette = evaluation.silhouette_score(count_matrix, df_train["lda_topic"])
    lda_metrics = {
        "ARI": evaluation.adjusted_rand_score(df_train["target"], df_train["lda_topic"]),
        "Silhouette": lda_silhouette
    }

    results_df = pd.DataFrame({
        "Method": ["K-means", "LDA"],
        "Adjusted Rand Index": [kmeans_metrics["ARI"], lda_metrics["ARI"]],
        "Silhouette Score": [kmeans_metrics["Silhouette"], lda_metrics["Silhouette"]],
    })

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    evaluation.plot_comparison_bar(results_df)

    # Confusion matrices
    evaluation.plot_confusion_matrix(
        df_train["target"],
        df_train["kmeans_cluster"],
        "K-means Clustering Confusion Matrix",
        categories,
    )

    evaluation.plot_confusion_matrix(
        df_train["target"],
        df_train["lda_topic"],
        "LDA Topic Modeling Confusion Matrix",
        categories,
    )

    # Topic analysis
    evaluation.plot_topic_distributions(lda_matrix)

    avg_topic_dist = np.mean(lda_matrix, axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(avg_topic_dist)), avg_topic_dist, alpha=0.7, color='skyblue')
    plt.title('Average Topic Distribution Across All Documents')
    plt.xlabel('Topic')
    plt.ylabel('Average Probability')
    plt.show()

    print("\n" + "="*50)
    print("Performing Cluster Stability Analysis")
    print("="*50)

    ari_stability, sil_stability = evaluation.cluster_stability_analysis(tfidf_matrix, n_clusters)

    evaluation.plot_stability(ari_stability, sil_stability)

    # Topic diversity
    lda_diversity = evaluation.topic_diversity(lda_topics, topk=10)
    print(f"LDA Topic Diversity (top 10 words): {lda_diversity:.3f}")

    evaluation.plot_topic_coherence(
        lda_topics,
        df_train,
        lda.components_,
        n_topics
    )

    # Advanced visualizations
    visualization.plot_advanced_dashboard(
        df_train,
        categories,
        tfidf_tsne,
        tfidf_umap,
        feature_names,
        tfidf_dense,
        lda_matrix,
        kmeans_metrics,
        lda_metrics,
        n_topics,
    )

    # Analyze clusters
    print("\nK-means Cluster Analysis")
    kmeans_analysis = evaluation.analyze_cluster_characteristics(
        df_train,
        cluster_col="kmeans_cluster",
        text_col="processed_text"
    )
    for cluster_id, analysis in kmeans_analysis.items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {analysis['size']} documents")
        print(f"  Average length: {analysis['avg_length']:.1f} characters")
        print(f"  Most common category: {analysis['most_common_category']}")
        print(f"  Category purity: {analysis['category_purity']:.3f}")
        print(f"  Top terms: {', '.join(top_terms_per_cluster[cluster_id][:5])}")

    print("\n\nLDA Topic Analysis")
    lda_analysis = evaluation.analyze_cluster_characteristics(
        df_train,
        cluster_col="lda_topic",
        text_col="processed_text"
    )
    for topic_id, analysis in lda_analysis.items():
        print(f"\nTopic {topic_id}:")
        print(f"  Size: {analysis['size']} documents")
        print(f"  Average length: {analysis['avg_length']:.1f} characters")
        print(f"  Most common category: {analysis['most_common_category']}")
        print(f"  Category purity: {analysis['category_purity']:.3f}")
        print(f"  Top words: {', '.join(lda_topics[topic_id][:5])}")

if __name__ == "__main__":
    main()
