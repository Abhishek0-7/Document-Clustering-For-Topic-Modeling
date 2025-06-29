from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def create_tfidf_vectorizer(**kwargs):
    return TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.8,
        ngram_range=(1,2),
        stop_words='english',
        **kwargs
    )

def create_count_vectorizer(**kwargs):
    return CountVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.8,
        ngram_range=(1,1),
        stop_words='english',
        **kwargs
    )
