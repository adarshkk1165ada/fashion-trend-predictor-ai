from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def run_lda(texts, n_topics=5, n_words=10):

    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        stop_words="english"
    )

    dtm = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42
    )

    topic_matrix = lda.fit_transform(dtm)

    words = vectorizer.get_feature_names_out()

    topics = {}

    for topic_idx, topic in enumerate(lda.components_):

        top_indices = topic.argsort()[:-n_words - 1:-1]

        topic_words = [words[i] for i in top_indices]

        topics[topic_idx] = topic_words

    return topics, topic_matrix