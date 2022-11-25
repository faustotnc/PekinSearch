import hdbscan
import umap
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(
        ngram_range=ngram_range,
        stop_words="english"
    ).fit(documents)

    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j])
                           for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words


# How many documents are in each cluster
def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic']).Doc.count().reset_index().rename(
        {"Topic": "Topic", "Doc": "Size"}, axis='columns').sort_values("Size", ascending=False))
    return topic_sizes


def do_topic_modeling(data, model):
    documents = list(data["clean_text"])
    embeddings = model.encode(documents)

    # Cluster the embedding in a lower dimensional space greater than 2D.
    umap_model_full = umap.UMAP(n_neighbors=3, n_components=5, metric='cosine', random_state=42).fit(embeddings)
    umap_embeddings = umap_model_full.transform(embeddings)
    cluster = hdbscan.HDBSCAN(min_cluster_size=3, metric='euclidean',
                              cluster_selection_method='eom', prediction_data=True,).fit(umap_embeddings)

    # Re-label the points based on the clusters generated above, but now in 2D.
    umap_model_2d = umap.UMAP(n_neighbors=3, n_components=2, metric='cosine', random_state=42).fit_transform(embeddings)
    result = pd.DataFrame(umap_model_2d, columns=['x', 'y'])
    result['labels'] = cluster.labels_

    # Group the Documents into Their Clusters
    docs_df = pd.DataFrame(documents, columns=["Doc"])
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Doc': ' '.join})

    # Create Class-Based TF-IDF Matrix (c-TF-IDF)
    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(data))

    # Extract the Relevant Words in Each Cluster
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=10)

    # Append clusters to dataset
    data['cluster_id'] = cluster.labels_
    data['cluster_x'] = result['x']
    data['cluster_y'] = result['y']

    return data, top_n_words
