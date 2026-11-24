from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import find, spmatrix
from src.loader import load_dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def encode_matrix():
    """
    Encode corpus data on a Bag-of_words matrix. Each line represent a corpus document title and each column is a word.
    Each cell count the number of apparitions of the word on the title.

    Returns:
        the sparse matrix and the CountVectorizer object.
    """
    corpus, _, _ = load_dataset()
    doc_ids = list(corpus.keys())
    doc_titles = [corpus[doc_id]["title"] for doc_id in corpus]
    vectorizer = CountVectorizer()
    X_corpus = vectorizer.fit_transform(doc_titles)

    return X_corpus, vectorizer, doc_ids


def print_feats(v, features, top_n=30):
    _, ids, values = find(v)
    feats = [(ids[i], values[i], features[ids[i]]) for i in range(len(list(ids)))]
    top_feats = sorted(feats, key=lambda x: x[1], reverse=True)[0:top_n]
    return pd.DataFrame(
        {"word": [t[2] for t in top_feats], "value": [t[1] for t in top_feats]}
    )


def corpus_words_freq_distrib(
    X_corpus: spmatrix, features: np.ndarray, top_n: int = 30
):
    v = np.sum(X_corpus, axis=0)

    # 2. Affichage du tableau des top_n mots les plus fréquents
    feats = print_feats(v, features, top_n=top_n)
    print(feats)

    plt.figure(figsize=(10, 7))
    # On inverse l'ordre ([::-1]) pour avoir le mot le plus fréquent tout en haut
    plt.barh(
        feats["word"][::-1], feats["value"][::-1], color="#66c2a5", edgecolor="white"
    )
    plt.title(
        f"Top {top_n} des mots les plus fréquents dans le corpus", fontweight="bold"
    )
    plt.xlabel("Nombre total d'occurrences")
    plt.ylabel("Mots")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    corpus, _, _ = load_dataset()
    X_corpus, vectoriser, doc_ids = encode_matrix()
    print(f"Forme de la matrice: {X_corpus.shape}")
    features = vectoriser.get_feature_names_out()

    print("--")
    print("Example of a line of the sparse matrix:")
    k = 1
    print(f"Title: {corpus[doc_ids[k]]["title"]}")
    print(print_feats(X_corpus[k], features))

    print("--")
    corpus_words_freq_distrib(X_corpus, features)
