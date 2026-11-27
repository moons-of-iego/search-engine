from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    ENGLISH_STOP_WORDS,
)
import re
from nltk.stem import SnowballStemmer
from scipy.sparse import find, spmatrix
from src.loader import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stemmer = SnowballStemmer(language="english")
token_pattern = re.compile(r"(?u)\b\w\w+\b")


def encode_matrix(corpus: dict) -> tuple[spmatrix, CountVectorizer, list[str]]:
    """
    Encode corpus data on a Bag-of_words matrix. Each line represent a corpus document title and each column is a word.
    Each cell count the number of apparitions of the word on the title.

    Returns:
        spmatrix: the sparse matrix Terms x Documents (a row is a term, a col is a document)
        CountVectorizer: the vectorizer
        list[str]: the document's ids.
    """
    doc_ids = list(corpus.keys())
    doc_titles = [corpus[doc_id]["title"] for doc_id in corpus]
    vectorizer = CountVectorizer()
    X_corpus = vectorizer.fit_transform(doc_titles)

    return X_corpus, vectorizer, doc_ids


def tokeniser(text: str) -> list[str]:
    """
    Performs tokenization for the sparse TFxIDF matrix implementation.
    Tokenizes the text, filters the stop-words and applies stemming using the SnowBall algorithm.
    Args:
        text (str): the text to tokenize.

    Returns:
        list[str]: the list of the tokenized words (or semantic units, considering that the stemming step produces non-english words) from the text.
    """
    tokens = token_pattern.findall(text.lower())
    return [stemmer.stem(token) for token in tokens if token not in ENGLISH_STOP_WORDS]


def encode_matrix_tfidf(corpus: dict) -> tuple:
    """
    Encode corpus data on a TFxIDF matrix. Each line represent a corpus document title and each column is a word.
    Each cell is the TFxIDF score of the col word in the row document.
    This encoding remove the english stop words and the hapax (less than 2 occcurences).

    Returns:
        spmatrix: the sparse matrix Terms x Documents (a row is a term, a col is a document)
        CountVectorizer: the vectorizer
        list[str]: the document's ids.
    """
    doc_ids = list(corpus.keys())
    doc_titles = [corpus[doc_id]["title"] for doc_id in corpus]
    vectorizer = TfidfVectorizer(tokenizer=tokeniser, token_pattern=None, min_df=2)
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
    """Returns the top_n most used words in the corpus as a vector and as a plot."""
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


def encode_query(query: str, vectorizer: CountVectorizer) -> spmatrix:
    """Encode a query of multiple keywords using the vectorizer of the corpus."""
    return vectorizer.transform([query])


if __name__ == "__main__":
    corpus, _, _ = load_dataset()
    X_corpus, vectoriser, doc_ids = encode_matrix(corpus)
    print(f"Forme de la matrice: {X_corpus.shape}")
    features = vectoriser.get_feature_names_out()

    print("--")
    print("Example of a line of the sparse matrix:")
    k = 1
    print(f"Title: {corpus[doc_ids[k]]["title"]}")
    print("Vector", print_feats(X_corpus[k], features))

    print("--")
    n = 30
    print(f"{n} most commons words of the corpus:")
    corpus_words_freq_distrib(X_corpus, features, n)
