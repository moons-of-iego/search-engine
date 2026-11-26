from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from scipy.sparse import spmatrix
from src.loader import load_dataset
from src.encoder import encode_matrix
import numpy as np


def compare_texts(
    X1: spmatrix | np.ndarray,
    X2: spmatrix | np.ndarray,
    vectorizer: CountVectorizer | None = None,
):
    score = cosine_similarity(X1, X2)

    common_words = []
    if vectorizer is not None:
        features = vectorizer.get_feature_names_out()
        idx1 = X1.nonzero()[1]
        idx2 = X2.nonzero()[1]
        common_idx = np.intersect1d(idx1, idx2)
        common_words = [str(features[i]) for i in common_idx]

    return score, common_words


def display_comparison(
    title1: str, title2: str, score: float, common_words: list[str]
) -> None:
    score = float(score[0, 0])
    sep = "=" * 75
    print(sep)
    print(f"Doc 1 : {title1}")
    print(f"Doc 2 : {title2}")
    print("-" * 75)
    print(f"Similarité cosinus : {score:.4f} (brut : {score})")
    print(
        f"Mots en commun ({len(common_words)}) : {', '.join(common_words) if common_words else 'Aucun'}"
    )
    print(sep)


if __name__ == "__main__":
    corpus, _, _ = load_dataset()
    X, vectorizer, doc_ids = encode_matrix(corpus)

    pairs = []
    rng = np.random.default_rng()
    for i in range(50):
        pairs.append(rng.integers(0, len(corpus), 2))

    for idx in pairs:

        v1, v2 = X[idx[0]], X[idx[1]]
        id1, id2 = doc_ids[idx[0]], doc_ids[idx[1]]
        title1, title2 = corpus[id1]["title"], corpus[id2]["title"]

        score, common = compare_texts(v1, v2, vectorizer)
        display_comparison(title1, title2, score, common)
