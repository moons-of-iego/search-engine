import numpy as np
import pandas as pd
from src.encoder import encode_query
from src.comparator import compare_texts
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import find, spmatrix


def search_engine(
    corpus: dict,
    X_corpus: spmatrix,
    vectorizer: CountVectorizer,
    doc_ids: list[str],
    query: str,
    top_k: int = 10,
) -> pd.DataFrame:

    X_query = encode_query(query, vectorizer)
    scores, _ = compare_texts(X_query, X_corpus)
    scores = scores.ravel()
    top_indices = np.argsort(scores)[::-1][:top_k]

    return pd.DataFrame(
        {
            "doc_id": [doc_ids[i] for i in top_indices],
            "title": [corpus[doc_ids[i]]["title"] for i in top_indices],
            "score": [round(float(scores[i]), 4) for i in top_indices],
        }
    )


if __name__ == "__main__":
    results = search_engine(
        query="Efficient Training and Evaluation of Recurrent Neural Network Language Models for Automatic Speech Recognition"
    )
    print(results)
