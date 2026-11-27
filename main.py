from src.loader import load_dataset
from src.encoder import encode_matrix, encode_matrix_tfidf
from src.engine import search_engine

query = (
    "DC and RF breakdown voltage characteristics of SiGe HBTs for WiFi PA applications"
)

if __name__ == "__main__":
    corpus, _, _ = load_dataset()
    X_corpus, vectorizer, doc_ids = encode_matrix_tfidf(corpus)
    result = search_engine(corpus, X_corpus, vectorizer, doc_ids, query=query)
    print(result.to_string(index=False))
