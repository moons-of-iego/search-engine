from src.loader import load_corpus, load_qrels, load_queries
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_avg_score(qrels_valid: dict):
    ratios = []
    for docs in qrels_valid.values():
        ratios.append(sum(docs.values()) / len(docs))
    return ratios


def plot_ratios_distribution(ratios: list[float]):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))

    mean_ratio = np.mean(ratios)
    median_ratio = np.median(ratios)

    sns.histplot(ratios, bins=20, kde=False, color="#66c2a5", edgecolor="white")

    plt.axvline(
        mean_ratio,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label=f"Moyenne : {mean_ratio:.3f} ({mean_ratio*100:.1f}%)",
    )
    plt.axvline(
        median_ratio,
        color="navy",
        linestyle=":",
        linewidth=1.5,
        label=f"Médiane : {median_ratio:.3f} ({median_ratio*100:.1f}%)",
    )

    plt.title(
        f"Distribution de la proportion de documents pertinents ({len(ratios)} requêtes)",
        fontweight="bold",
    )
    plt.xlabel("Proportion de documents pertinents par requête (ratio)")
    plt.ylabel("Nombre de requêtes")
    plt.legend()
    plt.tight_layout()
    plt.show()


def query_example(qrels: dict, corpus: dict, queries: dict):
    example_query_id = next(iter(qrels))
    candidates = qrels[example_query_id]
    positive_ids = [doc_id for doc_id, score in candidates.items() if score == 1]
    negative_ids = [doc_id for doc_id, score in candidates.items() if score == 0]

    query_title = queries[example_query_id]["text"]
    positive_titles = [corpus[doc_id]["title"] for doc_id in positive_ids]
    negative_titles = [corpus[doc_id]["title"] for doc_id in negative_ids]

    print(f"Requête : {query_title}\n")
    print("Articles cités (positifs) :")
    for title in positive_titles:
        print(f"  - {title}")

    print("\nArticles aléatoires (négatifs) :")
    for title in negative_titles[:5]:
        print(f"  - {title}")


if __name__ == "__main__":
    # Load the dataset

    print("Loading dataset...")
    corpus = load_corpus(Path("dataset/corpus.jsonl"))
    queries = load_queries(Path("dataset/queries.jsonl"))
    qrels_valid = load_qrels(Path("dataset/valid.tsv"))

    print(f"Loaded {len(corpus)} documents in corpus")
    print(f"Loaded {len(queries)} queries")
    print(f"Loaded relevance for {len(qrels_valid)} queries (dataset)")

    pair_query_nb = sum(len(docs) for docs in qrels_valid.values())
    print(f"Loaded {pair_query_nb} queries / documents pairs.")
    assert pair_query_nb == len(
        pd.read_csv(Path("dataset/valid.tsv"), sep="\t")
    ), "number of queries / documents pairs different than valid.tsv length."

    query_example(qrels_valid, corpus, queries)
    ratios = get_avg_score(qrels_valid)
    plot_ratios_distribution(ratios)
