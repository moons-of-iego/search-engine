import json
from typing import Dict
from pathlib import Path
import pandas as pd


def load_jsonl(file_path: Path):
    """Generic JSONL files loader.

    Args:
        file_path (Path): the path to the JSONL file.

    Returns:
        dict: the dictionnary mapping the ID (key '_id') to the content of the JSON.
    """
    res = {}
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            res[doc["_id"]] = doc
    return res


def load_corpus(file_path: Path) -> Dict[str, Dict]:
    """
    Load corpus data from JSONL file.
    Returns dictionary mapping document IDs to document data.
    """
    return load_jsonl(file_path)


def load_queries(file_path: Path) -> Dict[str, Dict]:
    """
    Load query data from JSONL file.
    Returns dictionary mapping query IDs to query data.
    """
    return load_jsonl(file_path)


def load_qrels(file_path: Path) -> Dict[str, Dict[str, int]]:
    """
    Load relevance judgments from TSV file.
    Returns dictionary mapping query IDs to candidate relevance scores.
    """
    qrels = {}
    valid = pd.read_csv(file_path, sep="\t")
    # scores = valid.groupby("query-id")["score"].mean()
    for id, group in valid.groupby("query-id"):
        qrels[str(id)] = dict(zip(group["corpus-id"], group["score"]))
    return qrels


if __name__ == "__main__":
    # Load the dataset
    print("Loading dataset...")
    corpus = load_corpus(Path("dataset/corpus.jsonl"))
    queries = load_queries(Path("dataset/queries.jsonl"))
    qrels_valid = load_qrels(Path("dataset/valid.tsv"))

    print(f"Loaded {len(corpus)} documents in corpus")
    print(f"Loaded {len(queries)} queries")
    print(f"Loaded relevance for {len(qrels_valid)} queries (dataset)")
