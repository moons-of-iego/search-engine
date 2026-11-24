import json
import pickle
from typing import Dict, TypeVar, Callable
from pathlib import Path
import pandas as pd

T = TypeVar("T")
CACHE_DIR = Path(".cache")


def _load_with_cache(
    cache_name: str, loader_fn: Callable[[Path], T], raw_path: Path
) -> T:
    """Charge depuis le cache pickle s'il existe, sinon parse le fichier brut et le sauvegarde."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / cache_name

    if cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)

    data = loader_fn(raw_path)
    with cache_path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data


def parse_jsonl(file_path: Path):
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


def _load_corpus(file_path: Path = Path("dataset/corpus.jsonl")) -> Dict[str, Dict]:
    """
    Load corpus data from JSONL file.
    Returns dictionary mapping document IDs to document data.
    """
    return _load_with_cache("corpus.pkl", parse_jsonl, file_path)


def _load_queries(file_path: Path = Path("dataset/queries.jsonl")) -> Dict[str, Dict]:
    """
    Load query data from JSONL file.
    Returns dictionary mapping query IDs to query data.
    """
    return _load_with_cache("queries.pkl", parse_jsonl, file_path)


def _parse_qrels(file_path: Path) -> Dict[str, Dict[str, int]]:
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


def _load_qrels(file_path: Path = Path("dataset/valid.tsv")):
    return _load_with_cache("qrels_valid.pkl", _parse_qrels, file_path)


def load_dataset():
    corpus = _load_corpus()
    queries = _load_queries()
    qrels_valid = _load_qrels()
    print("Loaded dataset in cache.")

    return corpus, queries, qrels_valid


if __name__ == "__main__":
    load_dataset()
