from __future__ import annotations
import json
import pathlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_vectorizer: TfidfVectorizer | None = None
_matrix = None
_catalog: list[dict] = []

CATALOG_PATH = pathlib.Path(__file__).parent / "catalog.json"

def _text(item: dict) -> str:
    levels = " ".join(item.get("competencies", []))
    type_full = item.get("test_type_full", item.get("test_type", ""))
    # repeat name for higher weight
    return (
        f"{item['name']} {item['name']} {item['name']} "
        f"{item.get('description', '')} "
        f"{levels} {type_full} "
        f"{'remote online' if item.get('remote_testing') else ''} "
        f"{'adaptive' if item.get('adaptive') else ''}"
    )

# query expansion: vague terms → catalog-relevant terms
_EXPANSIONS: dict[str, str] = {
    "analytical":    "numerical reasoning cognitive ability",
    "people skills": "personality behaviour interpersonal",
    "communication": "verbal personality behaviour",
    "under pressure":"personality resilience behaviour",
    "team":          "personality behaviour interpersonal",
    "leadership":    "manager personality behaviour competency",
    "technical":     "knowledge skills programming",
    "coding":        "programming knowledge skills developer",
    "logical":       "reasoning deductive inductive cognitive",
    "quick":         "short brief assessment",
    "senior":        "mid-professional experienced",
    "junior":        "graduate entry-level",
    "entry":         "graduate entry-level",
    "graduate":      "graduate entry-level",
}

def _expand(query: str) -> str:
    q = query.lower()
    extras = [v for k, v in _EXPANSIONS.items() if k in q]
    return query + " " + " ".join(extras)

def build_index(catalog: list[dict]) -> None:
    global _vectorizer, _matrix, _catalog
    print(f"[retrieval] building TF-IDF index over {len(catalog)} items...")
    _catalog = catalog
    texts = [_text(item) for item in catalog]
    _vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=1,
        max_features=30000,
        sublinear_tf=True,
        analyzer="word",
    )
    _matrix = _vectorizer.fit_transform(texts)
    print(f"[retrieval] ready — {_matrix.shape[0]} docs, {_matrix.shape[1]} features")

def retrieve(query: str, top_k: int = 15) -> list[dict]:
    if _vectorizer is None or _matrix is None:
        raise RuntimeError("build_index() not called")
    expanded = _expand(query)
    q_vec  = _vectorizer.transform([expanded])
    scores = cosine_similarity(q_vec, _matrix).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_idx:
        if scores[idx] > 0:
            entry = dict(_catalog[idx])
            entry["_score"] = float(scores[idx])
            results.append(entry)
    return results

def retrieve_by_name(name: str) -> dict | None:
    name_lower = name.lower()
    best, best_score = None, 0
    for item in _catalog:
        iname = item["name"].lower()
        if iname == name_lower:
            return item
        overlap = len(set(name_lower.split()) & set(iname.split()))
        if overlap > best_score:
            best_score, best = overlap, item
    return best if best_score > 0 else None

def load_catalog() -> list[dict]:
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)
