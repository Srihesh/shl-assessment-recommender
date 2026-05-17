from __future__ import annotations
import json
import pathlib
import numpy as np

_model = None
_index = None
_catalog: list[dict] = []

CATALOG_PATH = pathlib.Path(__file__).parent / "catalog.json"
MODEL_NAME = "all-MiniLM-L6-v2"

def _text(item: dict) -> str:
    levels = ", ".join(item.get("competencies", []))
    return (
        f"{item['name']}. {item.get('description', '')}. "
        f"Job levels: {levels}. Type: {item.get('test_type_full', item.get('test_type', ''))}. "
        f"Duration: {item.get('duration', 'unknown')}. "
        f"Remote: {'yes' if item.get('remote_testing') else 'no'}."
    )

def build_index(catalog: list[dict]) -> None:
    global _model, _index, _catalog
    import faiss
    from sentence_transformers import SentenceTransformer

    print(f"[retrieval] loading {MODEL_NAME}...")
    _model = SentenceTransformer(MODEL_NAME)

    texts = [_text(item) for item in catalog]
    print(f"[retrieval] embedding {len(texts)} items...")
    emb = _model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = (emb / np.where(norms == 0, 1, norms)).astype("float32")

    _index = faiss.IndexFlatIP(emb.shape[1])
    _index.add(emb)
    _catalog = catalog
    print(f"[retrieval] index ready — {_index.ntotal} vectors")

def retrieve(query: str, top_k: int = 15) -> list[dict]:
    if _model is None or _index is None:
        raise RuntimeError("build_index() not called")

    q = _model.encode([query], convert_to_numpy=True).astype("float32")
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.where(norm == 0, 1, norm)

    scores, indices = _index.search(q, min(top_k, len(_catalog)))
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            entry = dict(_catalog[idx])
            entry["_score"] = float(score)
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
