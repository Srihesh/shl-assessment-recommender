from __future__ import annotations
import json
import pathlib
import numpy as np

_embeddings: np.ndarray | None = None
_model = None
_catalog: list[dict] = []

CATALOG_PATH = pathlib.Path(__file__).parent / "catalog.json"
MODEL_NAME   = "BAAI/bge-small-en-v1.5"  # 24MB ONNX, no PyTorch

def _text(item: dict) -> str:
    levels = " ".join(item.get("competencies", []))
    return (
        f"{item['name']} {item['name']} "
        f"{item.get('description', '')} "
        f"{levels} "
        f"{item.get('test_type_full', item.get('test_type', ''))} "
        f"{'remote' if item.get('remote_testing') else ''}"
    )

def build_index(catalog: list[dict]) -> None:
    global _embeddings, _model, _catalog
    from fastembed import TextEmbedding

    print(f"[retrieval] loading {MODEL_NAME}...")
    _model   = TextEmbedding(MODEL_NAME)
    _catalog = catalog

    texts = [_text(item) for item in catalog]
    print(f"[retrieval] embedding {len(texts)} items...")
    raw = list(_model.embed(texts))
    _embeddings = np.array(raw, dtype="float32")

    # L2-normalise for cosine via dot product
    norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
    _embeddings = _embeddings / np.where(norms == 0, 1, norms)
    print(f"[retrieval] index ready — {_embeddings.shape[0]} vectors, dim={_embeddings.shape[1]}")

def retrieve(query: str, top_k: int = 15) -> list[dict]:
    if _model is None or _embeddings is None:
        raise RuntimeError("build_index() not called")

    q_vec = np.array(list(_model.embed([query])), dtype="float32")
    norm  = np.linalg.norm(q_vec)
    q_vec = q_vec / (norm if norm > 0 else 1)

    scores     = (_embeddings @ q_vec.T).flatten()
    top_idx    = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
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
