"""FAISS vector retrieval module."""

from __future__ import annotations

import json
import logging
import pathlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

# Lazy imports

_model: "SentenceTransformer | None" = None
_index: "faiss.IndexFlatIP | None" = None
_catalog: list[dict] = []

CATALOG_PATH = pathlib.Path(__file__).parent / "catalog.json"
MODEL_NAME = "all-MiniLM-L6-v2"

logger = logging.getLogger(__name__)


def _text_for_item(item: dict) -> str:
    """Concatenate relevant fields into a single indexable string."""
    competencies = ", ".join(item.get("competencies", []))
    return (
        f"{item['name']}. "
        f"{item.get('description', '')}. "
        f"Job levels: {competencies}. "
        f"Type: {item.get('test_type_full', item.get('test_type', ''))}. "
        f"Duration: {item.get('duration', 'unknown')}. "
        f"Remote testing: {'yes' if item.get('remote_testing') else 'no'}."
    )


def build_index(catalog: list[dict]) -> None:
    """
    Embed all catalog items and build the FAISS index.
    Called once at server startup.
    """
    global _model, _index, _catalog

    import faiss  # noqa: PLC0415
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    logger.info(f"Loading embedding model '{MODEL_NAME}' …")
    _model = SentenceTransformer(MODEL_NAME)

    texts = [_text_for_item(item) for item in catalog]
    logger.info(f"Embedding {len(texts)} catalog items …")
    embeddings = _model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    # L2-normalise → inner product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    embeddings = (embeddings / norms).astype("float32")

    dim = embeddings.shape[1]
    _index = faiss.IndexFlatIP(dim)
    _index.add(embeddings)

    _catalog = catalog
    logger.info(f"FAISS index ready — {_index.ntotal} vectors, dim={dim}")


def retrieve(query: str, top_k: int = 15) -> list[dict]:
    """
    Embed query and return top_k most similar catalog entries.

    Always call build_index() before retrieve().
    """
    if _model is None or _index is None:
        raise RuntimeError("retrieval.build_index() has not been called yet.")

    q_emb = _model.encode([query], convert_to_numpy=True).astype("float32")
    norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
    q_emb = q_emb / np.where(norm == 0, 1, norm)

    k = min(top_k, len(_catalog))
    scores, indices = _index.search(q_emb, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        entry = dict(_catalog[idx])
        entry["_score"] = float(score)
        results.append(entry)

    return results


def retrieve_for_comparison(query: str, top_k: int = 4) -> list[dict]:
    """
    For comparison queries, extract product names and retrieve each separately.
    Merge results so both products are guaranteed representation.
    """
    import re
    
    # Extract named products from query
    # Works for "difference between X and Y", "X vs Y", "compare X and Y"
    patterns = [
        r"difference between (.+?) and (.+?)[\?\.]?$",
        r"(.+?) vs \.? (.+?)[\?\.]?$",
        r"compare (.+?) and (.+?)[\?\.]?$",
    ]
    
    names = []
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            names = [match.group(1).strip(), match.group(2).strip()]
            break
    
    if not names:
        # fallback: just use full query
        return retrieve(query, top_k=top_k * 2)
    
    # Retrieve separately for each name, guarantee at least 2 hits per product
    seen_urls = set()
    lists_of_hits = [retrieve(name, top_k=4) for name in names]
    
    results = []
    # Interleave results to ensure both products appear in top N
    for i in range(4):
        for hits_list in lists_of_hits:
            if i < len(hits_list):
                h = hits_list[i]
                if h["url"] not in seen_urls:
                    results.append(h)
                    seen_urls.add(h["url"])
    
    return results


def retrieve_by_name(name: str) -> dict | None:
    """
    Fuzzy name lookup for COMPARE mode.
    Returns the best-matching catalog entry or None.
    """
    name_lower = name.lower()
    best: dict | None = None
    best_score = 0

    for item in _catalog:
        item_name = item["name"].lower()
        # Exact match
        if item_name == name_lower:
            return item
        # Substring / token overlap
        tokens_query = set(name_lower.split())
        tokens_item = set(item_name.split())
        overlap = len(tokens_query & tokens_item)
        if overlap > best_score:
            best_score = overlap
            best = item

    return best if best_score > 0 else None


def load_catalog() -> list[dict]:
    """Load catalog.json from disk."""
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)
