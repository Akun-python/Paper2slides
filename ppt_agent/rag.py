from __future__ import annotations

import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ppt_agent.logging_utils import get_logger, make_log_print

try:  # Optional vector store backend for upgraded RAG.
    import chromadb
    from chromadb.config import Settings
except Exception:  # noqa: BLE001
    chromadb = None  # type: ignore[assignment]
    Settings = None  # type: ignore[assignment]


_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBED_CLIENT: OpenAI | None = None
logger = get_logger("paper2slides.rag")
print = make_log_print(logger)


def _get_embedding_client() -> OpenAI:
    """Lazily construct a shared OpenAI client for embeddings.

    The client uses environment variables instead of hardcoded keys:
      - EMBEDDING_API_KEY or APIYI_API_KEY or OPENAI_API_KEY
      - EMBEDDING_BASE_URL or APIYI_BASE_URL (default https://api.apiyi.com/v1)
    """

    global _EMBED_CLIENT
    if _EMBED_CLIENT is not None:
        return _EMBED_CLIENT

    api_key = (
        os.getenv("EMBEDDING_API_KEY")
        or os.getenv("APIYI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    base_url = os.getenv("EMBEDDING_BASE_URL") or os.getenv("APIYI_BASE_URL") or "https://api.apiyi.com/v1"

    if not api_key:
        raise RuntimeError(
            "No embedding API key configured; set EMBEDDING_API_KEY or APIYI_API_KEY or OPENAI_API_KEY."
        )

    timeout_raw = os.getenv("EMBEDDING_TIMEOUT")
    if timeout_raw:
        try:
            timeout_val = float(timeout_raw)
        except ValueError:
            timeout_val = None
    else:
        timeout_val = None

    if timeout_val is not None and timeout_val > 0:
        _EMBED_CLIENT = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_val)
    else:
        _EMBED_CLIENT = OpenAI(api_key=api_key, base_url=base_url)
    return _EMBED_CLIENT


def _embed_text(text: str, model: str = _EMBEDDING_MODEL) -> List[float]:
    """Return a single embedding vector for the given text.

    Text is sent as a single-element batch to keep the call structure close to
    the user's example and to simplify error handling.
    """

    text = (text or "").strip()
    if not text:
        return []
    client = _get_embedding_client()
    retries_raw = os.getenv("EMBEDDING_RETRIES", "1").strip()
    try:
        retries = max(1, int(retries_raw))
    except ValueError:
        retries = 1
    backoff_raw = os.getenv("EMBEDDING_RETRY_BACKOFF", "0.5").strip()
    try:
        backoff = max(0.0, float(backoff_raw))
    except ValueError:
        backoff = 0.5

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.embeddings.create(input=[text], model=model)
            if not resp.data:
                return []
            return list(resp.data[0].embedding)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries and backoff > 0:
                time.sleep(backoff)

    if last_exc:
        raise last_exc
    return []


def _vector_rag_disabled() -> bool:
    raw = os.getenv("PIPELINE_DISABLE_VECTOR_RAG") or os.getenv("DISABLE_VECTOR_RAG")
    if not raw:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def build_keyword_index(pdf_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a hybrid RAG index over PDF chunks.

    The returned object keeps:
      - items: original per-chunk records with page_content/metadata/lower
      - collection: optional Chroma collection populated with embeddings

    Callers should treat the return value as an opaque index handle and pass
    it back to keyword_search. If vector indexing or embeddings are
    unavailable, collection is set to None and keyword_search will fall back
    to the original phrase-level substring scoring.
    """

    items: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(pdf_chunks or []):
        chunk = chunk or {}
        text = (chunk.get("page_content") or "").strip()
        metadata = chunk.get("metadata") or {}
        items.append(
            {
                "index": idx,
                "page_content": text,
                "metadata": metadata,
                "lower": text.lower(),
            }
        )

    collection = None

    # Best-effort: if chromadb or embeddings are not available, we still
    # return items so that substring-based RAG remains functional.
    if chromadb is not None and Settings is not None and not _vector_rag_disabled():
        try:
            client = chromadb.Client(Settings(anonymized_telemetry=False))
            # Use a per-index unique collection name so that repeated runs or
            # multiple indices in the same process do not clash with an
            # already-existing persistent collection.
            collection_name = f"pdf_chunks_index_{uuid.uuid4().hex[:8]}"
            collection = client.create_collection(name=collection_name)

            docs: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            ids: List[str] = []
            embeddings: List[List[float]] = []

            for item in items:
                text = item.get("page_content") or ""
                idx = item.get("index")
                metadata = item.get("metadata") or {}
                if not text or not isinstance(idx, int):
                    continue
                try:
                    emb = _embed_text(text)
                except Exception as exc:  # noqa: BLE001
                    print(f"[RAG] Skipping chunk {idx} during vector indexing due to embedding error: {exc}")
                    continue
                if not emb:
                    continue
                meta_with_index = dict(metadata)
                meta_with_index["index"] = idx
                docs.append(text)
                metadatas.append(meta_with_index)
                ids.append(f"chunk-{idx}")
                embeddings.append(emb)

            if docs:
                collection.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=embeddings)
                print(f"[RAG] Built Chroma index with {len(docs)} embedded chunks.")
        except Exception as exc:  # noqa: BLE001
            print(f"[RAG] Failed to build Chroma-based vector index; falling back to substring RAG only: {exc}")
            collection = None

    return {"items": items, "collection": collection}


def _simple_tokenize(text: str) -> List[str]:
    if not text:
        return []
    # Split on non-word characters, drop empty tokens.
    return [tok for tok in re.split(r"\W+", text.lower()) if tok]


def _extract_query_phrases(text: str) -> List[str]:
    """Extract short phrases/sentences from a natural-language query.

    The goal is to support sentence-level retrieval: the user provides one or
    several short sentences, and we look for chunks that contain these
    sentences (or long phrases) as substrings, instead of doing bag-of-words
    keyword matching.
    """

    if not text:
        return []

    raw = text.strip().lower()
    if not raw:
        return []

    # Split on common sentence delimiters (both English and Chinese) so that
    # we can treat each piece as a candidate phrase.
    parts = re.split(r"[。！？!?；;\n]+", raw)
    phrases: List[str] = []
    for part in parts:
        p = part.strip()
        # Skip very short fragments; we want reasonably informative phrases.
        if len(p) >= 4:
            phrases.append(p)

    # Fallback: if splitting produced nothing but the original text is longer
    # than a few characters, search using the whole query.
    if not phrases and len(raw) >= 4:
        phrases = [raw]

    return phrases


def _substring_keyword_search(
    items: List[Dict[str, Any]],
    phrases: List[str],
    pages: Optional[List[int]] = None,
    k: int = 8,
    preferred_pages: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Original phrase-level substring RAG implementation.

    Kept as a helper so that we can fall back when vector search is
    unavailable or fails.
    """

    if not items or not phrases:
        return []

    pages_set = set(pages) if pages else None
    preferred_set = set(preferred_pages) if preferred_pages else None
    scored: List[tuple[int, Dict[str, Any]]] = []

    for item in items:
        meta = item.get("metadata") or {}
        page = meta.get("page")

        if pages_set is not None and page not in pages_set:
            continue

        text_lower = item.get("lower") or ""
        score = 0

        for phrase in phrases:
            if not phrase:
                continue
            count = text_lower.count(phrase)
            if count > 0:
                score += count * max(len(phrase), 4)

        if score > 0:
            if preferred_set is not None and page in preferred_set:
                score += 10
            scored.append((score, item))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    top_items = [item for score, item in scored[: max(k, 1)]]

    by_index: Dict[int, Dict[str, Any]] = {}
    for it in items:
        idx = it.get("index")
        if isinstance(idx, int):
            by_index[idx] = it

    expanded_indices: set[int] = set()
    for it in top_items:
        idx = it.get("index")
        if not isinstance(idx, int):
            continue
        for j in (idx - 1, idx, idx + 1):
            if j in by_index:
                expanded_indices.add(j)

    if not expanded_indices:
        return top_items

    return [by_index[i] for i in sorted(expanded_indices)]


def keyword_search(
    index: Any,
    query: str,
    pages: Optional[List[int]] = None,
    k: int = 8,
    preferred_pages: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Return up to k chunks that best match the query using hybrid RAG.

    When the index contains a Chroma collection, we first try vector search:
      - Encode the full query into an embedding
      - Query the collection for the top-k nearest chunks (optionally
        filtered by pages)
      - Apply a small bonus for preferred_pages and then expand neighbors
        by index-1/index/index+1.

    If vector search is unavailable or fails, we fall back to the original
    phrase-level substring implementation.
    """

    if not index or not query:
        return []

    phrases = _extract_query_phrases(query)
    if not phrases:
        return []

    # Unpack the hybrid index format produced by build_keyword_index.
    if isinstance(index, dict) and "items" in index:
        items = index.get("items") or []
        collection = index.get("collection")
    else:
        # Backwards compatibility: index is the old plain list
        items = list(index or [])
        collection = None

    # Attempt vector search when a collection is present.
    if collection is not None and not _vector_rag_disabled():
        try:
            emb = _embed_text(query)
            if emb:
                where = None
                if pages:
                    where = {"page": {"$in": list(pages)}}

                n_results = max(k, 4)
                res = collection.query(
                    query_embeddings=[emb],
                    n_results=n_results,
                    where=where,
                )

                docs = (res.get("documents") or [[]])[0]
                metas = (res.get("metadatas") or [[]])[0]
                dists = (res.get("distances") or [[]])[0]

                by_index: Dict[int, Dict[str, Any]] = {}
                for it in items:
                    idx_val = it.get("index")
                    if isinstance(idx_val, int):
                        by_index[idx_val] = it

                scored_vec: List[tuple[float, Dict[str, Any]]] = []
                preferred_set = set(preferred_pages) if preferred_pages else None

                for doc, meta, dist in zip(docs, metas, dists):
                    meta = meta or {}
                    idx = meta.get("index")
                    page = meta.get("page")
                    if not isinstance(idx, int) or idx not in by_index:
                        continue
                    item = by_index[idx]
                    # Convert distance to a similarity-like score; smaller
                    # distance => larger score.
                    try:
                        base_score = -float(dist) if dist is not None else 0.0
                    except Exception:  # noqa: BLE001
                        base_score = 0.0
                    if preferred_set is not None and page in preferred_set:
                        base_score += 1.0
                    scored_vec.append((base_score, item))

                if scored_vec:
                    scored_vec.sort(key=lambda x: x[0], reverse=True)
                    top_items = [it for score, it in scored_vec[: max(k, 1)]]

                    expanded_indices: set[int] = set()
                    for it in top_items:
                        idx = it.get("index")
                        if not isinstance(idx, int):
                            continue
                        for j in (idx - 1, idx, idx + 1):
                            if j in by_index:
                                expanded_indices.add(j)

                    if expanded_indices:
                        return [by_index[i] for i in sorted(expanded_indices)]
                    return top_items
        except Exception as exc:  # noqa: BLE001
            print(f"[RAG] Vector search failed, falling back to substring RAG: {exc}")

    # Fallback: original substring-based behavior.
    return _substring_keyword_search(items, phrases, pages=pages, k=k, preferred_pages=preferred_pages)
