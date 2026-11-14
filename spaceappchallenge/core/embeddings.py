"""
Embedding utilities for the Flame in Freefall application.
Primary: OpenRouter (OpenAI-compatible) or Google Gemini text-embedding-004.
Fallback: Deterministic TF-IDF / hashing vectors when the API is unavailable.
"""

import hashlib
import logging
import math
import re
import time
from collections import Counter
from typing import Optional

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    requests = None
    HAS_REQUESTS = False

from config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    GEMINI_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_EMBEDDING_MODEL,
    is_openrouter_key,
)

logger = logging.getLogger(__name__)

_BATCH_SIZE = 50
_REQUESTS_PER_MINUTE = 60
_MIN_INTERVAL = 60.0 / _REQUESTS_PER_MINUTE
_LOCAL_EF = None


def embed_texts(
    texts: list[str],
    api_key: Optional[str] = None,
    dimension: int = EMBEDDING_DIMENSION,
) -> list[list[float]]:
    """Embed a list of texts with the configured provider, degrading gracefully."""
    if not texts:
        return []

    key = api_key or GEMINI_API_KEY
    provider = _resolve_provider(key)

    if provider == "openrouter" and HAS_REQUESTS:
        try:
            return _embed_with_openrouter(texts, key, dimension)
        except Exception as e:
            logger.warning("OpenRouter embedding failed, falling back to local: %s", e)
    elif provider == "gemini" and HAS_GENAI:
        try:
            return _embed_with_gemini(texts, key)
        except Exception as e:
            logger.warning("Gemini embedding failed, falling back to local: %s", e)
    elif provider == "tfidf":
        return _embed_tfidf_fallback(texts, dimension)

    return _embed_local_or_tfidf(texts, dimension)


def _resolve_provider(key: str) -> str:
    if EMBEDDING_PROVIDER in ("gemini", "openrouter", "local", "tfidf"):
        return EMBEDDING_PROVIDER
    if key and not is_openrouter_key(key) and key != "your_gemini_api_key_here" and HAS_GENAI:
        return "gemini"
    return "local"


def embedding_signature() -> str:
    """Identify the embedding backend without making a network call.

    Used to invalidate a persisted vector index when the provider or model
    changes (e.g. local MiniLM 384-dim -> Gemini 768-dim).
    """
    key = GEMINI_API_KEY
    provider = _resolve_provider(key)
    if provider == "openrouter":
        return f"openrouter:{OPENROUTER_EMBEDDING_MODEL}:{EMBEDDING_DIMENSION}"
    if provider == "gemini":
        return f"gemini:{EMBEDDING_MODEL}"
    if provider == "tfidf":
        return f"tfidf:{EMBEDDING_DIMENSION}"
    return "local:chroma-minilm-l6-v2:384"


def _embed_local_or_tfidf(texts: list[str], dimension: int) -> list[list[float]]:
    try:
        return _embed_local(texts)
    except Exception as e:
        logger.warning("Local embedding model unavailable (%s), falling back to TF-IDF", e)
        return _embed_tfidf_fallback(texts, dimension)


def _embed_local(texts: list[str]) -> list[list[float]]:
    """Embed offline with ChromaDB's bundled ONNX MiniLM model (~80 MB, cached)."""
    global _LOCAL_EF
    if _LOCAL_EF is None:
        from chromadb.utils import embedding_functions

        _LOCAL_EF = embedding_functions.DefaultEmbeddingFunction()
    return [[float(x) for x in vec] for vec in _LOCAL_EF(texts)]


def _embed_with_openrouter(
    texts: list[str], api_key: str, dimension: int
) -> list[list[float]]:
    """Call the OpenRouter (OpenAI-compatible) embeddings endpoint."""
    url = f"{OPENROUTER_BASE_URL}/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        payload = {
            "model": OPENROUTER_EMBEDDING_MODEL,
            "input": batch,
            "dimensions": dimension,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        rows = sorted(resp.json().get("data", []), key=lambda r: r.get("index", 0))
        embeddings = [r["embedding"] for r in rows]
        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"OpenRouter returned {len(embeddings)} embeddings for {len(batch)} inputs"
            )
        all_embeddings.extend(embeddings)

    return all_embeddings


def _embed_with_gemini(texts: list[str], api_key: str) -> list[list[float]]:
    """Call the Gemini embedding API with batching and rate limiting."""
    genai.configure(api_key=api_key)

    all_embeddings: list[list[float]] = []
    last_request_time = 0.0

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]

        elapsed = time.time() - last_request_time
        if elapsed < _MIN_INTERVAL and last_request_time > 0:
            time.sleep(_MIN_INTERVAL - elapsed)

        result = genai.embed_content(
            model=f"models/{EMBEDDING_MODEL}",
            content=batch,
            task_type="retrieval_document",
        )
        last_request_time = time.time()

        embeddings = result.get("embedding", [])
        if embeddings and isinstance(embeddings[0], float):
            all_embeddings.append(embeddings)
        else:
            all_embeddings.extend(embeddings)

    return all_embeddings


def _embed_tfidf_fallback(texts: list[str], dimension: int) -> list[list[float]]:
    """Generate dense, L2-normalized pseudo-embeddings via word hashing."""
    tokenized = [_tokenize(t) for t in texts]

    # Document frequency
    doc_freq: Counter = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))

    num_docs = max(len(texts), 1)
    vectors: list[list[float]] = []

    for tokens in tokenized:
        vec = [0.0] * dimension
        tf = Counter(tokens)

        for token, count in tf.items():
            df = doc_freq.get(token, 1)
            idf = math.log((1 + num_docs) / (1 + df)) + 1.0
            weight = count * idf
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dimension
            vec[idx] += weight

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        else:
            vec[0] = 1.0

        vectors.append(vec)

    return vectors


def _tokenize(text: str) -> list[str]:
    """Tokenize and normalize text."""
    words = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())
    stop_words = {
        "the", "and", "in", "of", "to", "a", "is", "for", "on", "that", "by", "this",
        "with", "i", "you", "it", "not", "or", "be", "are", "from", "at", "as", "your",
        "all", "have", "new", "more", "an", "was", "we", "will", "can", "us", "if",
    }
    return [w for w in words if w not in stop_words]
