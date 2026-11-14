"""
Vector store for the Flame in Freefall application.
Uses ChromaDB when installed, with seamless fallback to in-memory NumPy/Scikit-learn
cosine similarity vector store.
"""

import hashlib
import json
import logging
import os
import threading
from typing import Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    chromadb = None
    HAS_CHROMADB = False

import numpy as np

from config import CHROMA_DB_DIR, EMBEDDING_DIMENSION, TOP_K_RESULTS
from core.embeddings import embed_texts, embedding_signature

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "flame_in_freefall"


def _chunk_id(chunk: dict[str, Any]) -> str:
    meta = chunk.get("metadata", {}) or {}
    key = "|".join(
        str(meta.get(k, "")) for k in ("source", "experiment_name", "chunk_index")
    )
    return hashlib.sha1(f"{key}|{chunk.get('text', '')}".encode("utf-8")).hexdigest()


def _chunk_digest(chunks: list[dict[str, Any]]) -> str:
    """Stable fingerprint of the corpus, independent of chunk ordering."""
    joined = "|".join(sorted(_chunk_id(chunk) for chunk in chunks))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


class VectorStore:
    """Vector store providing semantic search over experiment data chunks."""

    def __init__(self, persist_dir: str | None = None):
        self._persist_dir = persist_dir or CHROMA_DB_DIR
        self._use_chroma = HAS_CHROMADB
        self._collection = None
        self._fallback_docs: list[dict[str, Any]] = []
        self._fallback_vectors: Optional[np.ndarray] = None
        self._embedding_dim: Optional[int] = None
        self._lock = threading.RLock()

        if self._use_chroma:
            try:
                os.makedirs(self._persist_dir, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=self._persist_dir,
                    settings=Settings(anonymized_telemetry=False),
                )
                self._collection = self._client.get_or_create_collection(
                    name=_COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(
                    "ChromaDB VectorStore initialised with %d documents",
                    self._collection.count(),
                )
            except Exception as e:
                logger.warning("ChromaDB init failed (%s), switching to in-memory store", e)
                self._use_chroma = False

        if not self._use_chroma:
            logger.info("Using NumPy vector store fallback (persistent JSON cache).")
            self._load_fallback_cache()

    # ── Fallback Cache Persistence ────────────────────────────────────────────

    def _cache_file_path(self) -> str:
        return os.path.join(self._persist_dir, "vector_cache.json")

    def _meta_file_path(self) -> str:
        return os.path.join(self._persist_dir, "index_meta.json")

    def get_index_meta(self) -> dict[str, Any]:
        try:
            with open(self._meta_file_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_index_meta(self, chunks: list[dict[str, Any]]) -> None:
        meta = {
            "chunk_digest": _chunk_digest(chunks),
            "chunk_count": len(chunks),
            "embedding_signature": embedding_signature(),
            "embedding_dimension": self._embedding_dim,
        }
        try:
            os.makedirs(self._persist_dir, exist_ok=True)
            path = self._meta_file_path()
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            os.replace(tmp_path, path)
        except OSError as e:
            logger.warning("Could not write vector index metadata: %s", e)

    def index_is_current(self, chunks: list[dict[str, Any]]) -> bool:
        """True when the persisted index matches the corpus and embedding model."""
        if not chunks:
            return True
        meta = self.get_index_meta()
        if not meta:
            return False
        return (
            meta.get("chunk_digest") == _chunk_digest(chunks)
            and meta.get("embedding_signature") == embedding_signature()
            and self.get_collection_stats().get("document_count") == len(chunks)
        )

    def _load_fallback_cache(self) -> None:
        cpath = self._cache_file_path()
        if os.path.exists(cpath):
            try:
                with open(cpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                docs = data.get("docs", [])
                vectors = data.get("vectors", [])
                if len(docs) != len(vectors):
                    raise ValueError(
                        f"cache mismatch: {len(docs)} docs vs {len(vectors)} vectors"
                    )
                if vectors:
                    arr = np.array(vectors, dtype=np.float32)
                    if arr.ndim != 2:
                        raise ValueError("cache vectors are not a 2D matrix")
                    self._fallback_vectors = arr
                    self._embedding_dim = int(arr.shape[1])
                self._fallback_docs = docs
                logger.info("Loaded %d documents from local vector cache.", len(self._fallback_docs))
            except Exception as e:
                logger.warning("Could not read vector cache: %s", e)

    def _save_fallback_cache(self) -> None:
        os.makedirs(self._persist_dir, exist_ok=True)
        cpath = self._cache_file_path()
        try:
            vectors_list = self._fallback_vectors.tolist() if self._fallback_vectors is not None else []
            tmp_path = f"{cpath}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump({"docs": self._fallback_docs, "vectors": vectors_list}, f)
            os.replace(tmp_path, cpath)
        except Exception as e:
            logger.warning("Could not write vector cache: %s", e)

    # ── Add Documents ─────────────────────────────────────────────────────────

    def add_documents(self, chunks: list[dict[str, Any]]) -> int:
        """Add chunked documents with embeddings and metadata.

        Uses deterministic IDs so re-ingesting the same chunks is a no-op
        instead of appending duplicates.
        """
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        if not embeddings or len(embeddings) != len(chunks):
            logger.error(
                "Embedding count mismatch (%d embeddings for %d chunks)",
                len(embeddings), len(chunks),
            )
            return 0

        dimension = len(embeddings[0])
        with self._lock:
            self._embedding_dim = dimension

            if self._use_chroma and self._collection is not None:
                try:
                    self._upsert_chroma(chunks, texts, embeddings)
                    self._save_index_meta(chunks)
                    return len(chunks)
                except Exception as e:
                    logger.warning(
                        "Failed to add to ChromaDB (%s); rebuilding collection.", e
                    )
                    try:
                        self.clear()
                        self._upsert_chroma(chunks, texts, embeddings)
                        self._save_index_meta(chunks)
                        return len(chunks)
                    except Exception as retry_error:
                        logger.error(
                            "ChromaDB rebuild failed (%s), using NumPy fallback.",
                            retry_error,
                        )

            # Fallback NumPy addition
            new_vecs = np.array(embeddings, dtype=np.float32)
            existing = {_chunk_id(d) for d in self._fallback_docs}
            keep = [i for i, c in enumerate(chunks) if _chunk_id(c) not in existing]
            if not keep:
                return 0

            if self._fallback_vectors is None or len(self._fallback_vectors) == 0:
                self._fallback_vectors = new_vecs[keep]
                self._fallback_docs = [chunks[i] for i in keep]
            elif self._fallback_vectors.shape[1] != dimension:
                logger.warning(
                    "Embedding dimension changed (%d -> %d); rebuilding NumPy store.",
                    self._fallback_vectors.shape[1], dimension,
                )
                self._fallback_vectors = new_vecs[keep]
                self._fallback_docs = [chunks[i] for i in keep]
            else:
                self._fallback_vectors = np.vstack(
                    [self._fallback_vectors, new_vecs[keep]]
                )
                self._fallback_docs.extend(chunks[i] for i in keep)

            self._save_fallback_cache()
            self._save_index_meta(chunks)
            return len(keep)

    def _upsert_chroma(
        self,
        chunks: list[dict[str, Any]],
        texts: list[str],
        embeddings: list[list[float]],
    ) -> None:
        ids = [f"chunk_{_chunk_id(c)}" for c in chunks]
        metadatas = [
            {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
             for k, v in c.get("metadata", {}).items()}
            for c in chunks
        ]
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def clear(self) -> None:
        """Remove all stored documents (used to rebuild after re-embedding)."""
        with self._lock:
            if self._use_chroma and self._collection is not None:
                self._client.delete_collection(_COLLECTION_NAME)
                self._collection = self._client.get_or_create_collection(
                    name=_COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )

            self._fallback_docs = []
            self._fallback_vectors = None
            self._embedding_dim = None
            cache = self._cache_file_path()
            if os.path.exists(cache):
                try:
                    os.remove(cache)
                except OSError as e:
                    logger.warning("Could not remove vector cache: %s", e)
            meta = self._meta_file_path()
            if os.path.exists(meta):
                try:
                    os.remove(meta)
                except OSError as e:
                    logger.warning("Could not remove vector index metadata: %s", e)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> list[dict[str, Any]]:
        """Search for the most relevant documents given a query string."""
        if not query.strip():
            return []

        embeddings = embed_texts([query])
        if not embeddings:
            return []
        query_vec = embeddings[0]

        with self._lock:
            if self._use_chroma and self._collection is not None and self._collection.count() > 0:
                try:
                    results = self._collection.query(
                        query_embeddings=[query_vec],
                        n_results=min(top_k, self._collection.count()),
                        include=["documents", "metadatas", "distances"],
                    )
                    hits: list[dict[str, Any]] = []
                    docs = results.get("documents", [[]])[0]
                    metas = results.get("metadatas", [[]])[0]
                    dists = results.get("distances", [[]])[0]

                    for doc, meta, dist in zip(docs, metas, dists):
                        score = max(0.0, 1.0 - (dist or 0.0))
                        hits.append({"text": doc, "metadata": meta, "score": score})
                    return hits
                except Exception as e:
                    logger.warning("ChromaDB query failed: %s; using fallback", e)

            # NumPy cosine similarity search
            if self._fallback_vectors is None or len(self._fallback_docs) == 0:
                return []

            q_arr = np.array(query_vec, dtype=np.float32)
            if self._fallback_vectors.shape[1] != q_arr.shape[0]:
                logger.error(
                    "Vector dimension mismatch (index=%d, query=%d); re-ingest required.",
                    self._fallback_vectors.shape[1], q_arr.shape[0],
                )
                return []

            q_norm = np.linalg.norm(q_arr)
            if q_norm > 0:
                q_arr = q_arr / q_norm

            # Vector norms
            norms = np.linalg.norm(self._fallback_vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized_vecs = self._fallback_vectors / norms

            scores = np.dot(normalized_vecs, q_arr)
            top_indices = np.argsort(scores)[::-1][:top_k]

            hits = []
            for idx in top_indices:
                doc = self._fallback_docs[idx]
                hits.append({
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {}),
                    "score": float(scores[idx]),
                })

            return hits

    def get_documents_by_source(
        self, source: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Return every indexed chunk belonging to one source file.

        Used by the challenge-brief extraction path, which needs the complete
        document rather than top-k similarity hits.
        """
        with self._lock:
            docs: list[dict[str, Any]] = []
            if self._use_chroma and self._collection is not None:
                try:
                    result = self._collection.get(
                        where={"source": source},
                        include=["documents", "metadatas"],
                    )
                    for text, meta in zip(
                        result.get("documents") or [],
                        result.get("metadatas") or [],
                    ):
                        docs.append({"text": text, "metadata": meta or {}})
                except Exception as e:
                    logger.warning("ChromaDB source lookup failed (%s); using fallback", e)

            if not docs:
                docs = [
                    doc for doc in self._fallback_docs
                    if (doc.get("metadata") or {}).get("source") == source
                ]

        docs.sort(
            key=lambda doc: int((doc.get("metadata") or {}).get("chunk_index", 0) or 0)
        )
        return docs[:limit] if limit else docs

    def get_collection_stats(self) -> dict[str, Any]:
        """Return basic statistics about the stored documents."""
        with self._lock:
            count = (
                self._collection.count()
                if (self._use_chroma and self._collection)
                else len(self._fallback_docs)
            )
            dimension = (
                int(self._fallback_vectors.shape[1])
                if self._fallback_vectors is not None and len(self._fallback_vectors)
                else self._embedding_dim
            )
            if dimension is None:
                meta = self.get_index_meta()
                dimension = meta.get("embedding_dimension")
                if dimension is None:
                    signature = str(meta.get("embedding_signature", ""))
                    suffix = signature.rsplit(":", 1)[-1]
                    dimension = int(suffix) if suffix.isdigit() else None
            return {
                "collection_name": _COLLECTION_NAME,
                "document_count": count,
                "storage_backend": "ChromaDB" if self._use_chroma else "NumPy Local Store",
                "persist_directory": self._persist_dir,
                "embedding_dimension": dimension,
            }
