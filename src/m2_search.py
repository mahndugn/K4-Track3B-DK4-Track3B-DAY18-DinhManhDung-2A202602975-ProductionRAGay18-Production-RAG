from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    from config import OFFLINE

    if OFFLINE:
        return text.lower().replace("_", " ")
    from underthesea import word_tokenize

    return word_tokenize(text.lower(), format="text").replace("_", " ")


def _tokens(text: str) -> list[str]:
    import re
    import unicodedata

    return re.findall(r"\w+", unicodedata.normalize("NFC", segment_vietnamese(text)))


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = list(chunks)
        self.corpus_tokens = [_tokens(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens) if any(self.corpus_tokens) else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0:
            return []
        scores = self.bm25.get_scores(_tokens(query))
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                self.documents[i]["text"],
                float(scores[i]),
                dict(self.documents[i].get("metadata", {})),
                "bm25",
            )
            for i in indices
            if scores[i] > 0
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient

        from config import OFFLINE

        self.backend = "qdrant_server"
        if OFFLINE:
            self.client = QdrantClient(":memory:")
            self.backend = "qdrant_memory"
            self._encoder = None
            return
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(":memory:")
            self.backend = "qdrant_memory"
            print("Qdrant server unavailable; using an in-memory index.")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from config import OFFLINE

            if OFFLINE:
                from src.offline import OfflineEncoder

                self._encoder = OfflineEncoder()
                return self._encoder
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        if not chunks:
            raise ValueError("cannot build dense index with no chunks")
        vectors = self._get_encoder().encode(
            [c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True
        )
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection, vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
        )
        for start in range(0, len(chunks), 64):
            points = [
                PointStruct(
                    id=i,
                    vector=vectors[i].tolist(),
                    payload={"text": chunks[i]["text"], "metadata": chunks[i].get("metadata", {})},
                )
                for i in range(start, min(start + 64, len(chunks)))
            ]
            self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(
        self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME
    ) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0 or not self.client.collection_exists(collection):
            return []
        vector = self._get_encoder().encode(query, normalize_embeddings=True).tolist()
        response = self.client.query_points(collection_name=collection, query=vector, limit=top_k)
        return [
            SearchResult(p.payload["text"], float(p.score), p.payload.get("metadata", {}), "dense")
            for p in response.points
        ]


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]], k: int = 60, top_k: int = HYBRID_TOP_K
) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    merged = {}
    for results in results_list:
        seen = set()
        for rank, result in enumerate(results):
            key = result.metadata.get("chunk_id") or (
                result.metadata.get("source", ""),
                result.text,
            )
            if key in seen:
                continue
            seen.add(key)
            if key not in merged:
                merged[key] = [0.0, result]
            merged[key][0] += 1.0 / (k + rank + 1)
    ranked = sorted(merged.values(), key=lambda item: item[0], reverse=True)
    return [
        SearchResult(r.text, score, dict(r.metadata), "hybrid")
        for score, r in ranked[: max(0, top_k)]
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""

    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
