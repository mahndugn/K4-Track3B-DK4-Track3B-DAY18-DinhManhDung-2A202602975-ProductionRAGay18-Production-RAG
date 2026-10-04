from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import glob
import hashlib
import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, HIERARCHICAL_CHILD_SIZE, HIERARCHICAL_PARENT_SIZE, SEMANTIC_THRESHOLD


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(
                f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR)."
            )

    families = {}
    for doc in docs:
        title = re.search(r"^#\s+(.+)$", doc["text"], re.MULTILINE)
        version = re.search(r"Phiên bản:\s*([\d.]+)", doc["text"])
        if title and version:
            family = re.sub(r"\s*\([^)]*\)", "", title.group(1)).strip()
            number = tuple(int(part) for part in version.group(1).rstrip(".").split("."))
            doc["metadata"].update(policy_family=family, policy_version=version.group(1))
            families.setdefault(family, []).append((number, doc))
    for members in families.values():
        latest = max(number for number, _ in members)
        for number, doc in members:
            doc["metadata"]["is_current"] = number == latest
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(
                Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)})
            )
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(
            Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)})
        )
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


@lru_cache(maxsize=1)
def _semantic_model():
    from config import OFFLINE

    if OFFLINE:
        from src.offline import OfflineEncoder

        return OfflineEncoder()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def _bounded_split(text: str, size: int) -> list[str]:
    """Prefer paragraph/word boundaries, including for oversized paragraphs."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    result = []
    remaining = text.strip()
    while len(remaining) > size:
        cut = remaining.rfind("\n\n", 0, size + 1)
        if cut <= 0:
            cut = remaining.rfind(" ", 0, size + 1)
        if cut <= 0:
            cut = size
        result.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        result.append(remaining)
    return result


def chunk_semantic(
    text: str, threshold: float = SEMANTIC_THRESHOLD, metadata: dict | None = None
) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    import numpy as np

    if not -1 <= threshold <= 1:
        raise ValueError("threshold must be between -1 and 1")
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n", text) if s.strip()]
    sentences, headings = [], []
    for part in parts:
        if all(re.match(r"^#{1,6}\s+.+$", line) for line in part.splitlines()):
            headings.append(part)
        else:
            sentences.append("\n".join([*headings, part]))
            headings = []
    if headings:
        sentences.append("\n".join(headings))
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(sentences[0], {**(metadata or {}), "strategy": "semantic", "chunk_index": 0})]
    embeddings = _semantic_model().encode(sentences, normalize_embeddings=True)
    groups = [[sentences[0]]]
    for i in range(1, len(sentences)):
        a, b = embeddings[i - 1], embeddings[i]
        similarity = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        if similarity < threshold:
            groups.append([])
        groups[-1].append(sentences[i])
    return [
        Chunk(" ".join(group), {**(metadata or {}), "strategy": "semantic", "chunk_index": i})
        for i, group in enumerate(groups)
    ]


def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    if child_size <= 0 or parent_size <= child_size:
        raise ValueError("require parent_size > child_size > 0")
    metadata = metadata or {}
    identity = str(metadata.get("source", "")) + "\0" + text
    prefix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    parents, children = [], []
    for i, parent_text in enumerate(_bounded_split(text, parent_size)):
        pid = f"{prefix}:parent:{i}"
        parents.append(Chunk(parent_text, {**metadata, "chunk_type": "parent", "parent_id": pid}))
        for child_text in _bounded_split(parent_text, child_size):
            cid = f"{prefix}:child:{len(children)}"
            children.append(
                Chunk(child_text, {**metadata, "chunk_type": "child", "chunk_id": cid}, pid)
            )
    return parents, children


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    chunks, lines, section = [], [], ""
    fence = None

    def flush():
        content = "\n".join(lines).strip()
        if content:
            chunks.append(
                Chunk(
                    content,
                    {
                        **(metadata or {}),
                        "section": section,
                        "strategy": "structure",
                        "chunk_index": len(chunks),
                    },
                )
            )

    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
        if fence is None and not marker and re.match(r"^#{1,6}\s+", line):
            flush()
            lines = []
            section = line.strip()
        lines.append(line)
    flush()
    return chunks


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """

    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
