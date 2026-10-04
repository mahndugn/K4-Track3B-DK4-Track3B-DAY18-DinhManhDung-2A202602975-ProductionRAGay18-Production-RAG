"""M5: combined enrichment with explicit extractive fallbacks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from config import LLM_MODEL, OFFLINE, OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def _request(instruction: str, text: str, json_mode: bool = False):
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY, timeout=30, max_retries=1)
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[{"role": "system", "content": instruction}, {"role": "user", "content": text}],
        max_tokens=500,
        **kwargs,
    )
    content = response.choices[0].message.content or ""
    return json.loads(content) if json_mode else content.strip()


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _local(text, source=""):
    sentences = _sentences(text)
    category = "policy"
    for name, words in {
        "it": ["mật khẩu", "vpn", "malware", "mfa"],
        "finance": ["lương", "chi phí", "tạm ứng", "vnđ"],
        "hr": ["nhân viên", "nghỉ phép", "thử việc"],
    }.items():
        if any(word in text.lower() for word in words):
            category = name
            break
    return {
        "summary": " ".join(sentences[:2]),
        "questions": [f"Quy định về {s.rstrip('.!?')} là gì?" for s in sentences[:3]],
        "context": f"Trích từ tài liệu {source}."
        if source
        else "Trích từ tài liệu chính sách nội bộ.",
        "metadata": {
            "topic": (sentences or ["general"])[0][:120],
            "entities": re.findall(r"\b[A-ZĐ]{2,}\b", text),
            "category": category,
            "language": "vi",
            "enrichment_backend": "extractive",
        },
    }


def summarize_chunk(text: str) -> str:
    if OPENAI_API_KEY and not OFFLINE:
        try:
            return _request("Tóm tắt trong 2 câu tiếng Việt; giữ nguyên phủ định và số liệu.", text)
        except Exception as exc:
            print(f"Enrichment fallback: {type(exc).__name__}")
    return _local(text)["summary"]


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    if n_questions <= 0:
        return []
    if OPENAI_API_KEY and not OFFLINE:
        try:
            result = _request(
                f"Tạo {n_questions} câu hỏi mà đoạn văn trả lời được, mỗi câu một dòng.", text
            )
            return [
                re.sub(r"^\s*[\d.)-]+\s*", "", q).strip() for q in result.splitlines() if q.strip()
            ][:n_questions]
        except Exception as exc:
            print(f"Enrichment fallback: {type(exc).__name__}")
    return [f"Quy định về {s.rstrip('.!?')} là gì?" for s in _sentences(text)[:n_questions]]


def contextual_prepend(text: str, document_title: str = "") -> str:
    context = _local(text, document_title)["context"]
    if OPENAI_API_KEY and not OFFLINE:
        try:
            context = _request(
                "Viết 1 câu mô tả vị trí/chủ đề đoạn văn trong tài liệu; không thêm dữ kiện.",
                f"Tài liệu: {document_title}\n{text}",
            )
        except Exception as exc:
            print(f"Enrichment fallback: {type(exc).__name__}")
    return f"{context}\n\n{text}"


def extract_metadata(text: str) -> dict:
    if OPENAI_API_KEY and not OFFLINE:
        try:
            result = _request(
                "Trả về JSON với topic, entities (array), category (policy/hr/it/finance), language.",
                text,
                json_mode=True,
            )
            if not isinstance(result, dict):
                raise ValueError("metadata must be an object")
            return result
        except Exception as exc:
            print(f"Enrichment fallback: {type(exc).__name__}")
    return _local(text)["metadata"]


def _enrich_single_call(text: str, source: str) -> dict:
    if OPENAI_API_KEY and not OFFLINE:
        try:
            result = _request(
                "Phân tích đoạn văn. Trả về JSON với summary (2 câu), questions (3 câu hỏi), "
                "context (1 câu mô tả vị trí trong tài liệu), metadata "
                "(topic, entities, category, language). Giữ nguyên số liệu, phủ định; không bịa dữ kiện.",
                f"Tài liệu: {source}\n\n{text}",
                json_mode=True,
            )
            if not isinstance(result, dict) or any(
                not isinstance(result.get(k), str) for k in ("summary", "context")
            ):
                raise ValueError("invalid enrichment strings")
            if not isinstance(result.get("questions"), list) or not all(
                isinstance(q, str) for q in result["questions"]
            ):
                raise ValueError("invalid enrichment questions")
            if not isinstance(result.get("metadata"), dict):
                raise ValueError("invalid enrichment metadata")
            result["metadata"]["enrichment_backend"] = "openai"
            return result
        except Exception as exc:
            print(f"Enrichment fallback: {type(exc).__name__}")
    return _local(text, source)


def enrich_chunks(chunks: list[dict], methods: list[str] | None = None) -> list[EnrichedChunk]:
    methods = ["combined"] if methods is None else methods
    if set(methods) - {"combined", "summary", "hyqa", "contextual", "metadata"}:
        raise ValueError("unknown enrichment method")
    enriched = []
    for chunk in chunks:
        text, meta = chunk["text"], dict(chunk.get("metadata", {}))
        source = meta.get("source", "")
        if "combined" in methods:
            result = _enrich_single_call(text, source)
            summary, questions = result["summary"], result["questions"]
            enriched_text = f"{result['context']}\n\n{text}"
            auto_meta = result["metadata"]
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}
        enriched.append(
            EnrichedChunk(
                text, enriched_text, summary, questions, {**auto_meta, **meta}, "+".join(methods)
            )
        )
    return enriched
