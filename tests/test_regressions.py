"""Regression coverage for provenance, parent expansion and external adapters."""

from types import SimpleNamespace
import numpy as np
import pytest
from src.m1_chunking import Chunk, chunk_hierarchical, chunk_structure_aware
from src.m2_search import DenseSearch, SearchResult, reciprocal_rank_fusion
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import evaluate_ragas


def test_hierarchy_limits_and_unique_document_ids():
    text = "x" * 500 + "\n\n" + "hello world " * 80
    parents, children = chunk_hierarchical(text, 200, 80, {"source": "a"})
    other, _ = chunk_hierarchical(text, 200, 80, {"source": "b"})
    assert all(len(p.text) <= 200 for p in parents)
    assert all(len(c.text) <= 80 for c in children)
    assert {p.metadata["parent_id"] for p in parents}.isdisjoint(
        p.metadata["parent_id"] for p in other
    )
    assert "".join(c.text for c in children).replace(" ", "") == text.replace("\n", "").replace(
        " ", ""
    )


def test_headers_inside_fenced_code_are_not_sections():
    chunks = chunk_structure_aware("# A\n```python\n# code comment\n```\n| a | b |\n## B\ntext")
    assert len(chunks) == 2
    assert "# code comment" in chunks[0].text
    assert "| a | b |" in chunks[0].text


def test_rrf_does_not_count_duplicates_within_one_ranked_list():
    doc = SearchResult("a", 1, {"chunk_id": "a"}, "bm25")
    result = reciprocal_rank_fusion([[doc, doc], [doc]])
    assert result[0].score == pytest.approx(2 / 61)
    assert doc.method == "bm25"


def test_dense_qdrant_adapter_with_fake_encoder():
    from qdrant_client import QdrantClient

    dense = DenseSearch.__new__(DenseSearch)
    dense.client = QdrantClient(":memory:")

    class Encoder:
        def encode(self, text, **kwargs):
            return (
                np.array([1.0, 0.0])
                if isinstance(text, str)
                else np.array([[1.0, 0.0], [0.0, 1.0]])
            )

    dense._encoder = Encoder()
    dense.index([{"text": "a", "metadata": {"chunk_id": "a"}}, {"text": "b"}], "regression")
    results = dense.search("a", 1, "regression")
    assert results[0].text == "a"
    assert results[0].metadata == {"chunk_id": "a"}
    assert results[0].method == "dense"
    dense.client.close()


def test_reranker_uses_predict_scores_and_preserves_metadata():
    reranker = CrossEncoderReranker()
    reranker._model = SimpleNamespace(predict=lambda pairs: np.array([0.1, 0.9]))
    docs = [
        {"text": "a", "score": 4, "metadata": {}},
        {"text": "b", "score": 2, "metadata": {"source": "b.md"}},
    ]
    results = reranker.rerank("q", docs, 1)
    assert results[0].text == "b"
    assert results[0].original_score == 2
    assert results[0].metadata["source"] == "b.md"


def test_unavailable_evaluation_preserves_records(monkeypatch):
    import config

    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    result = evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    assert result["evaluation_status"] == "unavailable"
    assert result["per_question"][0].contexts == ["c"]
    with pytest.raises(ValueError, match="lengths"):
        evaluate_ragas(["q"], [], [], [])


def test_query_expands_parent_without_synthetic_enrichment(monkeypatch):
    from src.pipeline import run_query
    import config

    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    meta = {"source": "a.md", "parent_id": "p", "original_text": "original child"}
    search = SimpleNamespace(
        search=lambda query: [SearchResult("synthetic", 1, meta, "hybrid")],
        parent_lookup={"p": Chunk("whole parent")},
        query_timings=[],
    )

    class Reranker:
        def rerank(self, query, documents, top_k):
            assert documents[0]["text"] == "original child"
            return [SimpleNamespace(text=documents[0]["text"], metadata=meta)]

    answer, contexts = run_query("q", search, Reranker())
    assert "whole parent" in contexts[0]
    assert "synthetic" not in answer


def test_production_model_loaders_use_sentence_transformers(monkeypatch):
    import config
    import sys
    import src.m1_chunking as chunking
    import src.m3_rerank as rerank

    calls = []

    def fake_model(name):
        calls.append(name)
        return object()

    monkeypatch.setattr(config, "OFFLINE", False)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=fake_model, CrossEncoder=fake_model),
    )
    chunking._semantic_model.cache_clear()
    rerank._cached_model.cache_clear()
    try:
        chunking._semantic_model()
        dense = DenseSearch.__new__(DenseSearch)
        dense._encoder = None
        dense._get_encoder()
        CrossEncoderReranker()._load_model()
        assert calls == ["all-MiniLM-L6-v2", config.EMBEDDING_MODEL, "BAAI/bge-reranker-v2-m3"]
    finally:
        chunking._semantic_model.cache_clear()
        rerank._cached_model.cache_clear()


def test_successful_ragas_adapter_aggregates_real_metric_columns(monkeypatch):
    import config
    import ragas
    import pandas as pd
    import langchain_openai

    monkeypatch.setattr(config, "OFFLINE", False)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: object())
    monkeypatch.setattr(langchain_openai, "OpenAIEmbeddings", lambda **kwargs: object())
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    def fake_evaluate(dataset, **kwargs):
        assert len(dataset) == 2
        assert len(kwargs["metrics"]) == 4
        assert kwargs["raise_exceptions"] is True
        return SimpleNamespace(to_pandas=lambda: pd.DataFrame({m: [0.4, 0.8] for m in metrics}))

    monkeypatch.setattr(ragas, "evaluate", fake_evaluate)
    result = evaluate_ragas(["q1", "q2"], ["a1", "a2"], [["c1"], ["c2"]], ["gt1", "gt2"])
    assert result["evaluation_status"] == "completed", result.get("evaluation_error")
    assert result["faithfulness"] == pytest.approx(0.6)
    assert result["per_question"][1].context_recall == 0.8


def test_enrichment_keeps_provenance_and_uses_one_call(monkeypatch):
    import src.m5_enrichment as enrichment

    calls = []
    monkeypatch.setattr(enrichment, "OFFLINE", False)
    monkeypatch.setattr(enrichment, "OPENAI_API_KEY", "test-key")

    def request(*args, **kwargs):
        calls.append(args)
        return {
            "summary": "summary",
            "questions": ["question?"],
            "context": "context",
            "metadata": {"source": "invented", "parent_id": "wrong"},
        }

    monkeypatch.setattr(enrichment, "_request", request)
    result = enrichment.enrich_chunks(
        [{"text": "original", "metadata": {"source": "a.md", "parent_id": "correct"}}]
    )[0]
    assert len(calls) == 1
    assert result.auto_metadata["source"] == "a.md"
    assert result.auto_metadata["parent_id"] == "correct"
    assert "original" in result.enriched_text


def test_loader_marks_latest_policy_version(tmp_path):
    from src.m1_chunking import load_documents

    for version in (1, 2):
        (tmp_path / f"policy_v{version}.md").write_text(
            f"# Policy (v{version})\n> Phiên bản: {version}.0\n\nContent", encoding="utf-8"
        )
    docs = load_documents(str(tmp_path))
    assert docs[0]["metadata"]["is_current"] is False
    assert docs[1]["metadata"]["is_current"] is True


def test_production_segmentation_normalizes_compound_words(monkeypatch):
    import config
    import sys
    from src.m2_search import segment_vietnamese

    calls = []

    def tokenize(text, format):
        calls.append((text, format))
        return "nghỉ_phép năm"

    monkeypatch.setattr(config, "OFFLINE", False)
    monkeypatch.setitem(sys.modules, "underthesea", SimpleNamespace(word_tokenize=tokenize))
    assert segment_vietnamese("Nghỉ PHÉP") == "nghỉ phép năm"
    assert calls == [("nghỉ phép", "text")]
