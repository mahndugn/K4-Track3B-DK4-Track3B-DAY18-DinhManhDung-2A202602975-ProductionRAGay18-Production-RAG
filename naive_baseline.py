"""
Basic RAG Baseline — Chạy TRƯỚC để có scores so sánh.
=====================================================
Basic = paragraph chunking + dense-only search (không hybrid, không rerank, không enrichment).
Đây là RAG đã học ở buổi trước — hôm nay sẽ cải thiện từng bước.
"""

import sys
import os
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.m1_chunking import load_documents, chunk_basic
from src.m2_search import DenseSearch
from src.m4_eval import load_test_set, evaluate_ragas, save_report
from config import NAIVE_COLLECTION, OFFLINE, LLM_MODEL, EMBEDDING_MODEL


def main():
    print("=" * 60)
    print("BASIC RAG BASELINE")
    print("(paragraph chunking + dense-only, no rerank, no enrichment)")
    print("=" * 60)

    docs = load_documents()
    chunks = []
    for doc in docs:
        for c in chunk_basic(doc["text"], metadata=doc["metadata"]):
            chunks.append({"text": c.text, "metadata": c.metadata})
    print(f"  {len(chunks)} basic paragraph chunks")

    search = DenseSearch()
    search.index(chunks, collection=NAIVE_COLLECTION)

    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    from config import OPENAI_API_KEY

    llm_client = None
    if OPENAI_API_KEY and not OFFLINE:
        from openai import OpenAI

        llm_client = OpenAI(api_key=OPENAI_API_KEY, timeout=30, max_retries=1)

    for i, item in enumerate(test_set):
        results = search.search(item["question"], top_k=3, collection=NAIVE_COLLECTION)
        contexts = [r.text for r in results]

        if llm_client and contexts:
            try:
                context_str = "\n\n".join(contexts)
                resp = llm_client.chat.completions.create(
                    model=LLM_MODEL,
                    temperature=0,
                    messages=[
                        {
                            "role": "system",
                            "content": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'",
                        },
                        {
                            "role": "user",
                            "content": f"Context:\n{context_str}\n\nCâu hỏi: {item['question']}",
                        },
                    ],
                )
                answer = resp.choices[0].message.content
            except Exception:
                answer = contexts[0]
        else:
            answer = contexts[0] if contexts else "Không tìm thấy."

        answers.append(answer)
        questions.append(item["question"])
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i + 1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    results["runtime"] = {
        "mode": "offline_smoke" if OFFLINE else "production",
        "embedding_backend": "hashing" if OFFLINE else EMBEDDING_MODEL,
        "vector_store": search.backend,
        "answer_backend": "extractive" if not llm_client else "openai",
    }
    print("\nBASIC BASELINE SCORES")
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        label = (
            f"{results.get(m, 0):.4f}"
            if results.get("evaluation_status") == "completed"
            else "N/A (chưa đo)"
        )
        print(f"  {m}: {label}")
    save_report(results, [], path="reports/naive_baseline_report.json")
    if all(
        results.get(m, 0) == 0
        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    ):
        print(
            f"\nEvaluation status: {results['evaluation_status']}; {results.get('evaluation_error')}"
        )
    print("\nBaseline report saved. Run python main.py for the production comparison.")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total: {time.time() - start:.1f}s")
