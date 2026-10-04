from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import math
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(
    questions: list[str], answers: list[str], contexts: list[list[str]], ground_truths: list[str]
) -> dict:
    """Run RAGAS evaluation."""
    from config import LLM_MODEL, OFFLINE, OPENAI_API_KEY

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    if len({len(questions), len(answers), len(contexts), len(ground_truths)}) != 1:
        raise ValueError("evaluation input lengths must match")
    rows = [
        EvalResult(q, a, list(c), gt, 0.0, 0.0, 0.0, 0.0)
        for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
    ]
    empty = {
        **{m: 0.0 for m in metrics},
        "per_question": rows,
        "evaluation_status": "unavailable",
        "evaluation_error": None,
    }
    if not questions:
        return {**empty, "evaluation_status": "empty"}
    if not OPENAI_API_KEY or OFFLINE:
        return {**empty, "evaluation_error": "RAGAS requires OPENAI_API_KEY and online mode"}
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
        from ragas.run_config import RunConfig

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY),
            embeddings=OpenAIEmbeddings(api_key=OPENAI_API_KEY),
            run_config=RunConfig(timeout=120, max_retries=2, max_workers=4),
            raise_exceptions=True,
        )
        df = result.to_pandas()
        for i, row in enumerate(rows):
            for metric in metrics:
                value = float(df.iloc[i][metric])
                if not math.isfinite(value):
                    raise ValueError(f"RAGAS returned non-finite {metric} for question {i}")
                setattr(row, metric, value)
        return {
            **{m: sum(getattr(r, m) for r in rows) / len(rows) for m in metrics},
            "per_question": rows,
            "evaluation_status": "completed",
            "evaluation_error": None,
        }
    except Exception as exc:
        # Never present fallback scores as measured RAGAS results.
        for row in rows:
            for metric in metrics:
                setattr(row, metric, 0.0)
        return {**empty, "evaluation_error": f"{type(exc).__name__}: {exc}"}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    tree = {
        "faithfulness": ("LLM hallucinating", "Ground answers in source text; lower temperature"),
        "context_recall": (
            "Missing relevant chunks",
            "Improve chunking, hybrid retrieval and multi-hop coverage",
        ),
        "context_precision": (
            "Too many irrelevant chunks",
            "Add reranking and version metadata filtering",
        ),
        "answer_relevancy": (
            "Answer does not match question",
            "Improve answer prompt and resolve ambiguity",
        ),
    }
    failures = []
    for row in eval_results:
        scores = {m: getattr(row, m) for m in tree}
        worst = min(scores, key=scores.get)
        diagnosis, fix = tree[worst]
        failures.append(
            {
                "question": row.question,
                "answer": row.answer,
                "ground_truth": row.ground_truth,
                "contexts": row.contexts,
                "worst_metric": worst,
                "score": scores[worst],
                "average_score": sum(scores.values()) / 4,
                "diagnosis": diagnosis,
                "suggested_fix": fix,
            }
        )
    return sorted(failures, key=lambda f: f["average_score"])[: max(0, bottom_n)]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
        "per_question": [asdict(r) for r in results.get("per_question", [])],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
