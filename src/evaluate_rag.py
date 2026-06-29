"""
RAGAS Evaluation Benchmark — Phase 4 of the Unified Personal Second Brain.

Evaluates the RAG pipeline using three metrics with 100% local models:
  - Faithfulness         : Is the answer grounded in the retrieved contexts?
  - Answer Relevancy     : Is the answer relevant to the question asked?
  - Context Precision    : Are the retrieved contexts relevant to the ground truth?

Critic LLM  : Ollama / llama3.2  (via langchain_community.chat_models.ChatOllama)
Critic Embed : keepitreal/vietnamese-sbert  (via langchain_huggingface)

No OpenAI key is required.

# ── Installation ──────────────────────────────────────────────────────────────
# pip install ragas>=0.2.0
# pip install datasets>=2.19.0
# pip install pandas>=2.0.0
# pip install pytest>=8.0.0          (for running tests/test_evaluate_rag.py)
# ─────────────────────────────────────────────────────────────────────────────

Usage:
    python src/evaluate_rag.py          # runs with built-in mock QA pairs
    python -m pytest tests/             # runs the unit test suite
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output path — report lands in the project root next to requirements.txt
# ---------------------------------------------------------------------------
_REPORT_PATH = Path(__file__).parent.parent / "ragas_evaluation_report.csv"

# ---------------------------------------------------------------------------
# Mock pipeline hooks — replace these with real ChromaDB + Ollama calls
# ---------------------------------------------------------------------------


def _mock_retrieve_contexts(question: str) -> list[str]:
    """
    Placeholder retriever: returns dummy contexts for a question.

    HOOK: Replace this body with a real ChromaDB similarity search, e.g.:
        docs = _vector_db.similarity_search(question, k=4)
        return [doc.page_content for doc in docs]
    """
    return [
        f"[MOCK CONTEXT A] A retrieved passage directly relevant to: '{question}'.",
        f"[MOCK CONTEXT B] A secondary passage providing supporting detail for: '{question}'.",
    ]


def _mock_generate_answer(question: str, contexts: list[str]) -> str:
    """
    Placeholder generator: returns a dummy answer given a question and contexts.

    HOOK: Replace this body with a real Ollama call, e.g.:
        llm = Ollama(model="llama3.2")
        prompt = _PROMPT.format(context="\\n".join(contexts), question=question)
        return _extract_answer(llm.invoke(prompt))
    """
    joined = " | ".join(c[:60] for c in contexts)
    return f"[MOCK ANSWER] Based on the retrieved context ({joined}), the answer is synthesized here."


# ---------------------------------------------------------------------------
# Local model initialisation — no OpenAI
# ---------------------------------------------------------------------------


def _build_critic_models() -> tuple[Any, Any]:
    """
    Initialise the critic LLM and embedding model using only local inference.

    Returns:
        (critic_llm, critic_embeddings) — RAGAS-compatible wrappers.

    Imports are lazy so the test suite can mock them without loading model weights.
    """
    from langchain_community.chat_models import ChatOllama  # type: ignore[import]
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    logger.info("Loading local critic LLM: llama3.2 via Ollama …")
    raw_llm = ChatOllama(model="llama3.2", temperature=0.0)

    logger.info("Loading local critic embeddings: keepitreal/vietnamese-sbert …")
    raw_emb = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")

    critic_llm = LangchainLLMWrapper(raw_llm)
    critic_emb = LangchainEmbeddingsWrapper(raw_emb)
    return critic_llm, critic_emb


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def run_evaluation(qa_pairs: list[dict[str, str]]) -> pd.DataFrame:
    """
    Run the RAGAS benchmark over a list of QA test pairs.

    Each element in *qa_pairs* must have at minimum:
        "question"     : str  — the user question
        "ground_truth" : str  — the reference (golden) answer

    The function wires in the mock pipeline hooks (_mock_retrieve_contexts and
    _mock_generate_answer) that you can replace with real RAG calls later.

    Args:
        qa_pairs: List of dicts with "question" and "ground_truth" keys.

    Returns:
        A pandas DataFrame with per-sample metric scores (faithfulness,
        answer_relevancy, context_precision).  The DataFrame is also written
        to *ragas_evaluation_report.csv* in the project root.
    """
    # ── Late imports to keep this module testable without loading weights ──
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    if not qa_pairs:
        raise ValueError("qa_pairs must contain at least one entry.")

    # ── 1. Build critic models (local, no OpenAI) ─────────────────────────
    critic_llm, critic_emb = _build_critic_models()

    # Belt-and-suspenders: set the llm/embeddings directly on each metric
    # in addition to passing them to evaluate() below.  This ensures
    # compatibility with both RAGAS 0.1.x and 0.2.x APIs.
    faithfulness.llm = critic_llm
    answer_relevancy.llm = critic_llm
    answer_relevancy.embeddings = critic_emb
    context_precision.llm = critic_llm

    # ── 2. Run the mock pipeline and build the HuggingFace Dataset ────────
    logger.info("Building evaluation dataset from %d QA pairs …", len(qa_pairs))
    rows: list[dict[str, Any]] = []
    for pair in qa_pairs:
        question: str = pair["question"]
        ground_truth: str = pair["ground_truth"]

        contexts: list[str] = _mock_retrieve_contexts(question)
        answer: str = _mock_generate_answer(question, contexts)

        rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,   # list[str] — RAGAS expects a list
                "ground_truth": ground_truth,
            }
        )

    dataset: Dataset = Dataset.from_list(rows)
    logger.info("Dataset schema: %s", dataset.features)

    # ── 3. Run RAGAS evaluate with local critic models ────────────────────
    logger.info("Running RAGAS evaluate …  (this may take a few minutes)")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=critic_llm,
        embeddings=critic_emb,
        show_progress=True,
        raise_exceptions=False,     # return NaN for individual failures instead of crashing
    )

    # ── 4. Export to CSV ─────────────────────────────────────────────────
    df: pd.DataFrame = result.to_pandas()
    df.to_csv(_REPORT_PATH, index=False)
    logger.info("Report saved → %s", _REPORT_PATH)
    print(f"\n[RAGAS] Evaluation complete.  Report → {_REPORT_PATH}")
    print(df[["question", "faithfulness", "answer_relevancy", "context_precision"]].to_string())
    return df


# ---------------------------------------------------------------------------
# Sample QA pairs — Database Systems domain
# ---------------------------------------------------------------------------

SAMPLE_QA_PAIRS: list[dict[str, str]] = [
    {
        "question": "What is Boyce-Codd Normal Form (BCNF) and how does it differ from 3NF?",
        "ground_truth": (
            "BCNF (Boyce-Codd Normal Form) is a stricter version of 3NF. "
            "A relation is in BCNF if, for every non-trivial functional dependency X → Y, "
            "X is a superkey of the relation. Unlike 3NF, BCNF does not allow non-key "
            "attributes to determine other attributes even if the determinant is part of a "
            "candidate key, which can lead to loss of some functional dependencies during "
            "decomposition."
        ),
    },
    {
        "question": "Explain the SELECT and PROJECT operations in Relational Algebra.",
        "ground_truth": (
            "SELECT (σ) filters tuples from a relation that satisfy a given predicate, "
            "producing a horizontal subset (fewer rows). "
            "PROJECT (π) extracts specific columns from a relation, producing a vertical "
            "subset (fewer columns) and eliminating duplicate tuples from the result."
        ),
    },
    {
        "question": "What is a functional dependency and why is it important for normalization?",
        "ground_truth": (
            "A functional dependency X → Y means that the value of attribute set X uniquely "
            "determines the value of attribute set Y in a relation. It is the foundation of "
            "database normalization: each normal form (1NF through BCNF) eliminates a "
            "specific class of anomalous functional dependencies to reduce redundancy and "
            "prevent update, insertion, and deletion anomalies."
        ),
    },
    {
        "question": "What are the ACID properties of a database transaction?",
        "ground_truth": (
            "ACID stands for: Atomicity (a transaction is all-or-nothing), "
            "Consistency (a transaction brings the database from one valid state to another), "
            "Isolation (concurrent transactions execute as if they were serial), and "
            "Durability (committed changes survive system failures, typically via a write-ahead log)."
        ),
    },
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    print("=" * 70)
    print("  Phase 4 — RAGAS Evaluation Benchmark (100% Local Models)")
    print("  Critic LLM  : llama3.2 via Ollama")
    print("  Critic Embed : keepitreal/vietnamese-sbert")
    print("=" * 70)

    report_df = run_evaluation(SAMPLE_QA_PAIRS)

    print("\n── Aggregate Scores ───────────────────────────────────────────────")
    numeric_cols = report_df.select_dtypes("number").columns
    print(report_df[numeric_cols].mean().round(4).to_string())
