"""Compare TF-IDF, semantic, and hybrid evidence ranking on one RAGTruth example.

Run from the LettuceDetect repository root with its virtual environment active:
    python test_hybrid_evidence_ranking.py

Requires: scikit-learn, sentence-transformers
Does not modify the existing LettuceDetect implementation or dataset.
"""

import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SOURCE_ID = "15592"
CLAIM = "Anne Frank died on February 7, 2022."
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3


def load_source(source_id: str) -> tuple[str, str]:
    path = Path("data/ragtruth/source_info.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"Dataset missing: {path}. Run from ~/research/LettuceDetect")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if str(item.get("source_id")) != str(source_id):
                continue
            source = item.get("source_info", "")
            if isinstance(source, str):
                evidence = source
            else:
                evidence = json.dumps(source, ensure_ascii=False)
            return evidence, item.get("task_type", "unknown")
    raise ValueError(f"Source ID {source_id} not present in {path}")


def make_passages(text: str, window: int = 2) -> list[str]:
    # Passage windows keep the surrounding context of a factual statement.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return []
    return [" ".join(sentences[i:i + window]) for i in range(len(sentences))]


def normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    low, high = scores.min(), scores.max()
    return np.zeros_like(scores) if high - low < 1e-10 else (scores - low) / (high - low)


def detail_bonus(claim: str, passages: list[str]) -> np.ndarray:
    # Only encourage evidence mentioning the *kind* of detail under investigation.
    # A different date/year can still be crucial contradictory evidence.
    has_year = bool(re.search(r"\b(?:1[5-9]|20)\d{2}\b", claim))
    has_date_word = bool(re.search(
        r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
        claim, flags=re.I))
    has_number = bool(re.search(r"\b\d+\b", claim))
    vals = []
    for passage in passages:
        year = bool(re.search(r"\b(?:1[5-9]|20)\d{2}\b", passage))
        month = bool(re.search(
            r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
            passage, flags=re.I))
        number = bool(re.search(r"\b\d+\b", passage))
        vals.append((0.8 if has_year and year else 0.0)
                    + (0.35 if has_date_word and month else 0.0)
                    + (0.10 if has_number and number else 0.0))
    return np.array(vals)


def show_results(name: str, passages: list[str], scores: np.ndarray) -> None:
    print(f"\n=== {name} ===")
    for rank, index in enumerate(np.argsort(-scores, kind="stable")[:TOP_K], start=1):
        print(f"\nRank {rank} | score: {scores[index]:.4f}")
        print(passages[index])


def main() -> None:
    evidence, task = load_source(SOURCE_ID)
    passages = make_passages(evidence, window=2)
    print(f"Source ID: {SOURCE_ID} | Task: {task}")
    print(f"Claim: {CLAIM}")
    print(f"Evidence length: {len(evidence)} characters | Passage windows: {len(passages)}")

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_vectors = vectorizer.fit_transform([CLAIM] + passages)
    lexical = cosine_similarity(tfidf_vectors[:1], tfidf_vectors[1:]).ravel()

    print(f"\nLoading semantic model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode([CLAIM] + passages, normalize_embeddings=True, show_progress_bar=False)
    semantic = np.asarray(embeddings[1:] @ embeddings[0])

    # Experimental heuristic, NOT a verifier or a trained ranking model.
    bonus = detail_bonus(CLAIM, passages)
    hybrid = 0.35 * normalize(lexical) + 0.45 * normalize(semantic) + 0.20 * normalize(bonus)

    show_results("TF-IDF (two-sentence windows)", passages, lexical)
    show_results("Semantic (two-sentence windows)", passages, semantic)
    show_results("Hybrid (TF-IDF + semantic + date-aware boost)", passages, hybrid)
    print("\nNOTE: Scores from different methods are not directly comparable.")
    print("NOTE: This tests relevance, NOT whether a claim is true or false.")
    print("NOTE: We must test multiple examples before claiming improved retrieval.")


if __name__ == "__main__":
    main()
