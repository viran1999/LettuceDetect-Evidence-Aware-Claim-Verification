#!/usr/bin/env python3
"""Small, qualitative evidence-ranking comparison for RAGTruth.

Run from the LettuceDetect repository root:
    python evaluate_evidence_ranking.py

This script does NOT measure retrieval accuracy. The CSV is for manual review.
Gold hallucination spans are used to select test claims only; label_type and
annotation explanations are never supplied to the ranking algorithms.
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TASKS = ("QA", "Data2Txt", "Summary")


def normalize_task(value):
    s = str(value or "").lower().replace("-", "").replace("_", "").replace(" ", "")
    if s in ("qa", "questionanswering", "questionanswer"):
        return "QA"
    if s in ("data2txt", "datatotext", "d2t"):
        return "Data2Txt"
    if s in ("summary", "summarization", "summarisation"):
        return "Summary"
    return None


def evidence_as_text(obj):
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        # Some QA/Data2Txt examples store structured evidence.
        return "\n".join(f"{k}: {evidence_as_text(v)}" for k, v in obj.items())
    if isinstance(obj, list):
        return "\n".join(evidence_as_text(x) for x in obj)
    return str(obj) if obj is not None else ""


def sentence_split(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def passage_windows(text):
    sentences = sentence_split(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        # Structured data may not be sentence-like; split by length for a
        # manageable output, retaining some overlap.
        s = sentences[0]
        return [s[i:i + 500] for i in range(0, len(s), 400)] if len(s) > 650 else [s]
    return [" ".join(sentences[i:i + 2]) for i in range(len(sentences) - 1)]


def sentence_containing_span(answer, start, end):
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end > len(answer) or end <= start:
        return ""
    # Find boundaries without requiring a full NLP sentence tokenizer.
    left_candidates = [answer.rfind(c, 0, start) for c in (".", "?", "!", "\n")]
    left = max(left_candidates) + 1
    right_candidates = [answer.find(c, end) for c in (".", "?", "!", "\n")]
    right_candidates = [x for x in right_candidates if x >= 0]
    right = min(right_candidates) + 1 if right_candidates else len(answer)
    return answer[left:right].strip()


def source_lookup(path):
    result = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                result[str(obj.get("source_id"))] = obj
    return result


def sample_examples(response_path, sources, each, max_source_chars):
    selected = {t: [] for t in TASKS}
    with response_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            response = json.loads(line)
            src = sources.get(str(response.get("source_id")))
            if src is None:
                continue
            task = normalize_task(src.get("task_type"))
            if task not in selected or len(selected[task]) >= each:
                continue
            answer = response.get("response")
            labels = response.get("labels") or []
            if not isinstance(answer, str) or not isinstance(labels, list):
                continue
            evidence = evidence_as_text(src.get("source_info"))
            if not evidence.strip():
                # Some examples may have the actual source in the prompt.
                evidence = evidence_as_text(src.get("prompt"))
            if not evidence.strip():
                continue
            for label in labels:
                claim = sentence_containing_span(answer, label.get("start"), label.get("end"))
                if len(claim) >= 12:
                    selected[task].append({
                        "task": task,
                        "source_id": str(response.get("source_id")),
                        "response_id": str(response.get("id", "")),
                        "claim": claim,
                        "annotated_span": answer[label["start"]:label["end"]],
                        "evidence": evidence[:max_source_chars],
                    })
                    break
    return selected


def rank_three(claim, passages, model, top_k=3):
    # Within-example TF-IDF similarity.
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = tfidf.fit_transform([claim] + passages)
    lexical = cosine_similarity(matrix[:1], matrix[1:]).ravel()

    embeddings = model.encode([claim] + passages, normalize_embeddings=True, show_progress_bar=False)
    semantic = np.asarray(embeddings[1:] @ embeddings[0], dtype=float)

    # Simple date-aware feature; no annotation details are inspected.
    claim_years = set(re.findall(r"\b(?:1[5-9]\d{2}|20\d{2})\b", claim))
    claim_has_date = bool(claim_years or re.search(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b", claim, re.I))
    date_boost = np.zeros(len(passages), dtype=float)
    if claim_has_date:
        for i, p in enumerate(passages):
            evidence_years = set(re.findall(r"\b(?:1[5-9]\d{2}|20\d{2})\b", p))
            if evidence_years:
                date_boost[i] = 0.12  # dates may differ; conflicting dates are useful evidence

    # No use of gold label, label_type, or annotation meta.
    hybrid = 0.35 * lexical + 0.65 * np.maximum(semantic, 0) + date_boost
    results = {}
    for method, scores in (("TF-IDF", lexical), ("Semantic", semantic), ("Hybrid", hybrid)):
        ordered = np.argsort(-scores, kind="stable")[:min(top_k, len(passages))]
        results[method] = [(int(i), float(scores[i]), passages[i]) for i in ordered]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/ragtruth")
    parser.add_argument("--examples-per-task", type=int, default=10)
    parser.add_argument("--output", default="evidence_ranking_review.csv")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-source-chars", type=int, default=15000)
    args = parser.parse_args()
    data = Path(args.data_dir)
    for f in (data / "source_info.jsonl", data / "response.jsonl"):
        if not f.is_file():
            raise SystemExit(f"Missing {f}; run from your repository root or specify --data-dir")

    print("Loading RAGTruth dataset...")
    sources = source_lookup(data / "source_info.jsonl")
    selected = sample_examples(data / "response.jsonl", sources, args.examples_per_task, args.max_source_chars)
    counts = {t: len(selected[t]) for t in TASKS}
    print("Selected examples:", counts)
    if any(n < args.examples_per_task for n in counts.values()):
        print("NOTE: Some task types have fewer usable annotated examples than requested.")
    if not any(counts.values()):
        raise SystemExit("No usable examples found. Check your RAGTruth files.")

    print(f"Loading semantic model: {args.model}")
    model = SentenceTransformer(args.model, device="cpu")
    rows = []
    for task in TASKS:
        for j, ex in enumerate(selected[task], 1):
            passages = passage_windows(ex["evidence"])
            if not passages:
                continue
            try:
                rankings = rank_three(ex["claim"], passages, model)
            except ValueError as err:
                print(f"Skipping {task} example {j}: {err}")
                continue
            for method, items in rankings.items():
                for rank, (idx, score, passage) in enumerate(items, 1):
                    rows.append({
                        "task": task,
                        "example_number": j,
                        "response_id": ex["response_id"],
                        "source_id": ex["source_id"],
                        "annotated_span_for_review_only": ex["annotated_span"],
                        "claim_sentence": ex["claim"],
                        "method": method,
                        "rank": rank,
                        "score": round(score, 4),
                        "passage_index": idx,
                        "evidence_passage": passage,
                        "useful_for_verification_manual_0_or_1": "",
                    })
            print(f"{task}: {j}/{len(selected[task])} processed")

    if not rows:
        raise SystemExit("No results generated")
    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to: {output.resolve()}")
    print("Each example has up to 3 passages for each of the 3 methods.")
    print("For manual review, fill the final column with 1 (useful) or 0 (not useful).")
    print("This CSV is NOT an accuracy/F1/Recall@3 result without reviewed relevance labels.")
    print("Please do not push dataset contents or model files to GitHub.")


if __name__ == "__main__":
    main()
