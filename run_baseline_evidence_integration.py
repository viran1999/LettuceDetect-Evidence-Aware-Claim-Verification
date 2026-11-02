#!/usr/bin/env python3
"""Connect real LettuceDetect v1 span predictions to an evidence ranker.

Run in the root of the user's LettuceDetect repo. Never uses RAGTruth gold labels.
The resulting JSON is a diagnostic artifact, NOT an enhanced hallucination verdict.
"""
from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "KRLabsOrg/lettucedect-base-modernbert-en-v1"


def evidence_as_text(obj: Any) -> str:
    """Serialize context for the baseline; preserve False and 0 values."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return "\n".join(f"{key}: {evidence_as_text(value)}" for key, value in obj.items() if value is not None)
    if isinstance(obj, (list, tuple)):
        return "\n".join(evidence_as_text(value) for value in obj if value is not None)
    return str(obj)


def extract_question(source_info: Any, source_record: dict | None = None) -> str:
    if isinstance(source_info, dict):
        for key in ("question", "query"):
            value = source_info.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    # 'prompt' can include the full source/document, not just a question.
    # Do not pass it as a question unless the dataset has an explicit question key.
    if source_record:
        for key in ("question", "query"):
            value = source_record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(source_info, str):
        match = re.search(r"(?im)^\s*question\s*:\s*(.+)$", source_info)
        if match:
            return match.group(1).strip()
    return ""


def claim_containing_span(answer: str, start: int, end: int) -> str:
    """Extract a surrounding sentence; start/end are exact answer offsets."""
    if not (0 <= start < end <= len(answer)):
        raise ValueError(f"Invalid span offsets {start}:{end} for answer of length {len(answer)}")
    left = max([answer.rfind(ch, 0, start) for ch in (".", "!", "?", "\n")]) + 1
    ends = [answer.find(ch, end) for ch in (".", "!", "?", "\n")]
    ends = [i for i in ends if i != -1]
    right = (min(ends) + 1) if ends else len(answer)
    return answer[left:right].strip() or answer[start:end]


def normalize_spans(answer: str, predictions: list[dict]) -> list[dict]:
    result = []
    for i, span in enumerate(predictions):
        if not isinstance(span, dict):
            raise TypeError(f"Span {i} is not a dictionary: {span!r}")
        start, end = span.get("start"), span.get("end")
        if type(start) is not int or type(end) is not int:  # noqa: E721 (bool is not an offset)
            raise ValueError(f"Missing/invalid start or end in span {i}: {span!r}")
        if not (0 <= start < end <= len(answer)):
            raise ValueError(f"Span {i} has offsets out of bounds: {start}:{end}")
        text = answer[start:end]
        # Model output sometimes has different whitespace; answer is authoritative.
        result.append({"start": start, "end": end, "text": text,
                       "confidence": float(span["confidence"]) if span.get("confidence") is not None else None})
    return result


def analyze_record(
    *, detector: Any, answer: str, source_info: Any, task: str,
    question: str, ranker: Callable[[str, Any, str, int], list[dict]],
    top_k: int = 3, source_id: str | None = None, response_id: str | None = None,
) -> dict:
    """Run the baseline first, then rank evidence for actual predicted spans."""
    context = evidence_as_text(source_info)
    if not context.strip():
        raise ValueError("Source evidence is empty")
    if not answer.strip():
        raise ValueError("Answer is empty")
    predictions = detector.predict(
        context=[context], question=question, answer=answer, output_format="spans"
    )
    if not isinstance(predictions, list):
        raise TypeError("Expected LettuceDetect predict(..., output_format='spans') to return a list")
    spans = normalize_spans(answer, predictions)
    retrieval_cache = {}
    reviews = []
    for span in spans:
        claim = claim_containing_span(answer, span["start"], span["end"])
        if claim not in retrieval_cache:
            retrieval_cache[claim] = ranker(claim, source_info, task, top_k)
        reviews.append({
            "baseline_span": dict(span),
            "claim": claim,
            "ranked_evidence": retrieval_cache[claim],
            "verification_status": "NOT_IMPLEMENTED",
        })
    return {
        "source_id": source_id,
        "response_id": response_id,
        "task": task,
        "question": question,
        "answer": answer,
        "baseline_spans": spans,
        "baseline_example_prediction": bool(spans),
        "claims_with_evidence": reviews,
        "note": "Evidence relevance is NOT a support/contradiction verdict. No baseline predictions modified.",
    }


def create_ranker(method: str):
    if method == "v1":
        from enhancements.evidence_ranking import rank_evidence

        def v1(claim: str, source: Any, task: str, k: int) -> list[dict]:
            original = rank_evidence(claim, evidence_as_text(source), top_k=k)
            return [{"rank": j, "evidence": item["evidence"], "score": item.get("score")}
                    for j, item in enumerate(original, start=1)]
        return v1
    if method == "v2":
        try:
            from sentence_transformers import SentenceTransformer
            from enhancements.evidence_ranking_v2 import rank_evidence_v2
        except ImportError as exc:
            raise RuntimeError(
                "For --ranker v2, keep enhancements/evidence_ranking_v2.py in the project "
                "and install sentence-transformers"
            ) from exc
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

        def v2(claim: str, source: Any, task: str, k: int) -> list[dict]:
            return rank_evidence_v2(claim, source, task, model, top_k=k)
        return v2
    if method == "final":
        try:
            from final_evidence_ranker import rank_evidence
            from sentence_transformers import SentenceTransformer, CrossEncoder
        except ImportError as exc:
            raise RuntimeError("For --ranker final, extract the final ranker package and install sentence-transformers") from exc
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device="cpu")

        def final(claim: str, source: Any, task: str, k: int) -> list[dict]:
            original = rank_evidence(claim, source, task, embedder, reranker=reranker, top_k=k)
            return [{"rank": j, "evidence": item.get("evidence_passage", ""),
                     "score": item.get("score"), "passage_kind": item.get("passage_kind")}
                    for j, item in enumerate(original, start=1)]
        return final
    raise ValueError(f"Unknown ranker: {method}")


def load_record(response_id: str, folder: Path) -> tuple[dict, dict]:
    response_path, source_path = folder / "response.jsonl", folder / "source_info.jsonl"
    for p in (response_path, source_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p}; check --data-dir")
    response = None
    with response_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                if str(item.get("id")) == response_id:
                    response = item
                    break
    if response is None:
        raise ValueError(f"No response with id {response_id} in {response_path}")
    source_id = str(response.get("source_id", ""))
    with source_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                if str(item.get("source_id")) == source_id:
                    return response, item
    raise ValueError(f"No source_info for source_id {source_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Use built-in France example")
    mode.add_argument("--response-id", help="RAGTruth response id, e.g. 5661")
    parser.add_argument("--ranker", choices=["v1", "v2", "final"], default="v1")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", default="data/ragtruth")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", default="baseline_evidence_integration_output.json")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    if args.demo:
        answer = "The capital of France is Paris. The population of France is 69 million."
        source_info = "The capital of France is Paris. The population of France is 67 million."
        question = "What is the capital and population of France?"
        task, source_id, response_id = "QA", "demo", "demo"
    else:
        response, source = load_record(args.response_id, Path(args.data_dir))
        answer = response.get("response", "")
        source_info = source.get("source_info")
        if source_info is None or source_info == "":
            source_info = source.get("prompt", "")
        question = extract_question(source_info, source)
        task = str(source.get("task_type", "unknown"))
        source_id, response_id = str(response.get("source_id", "")), str(response.get("id", ""))

    print(f"Loading real LettuceDetect baseline: {args.model}", flush=True)
    from lettucedetect.models.inference import HallucinationDetector
    detector = HallucinationDetector(method="transformer", model_path=args.model)
    ranker = create_ranker(args.ranker)
    result = analyze_record(detector=detector, answer=answer, source_info=source_info,
                            task=task, question=question, ranker=ranker, top_k=args.top_k,
                            source_id=source_id, response_id=response_id)
    result["baseline_model"] = args.model
    result["evidence_ranker"] = args.ranker
    result["ranking_score_note"] = "Relevance ranking only, NOT probability of hallucination."
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nBaseline hallucinated spans: {len(result['baseline_spans'])}")
    for item in result["claims_with_evidence"]:
        print(f"\nPREDICTED: {item['baseline_span']['text']!r}  confidence={item['baseline_span']['confidence']}")
        print(f"CLAIM: {item['claim']}")
        for passage in item["ranked_evidence"]:
            print(f"  #{passage['rank']} [{passage.get('score', '')}] {passage['evidence'][:240]}")
    if not result["baseline_spans"]:
        print("No spans detected. This predicted-span-first approach cannot rescue missed claims.")
    print(f"\nSaved: {output.resolve()}", flush=True)
    print("Not yet a complete enhanced detector: the evidence verifier and fusion are future steps.")


if __name__ == "__main__":
    main()
