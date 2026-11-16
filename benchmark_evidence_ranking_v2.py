#!/usr/bin/env python3
"""Reproducible source-diverse RAGTruth benchmark for the integrated LettuceDetect + v2 ranker.

Runs the ORIGINAL baseline ONCE per response; ranks each predicted claim with v1 and
v2. Raw RAGTruth labels NEVER enter predictions or ranking. Human reviewers must
label retrieval utility; the script does not manufacture precision/recall/F1.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from run_baseline_evidence_integration import (
    DEFAULT_MODEL, claim_containing_span, create_ranker, evidence_as_text,
    extract_question, normalize_spans,
)

TASKS = ("QA", "Data2Txt", "Summary")


def norm_task(v):
    key = "".join(ch for ch in str(v or "").lower() if ch.isalnum())
    return {"qa": "QA", "data2txt": "Data2Txt", "data2text": "Data2Txt",
            "summary": "Summary", "summarization": "Summary"}.get(key)


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def select_records(data_dir: Path, split: str, per_task: int, max_per_source: int,
                   seed: int, exclude_source_ids: set[str] | None = None):
    """Stratify by task with source diversity, deterministic sampling, train/test split."""
    sources = {str(s["source_id"]): s for s in jsonl(data_dir / "source_info.jsonl")}
    candidates = defaultdict(list)
    exclusions = exclude_source_ids or set()
    for r in jsonl(data_dir / "response.jsonl"):
        if str(r.get("split", "")).lower() != split:
            continue
        sid = str(r.get("source_id", ""))
        if sid in exclusions:
            continue
        s = sources.get(sid)
        task = norm_task(s.get("task_type")) if s else None
        if not task or not isinstance(r.get("response"), str) or not r["response"].strip():
            continue
        context = s.get("source_info")
        if context is None or not evidence_as_text(context).strip():
            context = s.get("prompt", "")
        if not evidence_as_text(context).strip():
            continue
        candidates[task].append((r, s))
    rng = random.Random(seed)
    selected = []
    for task in TASKS:
        items = sorted(candidates[task], key=lambda pair: (str(pair[0].get("source_id")), str(pair[0].get("id"))))
        rng.shuffle(items)
        used_sources = Counter()
        chosen = []
        for r, s in items:
            sid = str(r["source_id"])
            if used_sources[sid] < max_per_source:
                chosen.append((r, s))
                used_sources[sid] += 1
            if len(chosen) >= per_task:
                break
        selected.extend(chosen)
    return selected


def evaluation_one(detector, rankers, response, source, max_spans, top_k):
    """Keep model output fixed across rankers, using gold only outside this function."""
    source_info = source.get("source_info")
    if source_info is None or not evidence_as_text(source_info).strip():
        source_info = source.get("prompt", "")
    answer = response["response"]
    question = extract_question(source_info, source)
    prediction = detector.predict(context=[evidence_as_text(source_info)], question=question,
                                  answer=answer, output_format="spans")
    spans = normalize_spans(answer, prediction)
    task = norm_task(source.get("task_type"))
    reviews = []
    for span in spans[:max_spans]:
        claim = claim_containing_span(answer, span["start"], span["end"])
        methods = {name: ranker(claim, source_info, task, top_k) for name, ranker in rankers.items()}
        reviews.append({"baseline_span": span, "claim": claim, "ranked_evidence": methods})
    return {"task": task, "source_id": str(response["source_id"]),
            "response_id": str(response["id"]), "split": response.get("split"),
            "baseline_spans": spans, "ranked_claims": reviews,
            "spans_truncated_for_retrieval": max(0, len(spans) - max_spans)}


def write_manifest(outdir, args, selected):
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "split": args.split, "seed": args.seed, "per_task_requested": args.per_task,
        "max_per_source": args.max_per_source, "max_spans": args.max_spans,
        "top_k": args.top_k, "baseline_model": args.model,
        "rankers": ["v1", "v2"],
        "note": "RAGTruth labels are not supplied to either model. Ranker scores are NOT verification labels.",
        "selected": [{"response_id": str(r["id"]), "source_id": str(r["source_id"]),
                      "task": norm_task(s.get("task_type"))} for r, s in selected],
    }
    path = outdir / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(f"Manifest differs from previous run at {path}. Use a NEW --outdir.")
    else:
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run(args):
    data_dir, outdir = Path(args.data_dir), Path(args.outdir)
    for f in ("response.jsonl", "source_info.jsonl"):
        if not (data_dir / f).is_file():
            raise FileNotFoundError(f"Missing {data_dir / f}")
    exclude = set()
    if args.exclude_manifest:
        m = json.loads(Path(args.exclude_manifest).read_text(encoding="utf-8"))
        exclude = {str(item["source_id"]) for item in m.get("selected", [])}
    selected = select_records(data_dir, args.split, args.per_task, args.max_per_source,
                              args.seed, exclude)
    manifest = write_manifest(outdir, args, selected)
    counts = Counter(x["task"] for x in manifest["selected"])
    print("Selected:", dict(counts), "distinct sources:", len({x["source_id"] for x in manifest["selected"]}), flush=True)
    if any(counts[t] < args.per_task for t in TASKS):
        print("WARNING: Less than requested for at least one task; see manifest.", flush=True)
    results_path = outdir / "predicted_claims.jsonl"
    errors_path = outdir / "errors.jsonl"
    completed = set()
    if results_path.exists():
        completed = {str(x["response_id"]) for x in jsonl(results_path)}
    if not selected or len(completed) == len(selected):
        print("No unprocessed examples.")
        return
    # Heavy pretrained models are initialized once. No model is trained here.
    from lettucedetect.models.inference import HallucinationDetector
    print("Loading baseline:", args.model, flush=True)
    detector = HallucinationDetector(method="transformer", model_path=args.model)
    print("Loading evidence rankers v1 + v2", flush=True)
    rankers = {"v1": create_ranker("v1"), "v2": create_ranker("v2")}
    with results_path.open("a", encoding="utf-8") as results, errors_path.open("a", encoding="utf-8") as errors:
        for i, (r, s) in enumerate(selected, start=1):
            rid = str(r["id"])
            if rid in completed:
                continue
            try:
                result = evaluation_one(detector, rankers, r, s, args.max_spans, args.top_k)
                # Evaluation-only metadata; never passed into either ranker.
                result["gold_has_hallucination_review_only"] = any(
                    not bool(label.get("implicit_true", False)) for label in (r.get("labels") or [])
                    if isinstance(label, dict)
                )
                results.write(json.dumps(result, ensure_ascii=False) + "\n")
                results.flush()
            except Exception as e:
                errors.write(json.dumps({"response_id": rid, "source_id": str(r.get("source_id")),
                                         "error": f"{type(e).__name__}: {e}"}) + "\n")
                errors.flush()
                print(f"[{i}/{len(selected)}] ERROR {rid}: {e}", flush=True)
            else:
                print(f"[{i}/{len(selected)}] {result['task']} {rid}: {len(result['baseline_spans'])} predicted spans", flush=True)
    print("Completed. Output:", results_path, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/ragtruth")
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--per-task", type=int, default=100)
    p.add_argument("--max-per-source", type=int, default=2)
    p.add_argument("--max-spans", type=int, default=3)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261101)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--outdir", default="results/ranking_v2_dev")
    p.add_argument("--exclude-manifest", default="", help="Exclude all dev source IDs from test sampling")
    args = p.parse_args()
    if args.per_task < 1 or args.max_per_source < 1 or args.max_spans < 1 or args.top_k < 1:
        p.error("per-task, max-per-source, max-spans and top-k must be >=1")
    run(args)


if __name__ == "__main__":
    main()
