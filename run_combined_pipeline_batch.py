#!/usr/bin/env python3
"""Run the full LettuceDetect + Hybrid V2 + claim-verification CLI on a
reproducible selection of RAGTruth response IDs.

Requires the teammate's integrated run_lettucedetect_hybrid_v2.py, working
Ollama, Python environment, RAGTruth files, and a ranking benchmark manifest.

This driver invokes the existing CLI once per response so it does not depend on
undocumented implementation details of the teammate's verification modules.
Consequently pretrained models are loaded again for each response. This is
slower than an in-process batch runner, but isolates failures and saves progress.

No ground-truth labels are passed into the pipeline; category frequencies are
NOT classification accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

CATEGORIES = {
    "supported": "Supported",
    "contradicted": "Contradicted",
    "partially supported": "Partially Supported",
    "partial support": "Partially Supported",
    "unverifiable": "Unverifiable",
    "insufficient evidence": "Unverifiable",
}


def read_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run the earlier benchmark or supply --manifest PATH."
        )
    content = json.loads(path.read_text(encoding="utf-8"))
    rows = content.get("selected")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Manifest contains no selected responses: {path}")
    selected = []
    seen = set()
    for item in rows:
        rid = str(item.get("response_id", "")).strip()
        task = str(item.get("task", "Unknown"))
        if not rid:
            raise ValueError("Manifest entry lacks response_id")
        if rid not in seen:
            seen.add(rid)
            selected.append({"response_id": rid, "task": task})
    # Interleave tasks so even a 10-response pilot covers all three tasks.
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        groups[row["task"]].append(row)
    order = [name for name in ("QA", "Data2Txt", "Summary") if name in groups]
    order += sorted(set(groups) - set(order))
    balanced = []
    while any(groups.values()):
        for task in order:
            if groups[task]:
                balanced.append(groups[task].pop(0))
    return balanced


def normalized_category(value) -> str | None:
    if not isinstance(value, str):
        return None
    return CATEGORIES.get(value.strip().lower().replace("_", " "))


def find_category(obj) -> str | None:
    if not isinstance(obj, dict):
        return None
    # Names used across different implementations of a verification adapter.
    for key in ("category", "final_category", "final_status", "verdict", "label", "verification_status", "status"):
        matched = normalized_category(obj.get(key))
        if matched:
            return matched
    for key in ("verification", "categorization", "result", "decision"):
        inner = obj.get(key)
        if isinstance(inner, dict):
            matched = find_category(inner)
            if matched:
                return matched
    return None


def summarize_json(path: Path) -> tuple[int, int, Counter]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Output JSON must contain one object")
    spans = data.get("baseline_spans", [])
    claims = data.get("claims_with_evidence", [])
    if not isinstance(spans, list) or not isinstance(claims, list):
        raise ValueError("Missing baseline_spans or claims_with_evidence lists")
    categories = Counter()
    for claim in claims:
        category = find_category(claim) or "Unknown (inspect JSON)"
        categories[category] += 1
    return len(spans), len(claims), categories


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10, help="Number of responses to test; 10 first, then 300")
    parser.add_argument("--manifest", default="results/ranking_v2_dev/manifest.json")
    parser.add_argument("--data-dir", default="data/ragtruth")
    parser.add_argument("--outdir", default="results/combined_pipeline_batch")
    parser.add_argument("--program", default="run_lettucedetect_hybrid_v2.py")
    parser.add_argument("--timeout", type=int, default=0, help="Timeout per response in seconds; 0 means none")
    parser.add_argument("--retry-failed", action="store_true", help="Retry responses with a previous error log")
    args = parser.parse_args()
    if args.limit < 1 or args.timeout < 0:
        parser.error("--limit must be >=1 and --timeout >=0")

    manifest = Path(args.manifest)
    program = Path(args.program)
    data_dir = Path(args.data_dir)
    if not program.is_file():
        parser.error(f"Integrated entry point not found: {program}")
    for filename in ("response.jsonl", "source_info.jsonl"):
        if not (data_dir / filename).is_file():
            parser.error(f"RAGTruth dataset missing: {data_dir / filename}")
    selection = read_manifest(manifest)[:args.limit]
    if not selection:
        parser.error("No responses selected")
    outdir = Path(args.outdir)
    outputs = outdir / "json"
    logs = outdir / "logs"
    outputs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    print(f"Selected {len(selection)} responses: {dict(Counter(x['task'] for x in selection))}", flush=True)
    print(f"Output directory: {outdir.resolve()}", flush=True)
    print("Note: starts the integrated CLI separately for each response (repeated model loading).", flush=True)

    rows: list[dict] = []
    totals = Counter()
    for i, entry in enumerate(selection, 1):
        rid, task = entry["response_id"], entry["task"]
        path = outputs / f"{rid}.json"
        log = logs / f"{rid}.log"
        error = None
        duration = 0.0
        if path.exists():
            try:
                span_count, claim_count, labels = summarize_json(path)
                status = "cached"
            except (ValueError, json.JSONDecodeError, OSError) as exc:
                error = f"Existing output invalid: {exc}"
                # Rerun it below.
        if not path.exists() or error:
            # Do not retry failed cases by default, but always retry corrupt output.
            if log.is_file() and not error and not args.retry_failed and not path.exists():
                status, span_count, claim_count, labels = "previous_error", 0, 0, Counter()
            else:
                cmd = [sys.executable, str(program), "--response-id", rid,
                       "--data-dir", str(data_dir), "--output", str(path)]
                t0 = time.monotonic()
                try:
                    with log.open("w", encoding="utf-8") as logfile:
                        completed = subprocess.run(
                            cmd, stdout=logfile, stderr=subprocess.STDOUT,
                            timeout=args.timeout or None, check=False,
                        )
                    duration = time.monotonic() - t0
                    if completed.returncode != 0:
                        status = f"error_exit_{completed.returncode}"
                        span_count, claim_count, labels = 0, 0, Counter()
                    elif not path.is_file():
                        status = "error_missing_json"
                        span_count, claim_count, labels = 0, 0, Counter()
                    else:
                        try:
                            span_count, claim_count, labels = summarize_json(path)
                            status = "ok"
                        except (ValueError, json.JSONDecodeError, OSError) as exc:
                            status = f"error_invalid_json:{exc}"
                            span_count, claim_count, labels = 0, 0, Counter()
                except subprocess.TimeoutExpired:
                    duration = time.monotonic() - t0
                    status, span_count, claim_count, labels = "error_timeout", 0, 0, Counter()
                except OSError as exc:
                    duration = time.monotonic() - t0
                    status, span_count, claim_count, labels = f"error_os:{exc}", 0, 0, Counter()
        totals.update(labels)
        result = {"response_id": rid, "task": task, "status": status,
                  "predicted_spans": span_count, "claims": claim_count,
                  "categories": json.dumps(dict(labels), ensure_ascii=False),
                  "seconds": round(duration, 2), "output_path": str(path), "log_path": str(log)}
        rows.append(result)
        print(f"[{i}/{len(selection)}] {task} {rid}: {status}, {span_count} spans, "
              f"{claim_count} claims, labels={dict(labels)}", flush=True)
        with (outdir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    failures = [r for r in rows if r["status"] not in {"ok", "cached"}]
    summary = {
        "responses_selected": len(selection),
        "responses_completed": len(selection) - len(failures),
        "responses_failed_or_skipped": len(failures),
        "responses_with_predicted_spans": sum(r["predicted_spans"] > 0 for r in rows),
        "responses_without_predicted_spans": sum(r["predicted_spans"] == 0 and r["status"] in {"ok", "cached"} for r in rows),
        "predicted_spans_total": sum(r["predicted_spans"] for r in rows),
        "claims_total": sum(r["claims"] for r in rows),
        "verification_categories": dict(totals),
        "warning": "These are predicted category counts, NOT verification accuracy; validate against RAGTruth labels separately."
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSUMMARY\n" + json.dumps(summary, indent=2), flush=True)
    print(f"\nSaved {outdir / 'summary.csv'} and {outdir / 'summary.json'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
