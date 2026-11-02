#!/usr/bin/env python3
"""Create blinded evidence-relevance reviews; score paired v1/v2 rankings after human annotation.

Reviewers annotate one unique claim/passage pair, regardless of which system
returned it. Do not interpret ranking score as hallucination probability.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

TASKS = ("QA", "Data2Txt", "Summary")


def jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def claim_id(record, span):
    return f"{record['response_id']}:{span['start']}:{span['end']}"


def passage_id(claim_key, passage):
    clean = re.sub(r"\s+", " ", passage).strip().casefold()
    return hashlib.sha256((claim_key + "\0" + clean).encode()).hexdigest()[:20]


def collect_claims(results_path):
    all_claims = []
    for record in jsonl(results_path):
        for item in record.get("ranked_claims", []):
            span = item["baseline_span"]
            ranks = item.get("ranked_evidence", {})
            all_claims.append({
                "claim_id": claim_id(record, span),
                "source_id": record["source_id"], "response_id": record["response_id"],
                "task": record["task"], "claim": item["claim"], "rankings": ranks,
            })
    return all_claims


def selected_claims(claims, per_task, seed):
    rng = random.Random(seed)
    selected = []
    for task in TASKS:
        subset = [c for c in claims if c["task"] == task]
        rng.shuffle(subset)
        used_sources = set()
        for c in subset:
            if c["source_id"] in used_sources:
                continue
            used_sources.add(c["source_id"])
            selected.append(c)
            if len(used_sources) >= per_task:
                break
    return selected


def create_review(args):
    claims = collect_claims(Path(args.results))
    selected = selected_claims(claims, args.claims_per_task, args.seed)
    if not selected:
        raise ValueError("No predicted claims to annotate; check batch output")
    fieldnames = ["claim_id", "task", "response_id", "source_id", "claim", "passage_id", "evidence_passage", "useful_0_or_1_or_U"]
    rows = []
    for c in selected:
        seen = set()
        for method in ("v1", "v2"):
            for item in c["rankings"].get(method, []):
                passage = item.get("evidence", "")
                pid = passage_id(c["claim_id"], passage)
                if not passage.strip() or pid in seen:
                    continue
                seen.add(pid)
                rows.append({"claim_id": c["claim_id"], "task": c["task"],
                             "response_id": c["response_id"], "source_id": c["source_id"],
                             "claim": c["claim"], "passage_id": pid,
                             "evidence_passage": passage, "useful_0_or_1_or_U": ""})
    rng = random.Random(args.seed + 1)
    rng.shuffle(rows)
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite previous human review: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Generated {len(rows)} unique relevance judgments for {len(selected)} claims in {target}")
    print("Fill 1 = useful to verify claim, 0 = not useful, U = uncertain; do NOT use score or method identity.")
    print("Per-task claims:", dict(Counter(c["task"] for c in selected)))


def read_review(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    labels = {}
    reviewed = set()
    for row in rows:
        mark = str(row.get("useful_0_or_1_or_U", "")).strip().upper()
        if mark not in {"0", "1", "U"}:
            continue
        key = (row["claim_id"], row["passage_id"])
        if key in labels and labels[key] != mark:
            raise ValueError(f"Conflicting human ratings for {key}")
        labels[key] = mark
        reviewed.add(row["claim_id"])
    return labels, reviewed


def metrics_for(ranked_labels):
    """Each entry is a ranked list of binary labels; return retrieval metrics."""
    if not ranked_labels:
        return {"n": 0, "p_at_1": None, "hit_at_3": None, "mrr_at_3": None, "p_at_3": None}
    n = len(ranked_labels)
    return {"n": n,
            "p_at_1": sum(int(bool(a) and a[0] == 1) for a in ranked_labels) / n,
            "hit_at_3": sum(int(1 in a[:3]) for a in ranked_labels) / n,
            "mrr_at_3": sum(next((1 / (i + 1) for i, x in enumerate(a[:3]) if x == 1), 0) for a in ranked_labels) / n,
            "p_at_3": sum(sum(a[:3]) for a in ranked_labels) / (3 * n)}


def collect_scored_pairs(claims, labels):
    evaluated = []
    missing = []
    uncertain = []
    for c in claims:
        methods = {}
        complete = True
        for method in ("v1", "v2"):
            out = []
            for item in c["rankings"].get(method, [])[:3]:
                pid = passage_id(c["claim_id"], item.get("evidence", ""))
                key = (c["claim_id"], pid)
                mark = labels.get(key)
                if mark is None:
                    missing.append(key)
                    complete = False
                elif mark == "U":
                    uncertain.append(key)
                    complete = False
                else:
                    out.append(int(mark))
            methods[method] = out
        if complete and any(methods.values()):
            evaluated.append({"claim_id": c["claim_id"], "source_id": c["source_id"],
                              "task": c["task"], "methods": methods})
    return evaluated, missing, uncertain


def bootstrap_delta_source_level(rows, metric, B=1000, seed=20261101):
    """Paired bootstrap resampling SOURCES (not individual claims)."""
    sources = defaultdict(list)
    for row in rows:
        sources[row["source_id"]].append(row)
    ids = sorted(sources)
    if len(ids) < 2:
        return (None, None)
    rng = random.Random(seed)
    sampled_deltas = []
    for _ in range(B):
        draw = [row for _ in ids for row in sources[rng.choice(ids)]]
        score_a = metrics_for([row["methods"]["v1"] for row in draw])[metric]
        score_b = metrics_for([row["methods"]["v2"] for row in draw])[metric]
        sampled_deltas.append(score_b - score_a)
    sampled_deltas.sort()
    return sampled_deltas[int(0.025 * (B - 1))], sampled_deltas[int(0.975 * (B - 1))]


def score(args):
    claims = collect_claims(Path(args.results))
    labels, reviewed = read_review(args.review)
    pairs, missing, uncertain = collect_scored_pairs(
        [c for c in claims if c["claim_id"] in reviewed], labels)
    print(f"Complete paired claims: {len(pairs)}; missing judgments: {len(set(missing))}; uncertain: {len(set(uncertain))}")
    if not pairs:
        raise ValueError("No completed paired examples; fill both methods' unique passages in review file")
    summary = []
    for task in ("ALL",) + TASKS:
        subset = [r for r in pairs if task == "ALL" or r["task"] == task]
        if not subset:
            continue
        a = metrics_for([r["methods"]["v1"] for r in subset])
        b = metrics_for([r["methods"]["v2"] for r in subset])
        for metric in ("p_at_1", "hit_at_3", "mrr_at_3", "p_at_3"):
            lo, hi = bootstrap_delta_source_level(subset, metric, B=args.bootstrap)
            summary.append({"task": task, "metric": metric, "n_claims": len(subset),
                            "n_sources": len({r["source_id"] for r in subset}),
                            "v1": round(a[metric], 4), "v2": round(b[metric], 4),
                            "delta_v2_minus_v1": round(b[metric] - a[metric], 4),
                            "delta_95ci_low": "" if lo is None else round(lo, 4),
                            "delta_95ci_high": "" if hi is None else round(hi, 4)})
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for row in summary:
        if row["task"] == "ALL":
            print(f"{row['metric']}: v1={row['v1']:.3f}, v2={row['v2']:.3f}, delta={row['delta_v2_minus_v1']:+.3f}")
    print("Saved:", target)
    print("NOTE: These are HUMAN-ANNOTATED retrieval metrics, not hallucination-detection F1 or recall@3.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    review = sub.add_parser("prepare")
    review.add_argument("--results", required=True, help="Batch predicted_claims.jsonl")
    review.add_argument("--claims-per-task", type=int, default=40)
    review.add_argument("--seed", type=int, default=20261101)
    review.add_argument("--output", required=True)
    score_p = sub.add_parser("score")
    score_p.add_argument("--results", required=True)
    score_p.add_argument("--review", required=True)
    score_p.add_argument("--output", required=True)
    score_p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()
    if args.mode == "prepare":
        create_review(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
