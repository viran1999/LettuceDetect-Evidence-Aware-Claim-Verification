"""Offline audit of original alignment failures using saved result JSONs.

Run from repository root. Does not call Qwen, LettuceDetect, NLI, or GitHub.
Does not edit existing experiment output. Suggested matches are NOT accuracy.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from enhancements.span_alignment_candidate import propose_alignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/combined_pipeline_batch_bugfix/json")
    parser.add_argument("--output", default="results/span_alignment_phase1_candidates.csv")
    args = parser.parse_args()
    paths = sorted(Path(args.input).glob("*.json"))
    if not paths:
        parser.error("No JSON results in --input directory")

    rows = []
    statuses = Counter()
    prior = Counter()
    examples = 0
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("claims_with_evidence", []):
            verification = item.get("verification") or {}
            if verification.get("decision_reason") != "atomic_claim_match_not_unique":
                continue
            prior["alignment_review_cases"] += 1
            span = (item.get("baseline_span") or {}).get("text", "")
            atomic = [
                a.get("atomic_claim")
                for a in verification.get("atomic_verifications", [])
                if isinstance(a, dict)
            ]
            suggestion = propose_alignment(span, atomic)
            statuses[suggestion["status"]] += 1
            rows.append({
                "response_id": data.get("response_id"),
                "task": data.get("task"),
                "suspicious_span": span,
                "original_claim": item.get("claim"),
                "old_match_status": (item.get("span_verification_context") or {}).get("span_match_status"),
                "new_candidate_status": suggestion["status"],
                "reason": suggestion["reason"],
                "candidate_claims_json": json.dumps([x["claim"] for x in suggestion["candidates"]], ensure_ascii=False),
                "candidate_scores_json": json.dumps(suggestion["candidates"], ensure_ascii=False),
                "original_atomic_claims_json": json.dumps(atomic, ensure_ascii=False),
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["response_id", "task", "suspicious_span", "original_claim", "old_match_status", "new_candidate_status", "reason", "candidate_claims_json", "candidate_scores_json", "original_atomic_claims_json"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("JSON files:", len(paths))
    print("Original alignment-related review cases:", prior["alignment_review_cases"])
    print("Experimental candidate statuses:", dict(statuses))
    print("CSV saved:", output)
    print("IMPORTANT: Candidate counts do not measure correctness or reduced errors.")


if __name__ == "__main__":
    main()
