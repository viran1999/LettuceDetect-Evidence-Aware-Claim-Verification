#!/usr/bin/env python3
"""Export LettuceDetect + Hybrid v2 results for the categorization team.

Reads benchmark_evidence_ranking_v2.py's predicted_claims.jsonl and writes
one JSONL record for each *baseline-predicted span*, containing that span,
its surrounding claim, and only Hybrid v2's top-ranked evidence.

RAGTruth gold annotations, v1 evidence, and review labels are never exported.
The script does not run model inference; it converts existing benchmark output.

Ubuntu usage (from the LettuceDetect repository root):
    python export_categorization_input_v2.py

Optional paths:
    python export_categorization_input_v2.py \\
      --input results/ranking_v2_dev/predicted_claims.jsonl \\
      --output results/ranking_v2_dev/categorization_input_v2.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

DEFAULT_INPUT = Path("results/ranking_v2_dev/predicted_claims.jsonl")
DEFAULT_OUTPUT = Path("results/ranking_v2_dev/categorization_input_v2.jsonl")


def convert(input_path: Path, output_path: Path) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input not found: {input_path}\n"
            "Run the v1/v2 benchmark first, or pass --input with the correct path."
        )
    if input_path.resolve() == output_path.resolve():
        raise ValueError("--input and --output must be different files")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".tmp")
    responses = 0
    empty_responses = 0
    claims = 0
    by_task: Counter[str] = Counter()

    try:
        with input_path.open("r", encoding="utf-8-sig") as source, temp_path.open(
            "w", encoding="utf-8"
        ) as target:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Line {line_number}: expected a JSON object")
                required = ("task", "response_id", "source_id", "ranked_claims")
                missing = [k for k in required if k not in record]
                if missing:
                    raise ValueError(f"Line {line_number}: missing keys {missing}")
                ranked_claims = record["ranked_claims"]
                if not isinstance(ranked_claims, list):
                    raise ValueError(f"Line {line_number}: ranked_claims must be a list")

                responses += 1
                if not ranked_claims:
                    empty_responses += 1

                for item_index, item in enumerate(ranked_claims, start=1):
                    if not isinstance(item, dict) or not all(
                        k in item for k in ("baseline_span", "claim", "ranked_evidence")
                    ):
                        raise ValueError(
                            f"Line {line_number}, claim {item_index}: missing span/claim/rankings"
                        )
                    evidence_by_method = item["ranked_evidence"]
                    if not isinstance(evidence_by_method, dict) or "v2" not in evidence_by_method:
                        raise ValueError(
                            f"Line {line_number}, claim {item_index}: missing v2 evidence"
                        )
                    evidence = evidence_by_method["v2"]
                    if not isinstance(evidence, list):
                        raise ValueError(
                            f"Line {line_number}, claim {item_index}: v2 must be a list"
                        )
                    span = item["baseline_span"]
                    if not isinstance(span, dict) or not all(
                        k in span for k in ("start", "end", "text", "confidence")
                    ):
                        raise ValueError(
                            f"Line {line_number}, claim {item_index}: invalid baseline_span"
                        )
                    if not isinstance(item["claim"], str):
                        raise ValueError(
                            f"Line {line_number}, claim {item_index}: claim must be text"
                        )

                    output = {
                        "task": record["task"],
                        "response_id": record["response_id"],
                        "source_id": record["source_id"],
                        "baseline_span": span,
                        "claim": item["claim"],
                        "ranked_evidence": evidence,
                    }
                    target.write(json.dumps(output, ensure_ascii=False, allow_nan=False) + "\n")
                    claims += 1
                    by_task[str(record["task"])] += 1

        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Processed responses: {responses}")
    print(f"Responses without predicted claims (not exported): {empty_responses}")
    print(f"Exported predicted claim instances: {claims}")
    for task, count in sorted(by_task.items()):
        print(f"  {task}: {count}")
    print("Export contains v2 evidence only; no gold labels or v1 rankings.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
