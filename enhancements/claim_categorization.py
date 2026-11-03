
import json


def load_rank_one_records(file_path):
    """Read claims and their Rank 1 evidence from V2 JSONL."""

    records = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            item = json.loads(line)

            claim = item["claim"]
            ranked_evidence = item["ranked_evidence"]

            # Find the passage explicitly marked Rank 1.
            rank_one = next(
                (p for p in ranked_evidence if p["rank"] == 1),
                None
            )

            if rank_one is None:
                raise ValueError(
                    f"Rank 1 missing on line {line_number}"
                )

            records.append({
                "task": item["task"],
                "response_id": item["response_id"],
                "source_id": item["source_id"],
                "baseline_span": item["baseline_span"],
                "claim": claim,
                "evidence": rank_one["evidence"],
                "evidence_rank": 1,
                "ranking_score": rank_one["score"]
            })

    return records


if __name__ == "__main__":

    input_path = "data/verification/categorization_input_v2.jsonl"

    records = load_rank_one_records(input_path)

    print("Total claim records loaded:", len(records))

    for record in records[:2]:
        print("\n--- Claim ---")
        print(record["claim"])
        print("\n--- Rank 1 Evidence ---")
        print(record["evidence"])
        print("\nRanking Score:", record["ranking_score"])
