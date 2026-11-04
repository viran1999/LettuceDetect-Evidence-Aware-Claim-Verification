
import json
import os
from pathlib import Path

from sentence_transformers import CrossEncoder

from enhancements.claim_categorization import load_rank_one_records
from enhancements.atomic_claim_verifier import verify_atomic_claims
from enhancements.ollama_claim_decomposer import DEFAULT_MODEL


INPUT_PATH = "data/verification/categorization_input_v2.jsonl"

OUTPUT_PATH = Path(
    "results/categorization/atomic_rank1_predictions.jsonl"
)

NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"


def main():

    print("Loading real V2 claim records...")

    records = load_rank_one_records(INPUT_PATH)

    print("Total records available:", len(records))

    # Test only two records initially
    # selected_records = records
    selected_records = records[:5]

    print("Loading MiniLM NLI model...")

    nli_model = CrossEncoder(
        NLI_MODEL,
        device="cpu"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open(
        "w", encoding="utf-8"
    ) as output_file:

        for index, record in enumerate(
            selected_records, start=1
        ):

            print(f"\nProcessing claim {index}...")

            # Run Qwen decomposition + MiniLM verification
            verification = verify_atomic_claims(
                claim=record["claim"],
                evidence=record["evidence"],
                nli_model=nli_model
            )

            # Add traceability information
            verification["scope"] = "rank_1_only"
            verification["evidence_rank"] = 1
            verification["evidence"] = record["evidence"]
            verification["ranking_score"] = record["ranking_score"]
            verification["decomposition_model"] = (
                os.getenv("DECOMPOSITION_MODEL") or DEFAULT_MODEL
            )
            verification["nli_model"] = NLI_MODEL

            result = {
                "task": record["task"],
                "response_id": record["response_id"],
                "source_id": record["source_id"],
                "baseline_span": record["baseline_span"],
                "claim": record["claim"],
                "verification": verification
            }

            # Save one JSON object per line
            output_file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

            print("Category:", verification["category"])
            print("Status:", verification["status"])
            print(
                "Atomic claims:",
                verification["total_atomic_claims"]
            )

    print("\nProcessing completed.")
    print("Results saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
