import json
from pathlib import Path

from sentence_transformers import CrossEncoder
from enhancements.claim_categorization import load_rank_one_records


# Translate NLI relations into our project's categories.
CATEGORY_MAP = {
    "entailment": "Supported",
    "contradiction": "Contradicted",
    "neutral": "Unverifiable",
}


def main():
    input_path = "data/verification/categorization_input_v2.jsonl"
    output_path = Path("results/categorization/rank1_predictions.jsonl")

    print("Loading V2 claim records...")
    records = load_rank_one_records(input_path)

    print("Loading local NLI model...")
    model = CrossEncoder(
        "cross-encoder/nli-MiniLM2-L6-H768",
        device="cpu"
    )

    # Read the actual model label mapping.
    id2label = {
        int(k): str(v).lower()
        for k, v in model.model.config.id2label.items()
    }

    # Start with only two claims.
    selected = records[:2]

    pairs = [
        (record["evidence"], record["claim"])
        for record in selected
    ]

    scores = model.predict(pairs, apply_softmax=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for record, probabilities in zip(selected, scores):
            predicted_id = int(probabilities.argmax())
            nli_label = id2label[predicted_id]

            if nli_label not in CATEGORY_MAP:
                raise ValueError(f"Unknown NLI label: {nli_label}")

            category = CATEGORY_MAP[nli_label]

            result = {
                "task": record["task"],
                "response_id": record["response_id"],
                "source_id": record["source_id"],
                "baseline_span": record["baseline_span"],
                "claim": record["claim"],
                "verification": {
                    "category": category,
                    "evidence_rank": 1,
                    "evidence": record["evidence"],
                    "ranking_score": record["ranking_score"],
                    "nli_relation": nli_label,
                    "nli_scores": {
                        id2label[i]: round(float(p), 4)
                        for i, p in enumerate(probabilities)
                    },
                    "status": "provisional",
                    "scope": "rank_1_only"
                }
            }

            output_file.write(json.dumps(result) + "\n")

            print("\n" + "=" * 50)
            print("CLAIM:", record["claim"])
            print("CATEGORY:", category)
            print("NLI RELATION:", nli_label)

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
