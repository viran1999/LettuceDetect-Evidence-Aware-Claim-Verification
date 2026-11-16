
from enhancements.ollama_claim_decomposer import decompose_claim
from enhancements.claim_categorization import categorize_atomic_claims


def verify_atomic_claims(claim, evidence, nli_model):
    """
    Verify a claim against supplied evidence.

    Steps:
      1. Decompose the original claim.
      2. Run NLI for each atomic claim.
      3. Combine predictions into a provisional category.

    The NLI model is passed in so it can be reused.
    """

    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("Evidence must be a non-empty string")

    # Step 1: Automatic claim decomposition
    atomic_claims = decompose_claim(claim)

    # Step 2: Prepare evidence-claim pairs
    pairs = [
        (evidence, atomic_claim)
        for atomic_claim in atomic_claims
    ]

    scores = nli_model.predict(
        pairs,
        apply_softmax=True
    )

    labels = {
        int(k): str(v).lower()
        for k, v in nli_model.model.config.id2label.items()
    }

    predictions = []
    atomic_results = []

    for atomic_claim, probabilities in zip(
        atomic_claims, scores
    ):
        predicted_id = int(probabilities.argmax())
        relation = labels[predicted_id]

        predictions.append(relation)

        atomic_results.append({
            "atomic_claim": atomic_claim,
            "nli_relation": relation,
            "nli_scores": {
                labels[i]: round(float(probability), 4)
                for i, probability in enumerate(probabilities)
            }
        })

    # Step 3: Calculate the overall category
    decision = categorize_atomic_claims(predictions)

    return {
        "category": decision["category"],
        "status": decision["status"],
        "supported_count": decision["supported_count"],
        "contradicted_count": decision["contradicted_count"],
        "neutral_count": decision["neutral_count"],
        "total_atomic_claims": decision["total_atomic_claims"],
        "atomic_verifications": atomic_results
    }
