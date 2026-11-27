
RELATION_TO_CATEGORY = {
    "entailment": "Supported",
    "contradiction": "Contradicted",
    "neutral": "Unverifiable",
}


def decide_span_category(verification, span_context):
    """
    Determine a provisional category for the atomic claim
    corresponding to LettuceDetect's suspicious span.

    Does not execute any AI models.
    """

    def needs_review(reason):
        return {
            "category": None,
            "status": "needs_review",
            "decision_reason": reason,
        }

    if not isinstance(span_context, dict):
        return needs_review("missing_span_context")

    if span_context.get("span_match_status") != "matched":
        return needs_review("atomic_claim_match_not_unique")

    selected = span_context.get("selected_atomic_claim")

    if not isinstance(selected, str) or not selected.strip():
        return needs_review("missing_selected_claim")

    matches = [
        item
        for item in verification.get("atomic_verifications", [])
        if (
            isinstance(item, dict)
            and isinstance(item.get("atomic_claim"), str)
            and item["atomic_claim"].strip().casefold()
            == selected.strip().casefold()
        )
    ]

    if len(matches) != 1:
        return needs_review("selected_atomic_result_not_unique")

    if verification.get("evidence_relevance") == "irrelevant":
        return needs_review("irrelevant_rank_1_evidence")

    relation = str(matches[0].get("nli_relation", "")).lower()
    category = RELATION_TO_CATEGORY.get(relation)

    if category is None:
        return needs_review("unknown_nli_relation")

    return {
        "category": category,
        "status": "provisional",
        "decision_reason": "selected_atomic_claim_nli",
        "selected_nli_relation": relation,
    }
