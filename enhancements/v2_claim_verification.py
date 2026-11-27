
from sentence_transformers import CrossEncoder
from copy import deepcopy
from enhancements.structured_evidence import normalize_structured_evidence
from enhancements.atomic_claim_verifier import verify_atomic_claims
from enhancements.span_decision import decide_span_category
from enhancements.evidence_relevance_guard import assess_evidence_relevance
from enhancements.span_verification_context import (
    prepare_span_verification_context
)


NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"


def add_claim_verification(result, source_info=None):
    """
    Add atomic claim verification to Evidence Ranking V2 output.

    Input:
        Dictionary returned by analyze_record().

    Output:
        Same dictionary with verification results added
        to each claim in claims_with_evidence.
    """

    # Keep the existing v2/frontend contract. Enable the enhanced internal
    # verifier explicitly; omit this variable to use the original v2 behavior.
    import os
    if os.getenv("LETTUCE_VERIFIER", "enhanced").strip().lower() == "enhanced":
        from enhancements.v2_enhanced_adapter import add_enhanced_verification_for_frontend
        return add_enhanced_verification_for_frontend(result, source_info=source_info)

    reviews = result.get("claims_with_evidence", [])

    if not reviews:
        print("No suspicious claims found for verification.")
        return result

    print("\nLoading claim verification NLI model...")

    nli_model = CrossEncoder(
        NLI_MODEL,
        device="cpu"
    )

    # Cache verification results for identical claim-evidence pairs.
    verification_cache = {}

    for index, review in enumerate(reviews, start=1):

        claim = review["claim"]

        ranked_evidence = review["ranked_evidence"]

        rank_one = next(
            (
                item for item in ranked_evidence
                if item["rank"] == 1
            ),
            None
        )

        if rank_one is None or not rank_one.get("evidence"):
            review["verification_status"] = "NO_RANK_1_EVIDENCE"
            continue

        # Preserve the original ranked evidence.
        original_evidence = rank_one["evidence"]

        # Normalize structured evidence only for Data2Txt.
        task = str(result.get("task", "")).casefold()

        if task == "data2txt":
            evidence_for_nli = normalize_structured_evidence(
                original_evidence
            )
        else:
            evidence_for_nli = original_evidence

        print(f"\nVerifying claim {index}/{len(reviews)}")


        cache_key = (claim, evidence_for_nli)

        try:
            if cache_key in verification_cache:
                verification = deepcopy(verification_cache[cache_key])
                print(f"Reusing cached verification for claim {index}")

            else:
                verification = verify_atomic_claims(
                    claim=claim,
                    evidence=evidence_for_nli,
                    nli_model=nli_model
                )

                verification_cache[cache_key] = deepcopy(verification)

        except Exception as exc:
            print(
                f"Verification failed for claim {index}: "
                f"{type(exc).__name__}: {exc}"
            )

            review["verification_status"] = "VERIFICATION_ERROR"

            review["verification"] = {
                "category": None,
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:300],
                "evidence_rank": 1,
                "scope": "rank_1_only"
            }

            # Continue with the next claim.
            continue

        verification["evidence_rank"] = 1
        verification["evidence"] = rank_one["evidence"]
        verification["evidence_used_for_nli"] = evidence_for_nli
        verification["ranking_score"] = rank_one.get("score")
        verification["scope"] = "rank_1_only"

        # Preserve the existing ranking output.
        # Add our categorization alongside it.
        review["verification"] = verification

        # Prepare span-aware context using atomic claims already
        # produced by Qwen. No additional model calls required.
        if source_info is not None:
            span = review.get("baseline_span", {})

            span_text = (
                span.get("text")
                if isinstance(span, dict)
                else None
            )

            atomic_claims = [
                item["atomic_claim"]
                for item in verification.get("atomic_verifications", [])
                if (
                    isinstance(item, dict)
                    and isinstance(item.get("atomic_claim"), str)
                    and item["atomic_claim"].strip()
                )
            ]

            if span_text and atomic_claims:
                review["span_verification_context"] = (
                    prepare_span_verification_context(
                        suspicious_span=span_text,
                        atomic_claims=atomic_claims,
                        source_info=source_info,
                        task=result.get("task", "")
                    )
                )

        # Check whether Rank 1 evidence is relevant to the
        # suspicious span's structured attribute.
        span_context = review.get("span_verification_context")

        if span_context is not None:
            relevance = assess_evidence_relevance(
                span_context,
                original_evidence
            )

            verification["evidence_relevance"] = relevance["status"]

            if relevance["status"] == "irrelevant":
                # Preserve the original NLI prediction for debugging.
                verification["raw_nli_category"] = verification["category"]

                # Do not make a definitive decision using unrelated evidence.
                verification["category"] = None
                verification["status"] = "needs_review"
                verification["decision_reason"] = (
                    "rank_1_evidence_irrelevant_to_suspicious_span"
                )

        # Reuse existing atomic NLI predictions without additional model calls.
        if isinstance(span_context, dict):

            # Preserve the whole-sentence category for comparison.
            verification["whole_claim_category"] = verification.get(
                "raw_nli_category",
                verification.get("category")
            )

            span_decision = decide_span_category(
                verification,
                span_context
            )

            verification["category"] = span_decision["category"]
            verification["status"] = span_decision["status"]
            verification["decision_reason"] = span_decision["decision_reason"]

            if "selected_nli_relation" in span_decision:
                verification["selected_nli_relation"] = (
                    span_decision["selected_nli_relation"]
                )

        review["verification_status"] = (
            verification["category"]
            if verification["category"] is not None
            else "needs_review"
        )

        print("Category:", review["verification_status"])

    result["verification_method"] = "atomic_nli_rank1"

    return result
