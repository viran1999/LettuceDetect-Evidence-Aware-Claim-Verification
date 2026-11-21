
from sentence_transformers import CrossEncoder
from copy import deepcopy
from enhancements.structured_evidence import normalize_structured_evidence

from enhancements.atomic_claim_verifier import verify_atomic_claims


NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"


def add_claim_verification(result):
    """
    Add atomic claim verification to Evidence Ranking V2 output.

    Input:
        Dictionary returned by analyze_record().

    Output:
        Same dictionary with verification results added
        to each claim in claims_with_evidence.
    """

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

        review["verification_status"] = (
            verification["category"]
            if verification["category"] is not None
            else "needs_review"
        )

        print("Category:", review["verification_status"])

    result["verification_method"] = "atomic_nli_rank1"

    return result
