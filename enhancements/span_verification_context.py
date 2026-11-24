
from enhancements.span_claim_selector import select_span_claim
from enhancements.structured_source_lookup import lookup_span_attribute


def prepare_span_verification_context(
    suspicious_span,
    atomic_claims,
    source_info,
    task
):
    """
    Identify the atomic claim related to a suspicious span
    and inspect the original structured source when applicable.

    This function does not assign a verification category.
    It does not run any AI models or evidence ranking.
    """

    selection = select_span_claim(
        suspicious_span,
        atomic_claims
    )

    context = {
        "span_match_status": selection["status"],
        "selected_atomic_claim": selection["claim"],
        "matching_claims": selection["matches"],
        "attribute_lookup": None
    }

    # Do not look up structured attributes if the atomic
    # claim cannot be selected unambiguously.
    if selection["status"] != "matched":
        return context

    # Structured attribute lookup applies only to Data2Txt.
    if str(task).casefold() == "data2txt":
        context["attribute_lookup"] = lookup_span_attribute(
            source_info,
            suspicious_span
        )

    return context
