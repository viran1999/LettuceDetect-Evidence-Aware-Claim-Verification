
import re


def get_tokens(text):
    """Extract lowercase words and numbers."""
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def select_span_claim(suspicious_span, atomic_claims):
    """
    Select the atomic claim corresponding to a
    LettuceDetect suspicious span.

    Returns:
        status: matched, ambiguous, or no_match
        claim: selected claim, or None
        matches: all matching atomic claims
    """

    span_tokens = get_tokens(suspicious_span)

    if not span_tokens:
        return {
            "status": "no_match",
            "claim": None,
            "matches": []
        }

    matches = []
    seen = set()

    for claim in atomic_claims:
        claim_tokens = get_tokens(claim)

        if span_tokens.issubset(claim_tokens):
            key = claim.strip().casefold()

            if key not in seen:
                matches.append(claim)
                seen.add(key)

    if len(matches) == 1:
        status = "matched"
        selected = matches[0]

    elif len(matches) > 1:
        status = "ambiguous"
        selected = None

    else:
        status = "no_match"
        selected = None

    return {
        "status": status,
        "claim": selected,
        "matches": matches
    }
