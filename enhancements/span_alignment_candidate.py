"""Experimental span-to-atomic-claim candidate generation (NOT final verification).

Conservative lexical candidate retrieval for offline evaluation. Does not assign
Supported/Contradicted/etc., and does not alter existing pipeline behavior.
"""

from __future__ import annotations

import re
import unicodedata

_STOP = set("""a an the to of at in on for with from by and or is are was were be
been being as it this that these those they their its our his her who which
where there here can could should would will have has had do does did also
some what when how but if than then very into about out over under during
including according""".split())
_NEGATIONS = {"no", "not", "never", "without", "neither", "none"}
_NUM = re.compile(r"\d+(?:\.\d+)?")
_WORD = re.compile(r"[a-z]+|\d+(?:\.\d+)?")


def _tokens(text: str) -> set[str]:
    clean = unicodedata.normalize("NFKC", text).casefold()
    clean = re.sub(r"\bwi[\s-]?fi\b", "wifi", clean)
    return {t for t in _WORD.findall(clean) if t not in _STOP}


def propose_alignment(suspicious_span: str, atomic_claims: list[str]) -> dict:
    """Return inspectable candidates, not an automatic verification decision.

    `candidate_single` / `candidate_multi` are suggestions requiring validation.
    `unresolved` means existing claims do not provide a trustworthy mapping.
    Exact duplicates are deduplicated, preserving the first original wording.
    """
    if not isinstance(suspicious_span, str) or not isinstance(atomic_claims, list):
        raise TypeError("Expected span string and atomic-claim list")

    target = _tokens(suspicious_span)
    if not target:
        return {"status": "unresolved", "reason": "no_informative_span_tokens", "candidates": []}

    candidates = []
    seen = set()
    for raw in atomic_claims:
        if not isinstance(raw, str) or not raw.strip():
            continue
        claim = raw.strip()
        dedup_key = " ".join(claim.casefold().split())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        terms = _tokens(claim)
        common = terms & target
        if not terms or not common:
            continue

        if len(target) == 1:
            # One-word spans are often too vague to resolve from rewritten claims.
            if len(terms) > 7:
                continue
        elif len(common) < 2:
            continue

        atom_coverage = len(common) / len(terms)
        if len(target) > 1 and atom_coverage < 0.5 and len(common) < 3:
            continue

        # Guard against evidence-alignment errors from negation and numbers.
        if len(target) <= 6 and bool(target & _NEGATIONS) != bool(terms & _NEGATIONS):
            continue
        span_numbers = {t for t in target if _NUM.fullmatch(t)}
        claim_numbers = {t for t in terms if _NUM.fullmatch(t)}
        if span_numbers and claim_numbers and not (span_numbers & claim_numbers):
            continue
        if span_numbers and not claim_numbers and len(target) <= 6:
            continue

        candidates.append({
            "claim": claim,
            "shared_tokens": sorted(common),
            "claim_token_coverage": round(atom_coverage, 4),
            "span_token_coverage": round(len(common) / len(target), 4),
        })

    candidates.sort(key=lambda c: (
        -c["claim_token_coverage"], -c["span_token_coverage"], c["claim"]
    ))
    if not candidates:
        return {"status": "unresolved", "reason": "no_plausible_atomic_claim", "candidates": []}

    if len(target) == 1:
        if len(candidates) != 1:
            return {"status": "unresolved", "reason": "single_token_ambiguous", "candidates": candidates}
        return {"status": "candidate_single", "reason": "single_token_unique", "candidates": candidates}

    best_coverage = candidates[0]["claim_token_coverage"]
    selected = [
        c for c in candidates
        if c["claim_token_coverage"] >= 0.65
        and c["claim_token_coverage"] >= best_coverage - 0.25
    ]
    if not selected:
        selected = candidates[:1]
    return {
        "status": "candidate_multi" if len(selected) > 1 else "candidate_single",
        "reason": "lexical_candidate_generation",
        "candidates": selected,
    }
