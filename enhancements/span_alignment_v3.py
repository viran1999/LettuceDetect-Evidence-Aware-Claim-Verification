"""Conservative multi-claim alignment of LettuceDetect spans to atomic facts.

Only candidate alignment; never asserts that claims are true. If coverage or
provenance is weak, abstains rather than manufacturing verification results.
"""
from __future__ import annotations

import re
import unicodedata

STOP = set('a an the is are was were be been being to of at on for in from by with as this that these those it its they their has have had and or but which who what there here can could would will should do does did some many more most also such very including about while'.split())
WORDS = re.compile(r'[a-z]+|\d+(?:\.\d+)?')
NEG = {'no', 'not', 'never', 'without', 'neither', 'none'}
NUMBER = re.compile(r'\d+(?:\.\d+)?')


def content_tokens(text):
    if not isinstance(text, str):
        return set()
    value = unicodedata.normalize('NFKC', text).casefold()
    value = re.sub(r'\bwi[\s-]?fi\b', 'wifi', value)
    value = re.sub(r'\bdoesn.t\b|\bdidn.t\b|\bisn.t\b|\bwasn.t\b', 'not', value)
    return {tok for tok in WORDS.findall(value) if tok not in STOP}


def recover_span(answer, span):
    """Expand character-clipped words only; do not expand to unrelated clauses."""
    if not isinstance(answer, str) or not isinstance(span, dict):
        return str(span.get('text', '') if isinstance(span, dict) else '').strip()
    start, end = span.get('start'), span.get('end')
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(answer):
        return str(span.get('text', '')).strip()
    while start and answer[start - 1].isalnum() and answer[start].isalnum():
        start -= 1
    while end < len(answer) and answer[end - 1].isalnum() and answer[end].isalnum():
        end += 1
    return answer[start:end].strip()


def choose_atomic_claims(suspicious_span, atomic_claims, original_claim):
    """Select a minimally covering set of existing atomic claims.

    Multiple selected facts are allowed. Claims are candidates ONLY; checking
    support/contradiction is a separate evidence stage.
    """
    target = content_tokens(suspicious_span)
    if not target:
        return {'status': 'needs_review', 'reason': 'empty_informative_span',
                'claims': [], 'coverage': 0.0}
    parent_numbers = set(NUMBER.findall(original_claim))
    target_nums = {t for t in target if NUMBER.fullmatch(t)}
    parent_neg = content_tokens(original_claim) & NEG
    ranked = []
    seen = set()
    for raw in atomic_claims:
        if not isinstance(raw, str) or not raw.strip():
            continue
        claim = ' '.join(raw.split())
        key = claim.casefold()
        if key in seen:
            continue
        seen.add(key)
        tokens = content_tokens(claim)
        shared = tokens & target
        claim_nums = {t for t in tokens if NUMBER.fullmatch(t)}
        if not shared:
            continue
        # Do not accept an invented numeric assertion or an incompatible number.
        if claim_nums - parent_numbers:
            continue
        if target_nums and not target_nums.issubset(claim_nums | (target_nums - parent_numbers)):
            continue
        # For a short negative span, an affirmative paraphrase is not a safe match.
        if target & NEG and not (tokens & NEG):
            continue
        if len(target) > 2 and len(shared) < 2:
            continue
        ranked.append((claim, tokens, shared))
    if not ranked:
        return {'status': 'needs_review', 'reason': 'no_grounded_candidate',
                'claims': [], 'coverage': 0.0}

    selected, remaining = [], set(target)
    available = list(ranked)
    while remaining and available:
        available.sort(key=lambda item: (
            -len(item[2] & remaining),
            -(len(item[2] & remaining) / max(1, len(item[1]))),
            len(item[1]), item[0]
        ))
        best = available.pop(0)
        gain = best[2] & remaining
        if not gain:
            break
        selected.append(best[0])
        remaining -= gain
    coverage = round(1.0 - len(remaining) / len(target), 4)
    # 1- or 2-token spans need complete coverage to avoid accidental matches.
    minimum = 1.0 if len(target) <= 2 else 0.8
    if coverage < minimum:
        return {'status': 'needs_review', 'reason': 'insufficient_span_coverage',
                'claims': [], 'coverage': coverage}

    # Don't silently resolve competing interpretations that explain exactly the
    # same target words (rather than distinct facts in a compound span).
    if len(selected) == 1:
        overlap = content_tokens(selected[0]) & target
        rivals = [c for c, _, shared in ranked if c != selected[0] and shared == overlap]
        if rivals:
            return {'status': 'needs_review', 'reason': 'competing_atomic_interpretations',
                    'claims': [], 'coverage': coverage}
    return {'status': 'matched', 'reason': 'existing_atoms_covered_span',
            'claims': selected, 'coverage': coverage}


def align_span(answer, span, parent_claim, atomic_claims, targeted_fallback=None):
    """Align using existing atoms, optionally recover omitted facts via Qwen.

    targeted_fallback(original_claim, normalized_span) returns an audited dict
    with proposed_claims + flags. Flagged proposals are never automatically used.
    """
    normalized_span = recover_span(answer, span)
    result = choose_atomic_claims(normalized_span, atomic_claims, parent_claim)
    result.update({'normalized_span': normalized_span, 'method': 'existing_atomic_claims'})
    if result['status'] == 'matched' or targeted_fallback is None:
        return result

    try:
        fallback = targeted_fallback(parent_claim, normalized_span)
    except Exception as exc:
        result.update({'fallback_error': f'{type(exc).__name__}: {exc}'[:200]})
        return result
    proposal = fallback.get('proposed_claims', []) if isinstance(fallback, dict) else []
    flags = fallback.get('flags', []) if isinstance(fallback, dict) else []
    # Filter unrelated proposals by alignment and do not allow ungrounded facts.
    refined = choose_atomic_claims(normalized_span, proposal, parent_claim)
    if flags:
        # Whole-output flags may concern extraneous discarded proposals. Retain
        # them in audit, but do not silently auto-accept this batch.
        result.update({'fallback_flags': flags, 'fallback_proposals': proposal})
        return result
    if refined['status'] == 'matched':
        refined.update({'normalized_span': normalized_span,
                        'method': 'targeted_qwen_validated', 'fallback_proposals': proposal})
        return refined
    result.update({'fallback_proposals': proposal})
    return result
