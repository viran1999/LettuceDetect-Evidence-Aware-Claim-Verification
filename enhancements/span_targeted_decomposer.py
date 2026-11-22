"""Experimental, span-targeted decomposition for offline review only.

Never treats generated claims as proven, and never replaces detector spans,
rankings, existing verification labels, or gold annotations.
"""
from __future__ import annotations

import json
import re
import urllib.request

OLLAMA_URL = 'http://localhost:11434/api/chat'
MODEL = 'qwen2.5:0.5b-instruct'

# Numeric-precision guard: an invented numeric value must not be silently accepted.
NUMERIC_RE = re.compile(r'(?<![\w])\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?(?=[A-Za-z])')
WORD_RE = re.compile(r'[a-z]+|\d+(?:\.\d+)?')
STOPWORDS = set('a an and the to of for in on from by at with it is are was were do does did that this there as or but some they their who which how what'.split())


def meaningful_tokens(value: str) -> set[str]:
    normal = re.sub(r'\bwi[\s-]?fi\b', 'wifi', value.casefold())
    return {x for x in WORD_RE.findall(normal) if x not in STOPWORDS}


def numeric_values(value: str) -> set[str]:
    # Fractions/dates/times retain their component numbers; conservative guard.
    return set(re.findall(r'\d+(?:\.\d+)?', value.replace(',', '')))


def assess_generated_claims(original_claim: str, suspicious_span: str, proposed: list[str]) -> dict:
    """Inspect proposals; labels are screening flags, never accuracy certificates."""
    if not isinstance(proposed, list) or any(not isinstance(x, str) for x in proposed):
        raise ValueError('Expected target_atomic_claims to be a list of strings')
    cleaned = []
    seen = set()
    for item in proposed:
        val = ' '.join(item.split()).strip()
        if val and val.casefold() not in seen:
            cleaned.append(val)
            seen.add(val.casefold())
    original_nums = numeric_values(original_claim)
    span_terms = meaningful_tokens(suspicious_span)
    flags = []
    for claim in cleaned:
        extra = sorted(numeric_values(claim) - original_nums)
        if extra:
            flags.append({'claim': claim, 'flag': 'new_number_not_in_source', 'values': extra})
        overlap = sorted(span_terms & meaningful_tokens(claim))
        if span_terms and not overlap:
            flags.append({'claim': claim, 'flag': 'no_lexical_overlap_with_span', 'values': []})
    if not cleaned:
        flags.append({'claim': '', 'flag': 'no_targeted_claims_returned', 'values': []})
    return {'proposed_claims': cleaned, 'flags': flags, 'status': 'requires_human_review'}


def targeted_decompose(original_claim: str, suspicious_span: str, *, model: str = MODEL,
                       timeout: int = 300, url: str = OLLAMA_URL) -> dict:
    """Ask local Qwen for *candidate* atomic facts related to a marked span.

    Do not call this in production classification until measured on labeled data.
    """
    if not isinstance(original_claim, str) or not original_claim.strip():
        raise ValueError('Original claim missing')
    if not isinstance(suspicious_span, str) or not suspicious_span.strip():
        raise ValueError('Suspicious span missing')

    prompt = (
        'You are an evidence-free claim decomposition assistant. Your job is NOT to decide truth.\n'
        'Read the original generated answer sentence and the suspicious text span.\n'
        'Extract the independent factual claims in the ORIGINAL SENTENCE that are expressed by, '
        'or necessary to understand, the suspicious span.\n'
        'RULES:\n'
        '1. A single suspicious span may cover multiple facts: return each distinct factual claim.\n'
        '2. Do not include unrelated facts from elsewhere in the sentence.\n'
        '3. Preserve exact names, numerical values, comparisons, negation, and conditions.\n'
        '4. Never add a fact that is not in the original sentence.\n'
        '5. If the span is just punctuation, a meaningless fragment, or cannot be resolved, return [].\n'
        '6. If the suspicious part concerns a number that is missing from a previous decomposition, keep that number.\n'
        '7. Output only JSON: {"target_atomic_claims": ["sentence", ...]}.\n'
        '\nEXAMPLE:\n'
        'Original: The hotel has no pool or spa, but offers free breakfast.\n'
        'Suspicious span: no pool or spa\n'
        'Output: {"target_atomic_claims": ["The hotel has no pool.", "The hotel has no spa."]}\n'
        '\nEXAMPLE:\n'
        'Original: Most dishes cost $15 to $30, and the waiters are friendly.\n'
        'Suspicious span: $15 to $30\n'
        'Output: {"target_atomic_claims": ["Most dishes cost $15 to $30."]}\n'
        f'\nOriginal: {original_claim}\n'
        f'Suspicious span: {suspicious_span}\n'
        'Output: '
    )
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'format': {'type': 'object', 'properties': {
            'target_atomic_claims': {'type': 'array', 'items': {'type': 'string'}}},
            'required': ['target_atomic_claims']},
        'options': {'temperature': 0},
        'keep_alive': '10m',
    }
    request = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'),
                                     headers={'Content-Type': 'application/json'},method='POST')
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    content = result['message']['content']
    parsed = json.loads(content)
    if not isinstance(parsed, dict) or not isinstance(parsed.get('target_atomic_claims'), list):
        raise ValueError('Invalid Ollama JSON shape')
    checked = assess_generated_claims(original_claim, suspicious_span, parsed['target_atomic_claims'])
    return checked
