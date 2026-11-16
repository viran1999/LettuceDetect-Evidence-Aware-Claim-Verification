"""Hybrid Evidence Ranking v2 — reusable research ranking module.

Use rank_claim(claim, make_passages(source_text, task), encoder, top_k=3).
No gold annotations or verification categories are read during ranking.
Requires sentence-transformers, scikit-learn and numpy.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FIELD_PATTERN = re.compile(r"(?<!\w)([A-Za-z][A-Za-z0-9_]{1,35})\s*:\s*")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\d])")
YEAR_PATTERN = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")
NUMBER_PATTERN = re.compile(r"(?<!\w)\d+(?:\.\d+)?(?!\w)")
TOKENS = re.compile(r"[a-z]+|\d+(?:\.\d+)?")


def tokens(s):
    return set(TOKENS.findall(s.lower()))


def normalized(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def evidence_to_text(v):
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return "\n".join(f"{k}: {evidence_to_text(val)}" for k, val in v.items())
    if isinstance(v, list):
        return "\n".join(evidence_to_text(x) for x in v)
    return "" if v is None else str(v)


def structured_passages(text):
    """Extract key-value records, including flattened forms like 'WiFi: no RestaurantsTakeOut: True'."""
    results = []
    matches = list(FIELD_PATTERN.finditer(text))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        value = normalized(text[match.end():end])
        # Limit very long free-text fields; those are handled by ordinary chunking.
        if value and value.lower() not in {"none", "null", "unknown", "n/a", "not available"} and len(value) <= 160:
            results.append(f"{match.group(1)}: {value}")
    return results


def make_passages(text, task):
    """Produce non-overlapping source chunks + structured key/value statements."""
    text = normalized(text)
    if task.lower() == "qa":
        # RAGTruth sometimes embeds the question in the source as metadata.
        # It is not a valid evidence passage.
        text = re.sub(r"(?is)^question\s*:\s*.*?\bpassages\s*:\s*", "", text, count=1)
    chunks = []
    if task.lower() in {"data2txt", "data2text"}:
        chunks.extend(structured_passages(text))
    # QA passage labels can provide natural boundaries; avoid overlapping windows.
    if task == "QA" and re.search(r"\bpassage\s+\d+\s*:", text, re.I):
        raw = re.split(r"(?i)\bpassage\s+\d+\s*:", text)
        chunks.extend(normalized(p) for p in raw if normalized(p))
    else:
        sents = [normalized(x) for x in SENTENCE_SPLIT.split(text) if normalized(x)]
        # Non-overlapping pairs: unlike v1's sliding pairs, this avoids duplicate windows.
        chunks.extend(" ".join(sents[i:i + 2]) for i in range(0, len(sents), 2))
    # Remove exact duplicates and chunks containing the same content with trivial spacing.
    unique, seen = [], set()
    for passage in chunks:
        key = re.sub(r"\W+", "", passage.lower())
        if key and key not in seen and not re.match(r"(?i)^(?:question|query|passages)\s*:", passage):
            seen.add(key)
            unique.append(passage)
    return unique


def minmax(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return a
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def evidence_signal(claim, passage):
    """Heuristic relevance boost, NOT a support/contradiction classifier."""
    c, p = claim.lower(), passage.lower()
    ct, pt = tokens(c), tokens(p)
    overlap = ct & pt
    score = 0.0
    # Match a named structured field even when the field's value conflicts.
    key_value = FIELD_PATTERN.match(passage)
    if key_value:
        key = key_value.group(1).lower()
        aliases = {
            "wifi": ("wifi", "wi-fi", "wi fi", "internet"),
            "businessparking": ("parking",),
            "outdoorseating": ("outdoor seating", "outside seating", "patio"),
            "restaurantsreservations": ("reservation", "booking"),
            "restaurantstakeout": ("takeout", "take-out", "takeaway"),
            "music": ("music",),
            "intimate": ("intimate",),
        }
        terms = aliases.get(key, (key,))
        if any(term in c for term in terms):
            score += 1.0
    # If a claim has a year, a passage discussing dates and the same subject is useful.
    if YEAR_PATTERN.search(c) and YEAR_PATTERN.search(p) and len(overlap) >= 2:
        score += 0.6
    # Numerical claim and a numerical candidate with thematic word overlap.
    if NUMBER_PATTERN.search(c) and NUMBER_PATTERN.search(p) and len(overlap) >= 2:
        score += 0.3
    # Prioritize passages containing specific relevant terminology.
    score += min(len(overlap), 6) / 12.0
    return score


def rank_claim(claim, passages, model, top_k=3):
    if not passages:
        return []
    try:
        tf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform([claim] + passages)
        tf_scores = cosine_similarity(tf[:1], tf[1:]).flatten()
    except ValueError:
        tf_scores = np.zeros(len(passages))
    emb = model.encode([claim] + passages, normalize_embeddings=True, show_progress_bar=False)
    sem_scores = np.asarray(emb[1:]) @ np.asarray(emb[0])
    sig = np.array([evidence_signal(claim, p) for p in passages])
    # Within-document normalization; scores across methods are NOT directly comparable.
    combined = .40 * minmax(sem_scores) + .30 * minmax(tf_scores) + .30 * minmax(sig)
    indices = sorted(range(len(passages)), key=lambda i: (-combined[i], i))
    picked = []
    for idx in indices:
        # Suppress nearly identical alternatives in the top three.
        t = tokens(passages[idx])
        if any(len(t & tokens(passages[j])) / max(1, len(t | tokens(passages[j]))) > .80 for j in picked):
            continue
        picked.append(idx)
        if len(picked) >= top_k:
            break
    return [(i, round(float(combined[i]), 4), passages[i]) for i in picked]


def rank_evidence_v2(claim, source_info, task, encoder, top_k=3):
    """Stable handoff API: rank evidence WITHOUT assigning a verification label.

    Returns [{rank, evidence, score, passage_index}, ...], ranked best first.
    `score` is a within-document relevance score, never a probability.
    """
    passages = make_passages(evidence_to_text(source_info), task)
    ranked = rank_claim(claim, passages, encoder, top_k=top_k)
    return [
        {"rank": rank, "evidence": passage, "score": score, "passage_index": idx}
        for rank, (idx, score, passage) in enumerate(ranked, start=1)
    ]
