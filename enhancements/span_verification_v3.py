"""Span-aware atomic verification using reusable ranking and NLI models.

Experimental: scores are uncalibrated, results are provisional. Gold labels are
never read. Existing pipeline results are not modified in place by runner.
"""
from __future__ import annotations

from copy import deepcopy

from enhancements.span_alignment_v3 import align_span
from enhancements.ollama_claim_decomposer import decompose_claim
from enhancements.span_targeted_decomposer import targeted_decompose
from enhancements.structured_evidence import normalize_structured_evidence
from enhancements.evidence_relevance_guard import assess_evidence_relevance
from enhancements.structured_source_lookup import lookup_span_attribute

NLI_MODEL = 'cross-encoder/nli-MiniLM2-L6-H768'


def _span_context(span, source_info, task):
    lookup = None
    if str(task).casefold() == 'data2txt' and isinstance(span, str):
        lookup = lookup_span_attribute(source_info, span)
    return {'span_match_status': 'matched', 'attribute_lookup': lookup}


def _verify_one_atomic(atom, evidence_list, task, nli_model, span_context, min_score=0.80):
    usable = []
    for passage in evidence_list:
        original = passage.get('evidence', '')
        if not isinstance(original, str) or not original.strip():
            continue
        relevance = assess_evidence_relevance(span_context, original)
        if relevance.get('status') == 'irrelevant':
            continue
        e = normalize_structured_evidence(original) if str(task).casefold() == 'data2txt' else original
        usable.append((passage, e, relevance.get('status', 'uncertain')))

    if not usable:
        return {'atomic_claim': atom, 'category': None, 'status': 'needs_review',
                'reason': 'no_relevant_evidence', 'evidence_checks': []}

    pairs = [(normalized, atom) for _, normalized, _ in usable]
    scores = nli_model.predict(pairs, apply_softmax=True)
    labels = {int(k): str(v).casefold() for k, v in nli_model.model.config.id2label.items()}
    checks = []
    for (entry, normalized, relevance), probs in zip(usable, scores):
        values = {labels[i]: round(float(prob), 4) for i, prob in enumerate(probs)}
        relation = labels[int(probs.argmax())]
        checks.append({'rank': entry.get('rank'), 'evidence': entry['evidence'],
                       'evidence_used_for_nli': normalized, 'ranking_score': entry.get('score'), 'relevance': relevance,
                       'nli_relation': relation, 'nli_scores': values})

    strong_supported = [x for x in checks if x['nli_relation'] == 'entailment'
                        and x['nli_scores']['entailment'] >= min_score]
    strong_contradicted = [x for x in checks if x['nli_relation'] == 'contradiction'
                          and x['nli_scores']['contradiction'] >= min_score]
    if strong_supported and strong_contradicted:
        category, status, reason = None, 'needs_review', 'conflicting_evidence'
    elif strong_contradicted:
        category, status, reason = 'Contradicted', 'provisional', 'multi_evidence_contradiction'
    elif strong_supported:
        category, status, reason = 'Supported', 'provisional', 'multi_evidence_entailment'
    else:
        category, status, reason = 'Unverifiable', 'provisional', 'no_confident_relation'
    return {'atomic_claim': atom, 'category': category, 'status': status,
            'reason': reason, 'evidence_checks': checks}


def aggregate_atom_results(atom_results):
    if not atom_results:
        return {'category': None, 'status': 'needs_review', 'decision_reason': 'no_aligned_facts'}
    categories = [item.get('category') for item in atom_results]
    if None in categories:
        return {'category': None, 'status': 'needs_review', 'decision_reason': 'atomic_verification_uncertain'}
    unique = set(categories)
    if unique == {'Supported'}:
        category = 'Supported'
    elif unique == {'Contradicted'}:
        category = 'Contradicted'
    elif 'Contradicted' in unique and len(unique) > 1:
        return {'category': None, 'status': 'needs_review', 'decision_reason': 'mixed_contradiction_other'}
    elif unique == {'Unverifiable'}:
        category = 'Unverifiable'
    elif unique == {'Supported', 'Unverifiable'}:
        category = 'Partially Supported'
    else:
        return {'category': None, 'status': 'needs_review', 'decision_reason': 'unexpected_category_mixture'}
    return {'category': category, 'status': 'provisional', 'decision_reason': 'multi_atom_evidence_decision'}


def add_spanaware_verification(result, source_info, ranker, *, targeted_fallback=True,
                               min_nli_score=0.80, nli_model=None):
    """Add provisional verification to the existing analyze_record() output.

    Supports injection of an NLI stub and deterministic fallback for offline tests.
    Decomposition is cached by parent claim, and evidence is reranked for each
    atomic fact using the SAME reusable encoder passed via ranker.
    """
    reviews = result.get('claims_with_evidence', [])
    if not reviews:
        result['verification_method'] = 'spanaware_v3_multi_evidence'
        return result
    if nli_model is None:
        from sentence_transformers import CrossEncoder
        nli_model = CrossEncoder(NLI_MODEL, device='cpu')
    parent_cache = {}
    rank_cache = {}
    verify_cache = {}
    for item in reviews:
        parent = item.get('claim', '')
        span = item.get('baseline_span', {})
        task = result.get('task', '')
        if not isinstance(span, dict):
            span = {}
        try:
            if parent not in parent_cache:
                parent_cache[parent] = decompose_claim(parent)
            atomic_claims = parent_cache[parent]
            fallback = targeted_decompose if targeted_fallback else None
            alignment = align_span(result.get('answer', ''), span, parent, atomic_claims, fallback)
            item['span_alignment_v3'] = alignment
            if alignment['status'] != 'matched':
                item['verification'] = {'category': None, 'status': 'needs_review',
                                        'decision_reason': 'span_alignment_unresolved'}
                item['verification_status'] = 'needs_review'
                continue
            atom_results = []
            for atom in alignment['claims']:
                key = (atom, str(task))
                if key not in rank_cache:
                    rank_cache[key] = ranker(atom, source_info, task, 3)
                ranked = rank_cache[key]
                context = _span_context(alignment['normalized_span'], source_info, task)
                vkey = (atom, str(task), str(context.get('attribute_lookup')),
                        tuple(p.get('evidence', '') for p in ranked))
                if vkey not in verify_cache:
                    verify_cache[vkey] = _verify_one_atomic(
                        atom, ranked, task, nli_model, context,
                        min_score=min_nli_score)
                atom_results.append(deepcopy(verify_cache[vkey]))
            decision = aggregate_atom_results(atom_results)
            item['verification'] = {**decision,
                                    'atomic_verifications_v3': atom_results,
                                    'scope': 'atomic_claim_top3_reranked',
                                    'alignment_method': alignment['method']}
            item['verification_status'] = decision['category'] or 'needs_review'
        except Exception as exc:
            item['verification_status'] = 'VERIFICATION_ERROR'
            item['verification'] = {'category': None, 'status': 'error',
                                    'error_type': type(exc).__name__,
                                    'error_message': str(exc)[:300]}
    result['verification_method'] = 'spanaware_v3_multi_evidence'
    return result
