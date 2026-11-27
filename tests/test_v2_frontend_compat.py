"""Offline compatibility tests; no model loading or network access."""
import copy
import json
from unittest.mock import patch

from enhancements.v2_enhanced_adapter import (
    adapt_enhanced_output_for_frontend,
    add_enhanced_verification_for_frontend,
)


def _result():
    return {
        'task': 'Data2Txt', 'answer': 'Wi-Fi is available, but parking is not mentioned.',
        'claims_with_evidence': [{
            'claim': 'Wi-Fi is available, but parking is not mentioned.',
            'baseline_span': {'text': 'Wi-Fi is available', 'confidence': .96, 'start': 0, 'end': 19},
            'ranked_evidence': [{'rank': 1, 'evidence': 'WiFi: no', 'score': .82}],
            'verification': {
                'category': 'Partially Supported', 'status': 'provisional',
                'decision_reason': 'multi_atom_evidence_decision',
                'atomic_verifications_v3': [
                    {'atomic_claim': 'The restaurant offers free Wi-Fi.', 'category': 'Supported',
                     'status': 'provisional', 'reason': 'multi_evidence_entailment',
                     'evidence_checks': [{'rank': 1, 'evidence': 'WiFi: free',
                                          'ranking_score': .89, 'nli_relation': 'entailment',
                                          'nli_scores': {'entailment': .98, 'neutral': .01,
                                                         'contradiction': .01}}]},
                    {'atomic_claim': 'The restaurant offers parking.', 'category': 'Unverifiable',
                     'status': 'provisional', 'reason': 'no_confident_relation',
                     'evidence_checks': [{'rank': 1, 'evidence': 'Parking: unknown',
                                          'ranking_score': .41, 'nli_relation': 'neutral',
                                          'nli_scores': {'entailment': .05, 'neutral': .91,
                                                         'contradiction': .04}}]},
                ],
            },
        }],
    }


def test_schema_preserves_verdict_and_evidence_provenance():
    r = adapt_enhanced_output_for_frontend(_result())
    item = r['claims_with_evidence'][0]
    v = item['verification']
    assert v['category'] == 'Partially Supported'
    assert v['supported_count'] == 1
    assert v['total_atomic_claims'] == 2
    assert v['verification_backend'] == 'v2_spanaware_enhanced'
    assert v['atomic_verifications'][0]['nli_relation'] == 'entailment'
    assert v['atomic_verifications'][1]['nli_relation'] == 'neutral'
    assert item['parent_ranked_evidence'][0]['evidence'] == 'WiFi: no'
    assert item['ranked_evidence'][0]['evidence'] == 'WiFi: free'
    assert item['ranked_evidence'][1]['evidence'] == 'Parking: unknown'
    assert v['evidence'] == 'WiFi: free'
    assert v['ranking_score'] == .89
    json.dumps(r)


def test_unresolved_keeps_parent_evidence():
    r = _result()
    v = r['claims_with_evidence'][0]['verification']
    v.update(category=None, status='needs_review', atomic_verifications_v3=[])
    out = adapt_enhanced_output_for_frontend(r)
    assert out['claims_with_evidence'][0]['ranked_evidence'][0]['evidence'] == 'WiFi: no'
    assert v['total_atomic_claims'] == 0
    assert v['supported_count'] == 0


def test_backend_injects_same_result_structure():
    r = _result()
    with patch('enhancements.v2_enhanced_adapter._run_enhanced', return_value=copy.deepcopy(r)) as verifier, \
         patch('enhancements.v2_enhanced_adapter._ranker', return_value=lambda *_: []) as ranker:
        out = add_enhanced_verification_for_frontend(r, source_info='WiFi: free')
        assert out['claims_with_evidence'][0]['verification']['verification_backend'] == 'v2_spanaware_enhanced'
        verifier.assert_called_once()
        ranker.assert_called_once()


def test_source_required_for_enhanced_path():
    try:
        add_enhanced_verification_for_frontend({}, source_info=None)
    except ValueError:
        pass
    else:
        raise AssertionError('Should require source_info for atomic reranking')
