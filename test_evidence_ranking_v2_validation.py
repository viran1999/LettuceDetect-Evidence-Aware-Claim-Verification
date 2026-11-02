"""Offline regression and integrity tests for the v2 ranking validation workflow."""
import json

import numpy as np

from enhancements.evidence_ranking_v2 import evidence_signal, make_passages, rank_claim
from benchmark_evidence_ranking_v2 import evaluation_one, norm_task, select_records
from review_and_score_ranking_v2 import (
    collect_scored_pairs, metrics_for, passage_id,
)


class StubEncoder:
    def encode(self, texts, **_):
        return np.array([[float('wifi' in t.lower() or 'wi-fi' in t.lower()),
                          float('seating' in t.lower()), float('parking' in t.lower())]
                         for t in texts], dtype=float)


class FakeDetector:
    def __init__(self, predictions):
        self.predictions = predictions
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return self.predictions


def test_preserve_false_and_no_but_not_unknown():
    chunks = make_passages('WiFi: no OutdoorSeating: False Music: None', 'Data2Txt')
    assert 'WiFi: no' in chunks
    assert 'OutdoorSeating: False' in chunks
    assert 'Music: None' not in chunks


def test_qa_question_metadata_not_ranked():
    chunks = make_passages('question: tips how to conserve water passages: Turn off the tap when brushing your teeth. Use a shower bucket.', 'QA')
    assert chunks
    assert all(not p.lower().startswith('question:') for p in chunks)
    assert any('turn off the tap' in p.lower() for p in chunks)


def test_wifi_retrieval_top_one():
    passages = make_passages('name: Apna Indian Kitchen WiFi: no OutdoorSeating: True', 'Data2Txt')
    ranked = rank_claim('The restaurant provides free WiFi.', passages, StubEncoder())
    assert ranked[0][2].lower() == 'wifi: no'


def test_parking_field_boost():
    assert evidence_signal('Business parking is available', 'BusinessParking: False') > evidence_signal('Business parking is available', 'classy: True')


def test_task_normalization():
    assert all(norm_task(x) == 'Data2Txt' for x in ('Data2Txt', 'Data2txt', 'Data2Text'))


def test_sample_per_source_and_split(tmp_path):
    root = tmp_path / 'ragtruth'; root.mkdir()
    sources = [dict(source_id=str(i), task_type='QA', source_info='Facts about test') for i in range(4)]
    with (root / 'source_info.jsonl').open('w') as f:
        for s in sources: f.write(json.dumps(s) + '\n')
    with (root / 'response.jsonl').open('w') as f:
        for sid in range(4):
            for j in range(5):
                r = dict(id=f'{sid}-{j}', source_id=str(sid), response='Incorrect answer',
                         split='test' if j > 0 else 'train', labels=[dict(start=0, end=9)])
                f.write(json.dumps(r) + '\n')
    selection = select_records(root, 'test', 4, 1, 42)
    assert len(selection) == 4
    assert len({r['source_id'] for r, _ in selection}) == 4
    assert all(r['split'] == 'test' for r, _ in selection)
    assert len(select_records(root, 'test', 4, 1, 42, {'0'})) == 3


def test_gold_label_not_passed_to_rankers():
    answer = 'The restaurant offers free WiFi.'
    start = answer.index('free WiFi')
    detector = FakeDetector([{'start': start, 'end': start + 9, 'confidence': .9}])
    seen = []

    def ranker(claim, source, task, k):
        seen.append((claim, source, task, k))
        return [dict(rank=1, evidence='WiFi: no', score=.6)]
    source = {'task_type': 'Data2Txt', 'source_info': 'WiFi: no'}
    response = {'id': '3', 'source_id': '7', 'response': answer,
                'labels': [{'start': start, 'end': start + 9, 'meta': 'GOLD SECRET', 'label_type': 'Evident Conflict'}],
                'split': 'test'}
    out = evaluation_one(detector, {'v1': ranker, 'v2': ranker}, response, source, 3, 3)
    assert len(detector.calls) == 1
    assert len(seen) == 2
    assert all('GOLD SECRET' not in str(x) for x in detector.calls + seen)
    assert out['ranked_claims'][0]['baseline_span']['text'] == 'free WiFi'


def test_pair_metrics():
    m = metrics_for([[0, 1, 0], [1, 0, 1]])
    assert m['p_at_1'] == .5
    assert m['hit_at_3'] == 1
    assert m['mrr_at_3'] == .75
    assert m['p_at_3'] == .5


def test_blinded_shared_label_pair():
    c = {'claim_id': '10:0:9', 'source_id': '4', 'task': 'QA', 'claim': 'Bad claim',
         'rankings': {'v1': [{'evidence': 'correct passage'}, {'evidence': 'other passage'}],
                      'v2': [{'evidence': 'correct passage'}, {'evidence': 'new passage'}]}}
    labels = {(c['claim_id'], passage_id(c['claim_id'], text)): '1' if text == 'correct passage' else '0'
              for text in ('correct passage', 'other passage', 'new passage')}
    scored, missing, uncertain = collect_scored_pairs([c], labels)
    assert len(scored) == 1 and not missing and not uncertain
    assert scored[0]['methods']['v1'] == [1, 0]
    assert scored[0]['methods']['v2'] == [1, 0]
