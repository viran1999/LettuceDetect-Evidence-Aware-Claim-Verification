#!/usr/bin/env python3
"""Replay span alignment on saved v2 JSONs without loading any ML models.

Only reports candidate coverage, never categorization accuracy.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from enhancements.span_alignment_v3 import align_span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='results/combined_pipeline_batch_bugfix/json')
    ap.add_argument('--output', default='results/spanaware_v3_alignment_offline.csv')
    args = ap.parse_args()
    destination = Path(args.output)
    if destination.exists():
        ap.error(f'Output exists; refusing to overwrite: {destination}')
    src = Path(args.input)
    if not src.is_dir():
        ap.error(f'Input directory missing: {src}')
    destination.parent.mkdir(exist_ok=True, parents=True)
    totals = Counter()
    fields = ['response_id', 'task', 'previous_status', 'suspicious_span',
              'normalized_span', 'alignment_status', 'reason', 'coverage',
              'selected_claims_json']
    with destination.open('w', encoding='utf-8-sig', newline='') as output:
        w = csv.DictWriter(output, fieldnames=fields)
        w.writeheader()
        for path in sorted(src.glob('*.json')):
            record = json.loads(path.read_text(encoding='utf-8'))
            for review in record.get('claims_with_evidence', []):
                previous = review.get('verification_status', 'unknown')
                if previous not in ('needs_review', 'VERIFICATION_ERROR'):
                    continue
                atoms = [i.get('atomic_claim') for i in
                         (review.get('verification') or {}).get('atomic_verifications', [])
                         if isinstance(i, dict) and isinstance(i.get('atomic_claim'), str)]
                span = review.get('baseline_span', {})
                result = align_span(record.get('answer',''), span, review.get('claim',''), atoms)
                totals[result['status']] += 1
                w.writerow({'response_id': path.stem, 'task': record.get('task'),
                            'previous_status': previous,
                            'suspicious_span': span.get('text') if isinstance(span, dict) else '',
                            'normalized_span': result['normalized_span'],
                            'alignment_status': result['status'],
                            'reason': result['reason'], 'coverage': result['coverage'],
                            'selected_claims_json': json.dumps(result['claims'], ensure_ascii=False)})
    print('Offline candidate counts:', dict(totals))
    print('Saved:', destination)
    print('No models executed. These matches are hypotheses, NOT verified corrections.')

if __name__ == '__main__':
    main()
