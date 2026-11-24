#!/usr/bin/env python3
"""Pilot targeted Qwen re-decomposition on original Phase 1 failures.

No modification of original result JSON files or Git-tracked pipeline modules.
Use --max-cases 3 at first. Each row flushed immediately to output CSV.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from enhancements.span_targeted_decomposer import targeted_decompose

FIELDS = ['response_id', 'task', 'suspicious_span', 'original_claim',
          'phase1_reason', 'phase1_status', 'phase2_status', 'target_claims_json',
          'screening_flags_json', 'error']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='results/span_alignment_phase1_candidates.csv')
    parser.add_argument('--output', default='results/span_alignment_phase2_targeted_pilot.csv')
    parser.add_argument('--max-cases', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--response-ids', default='',
                        help='Optional comma-separated response IDs for a representative pilot')
    parser.add_argument('--model', default='qwen2.5:0.5b-instruct')
    args = parser.parse_args()
    if args.max_cases < 1:
        parser.error('--max-cases must be positive')
    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.resolve() == output_path.resolve():
        parser.error('Output must differ from the input')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding='utf-8-sig', newline='') as f:
        inputs = list(csv.DictReader(f))
    targets = [r for r in inputs if r.get('new_candidate_status') == 'unresolved'
               and r.get('reason') == 'no_plausible_atomic_claim']
    if args.response_ids.strip():
        wanted = {x.strip() for x in args.response_ids.split(',') if x.strip()}
        targets = [r for r in targets if r['response_id'] in wanted]
    selected = targets[:args.max_cases]
    print(f'Eligible original no-match cases: {len(targets)}; pilot selected: {len(selected)}', flush=True)
    if not selected:
        return
    # Fail rather than overwriting earlier pilot rows accidentally.
    if output_path.exists():
        parser.error(f'Output exists: {output_path}; select a fresh output filename')
    with output_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        f.flush()
        for i, row in enumerate(selected, 1):
            out = {'response_id': row['response_id'], 'task': row['task'],
                   'suspicious_span': row['suspicious_span'],
                   'original_claim': row['original_claim'],
                   'phase1_reason': row['reason'],
                   'phase1_status': row['new_candidate_status'],
                   'phase2_status': 'error', 'target_claims_json': '[]',
                   'screening_flags_json': '[]', 'error': ''}
            try:
                res = targeted_decompose(row['original_claim'], row['suspicious_span'],
                                         model=args.model, timeout=args.timeout)
                out['phase2_status'] = res['status']
                out['target_claims_json'] = json.dumps(res['proposed_claims'], ensure_ascii=False)
                out['screening_flags_json'] = json.dumps(res['flags'], ensure_ascii=False)
            except Exception as exc:
                out['error'] = f'{type(exc).__name__}: {exc}'[:350]
            writer.writerow(out)
            f.flush()
            print(f'[{i}/{len(selected)}] Response {out["response_id"]}: {out["phase2_status"]}', flush=True)
    print(f'Pilot CSV: {output_path}', flush=True)
    print('IMPORTANT: All model outputs still require human verification.', flush=True)

if __name__ == '__main__':
    main()
