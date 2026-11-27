#!/usr/bin/env python3
"""Separate experimental span-aware runner. Does not modify the v2 pipeline.

Compatible with run_combined_pipeline_batch.py --program run_spanaware_v3.py.
"""
import argparse
import json
from pathlib import Path

from run_baseline_evidence_integration import (
    DEFAULT_MODEL, analyze_record, extract_question, load_record,
)
from enhancements.span_verification_v3 import add_spanaware_verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--response-id')
    group.add_argument('--demo', action='store_true')
    parser.add_argument('--data-dir', default='data/ragtruth')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--output', required=True)
    parser.add_argument('--no-targeted-fallback', action='store_true')
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error('--top-k must be >= 1')
    output = Path(args.output)
    if output.exists():
        parser.error(f'Output already exists (will not overwrite): {output}')
    if not output.parent.is_dir():
        parser.error(f'Output parent folder does not exist: {output.parent}')
    if args.demo:
        answer = 'The capital of France is Paris. The population of France is 69 million.'
        source_info = 'The capital of France is Paris. The population of France is 67 million.'
        question = 'What is the capital and population of France?'
        task, source_id, response_id = 'QA', 'demo', 'demo'
    else:
        response, source = load_record(str(args.response_id), Path(args.data_dir))
        answer = response.get('response', '')
        source_info = source.get('source_info')
        if source_info is None or source_info == '':
            source_info = source.get('prompt', '')
        question = extract_question(source_info, source)
        task = str(source.get('task_type', 'unknown'))
        source_id, response_id = str(response.get('source_id', '')), str(response.get('id', ''))

    from lettucedetect.models.inference import HallucinationDetector
    from sentence_transformers import SentenceTransformer
    from enhancements.evidence_ranking_v2 import rank_evidence_v2

    print(f'Loading LettuceDetect: {args.model}', flush=True)
    detector = HallucinationDetector(method='transformer', model_path=args.model)
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')

    def ranker(claim, source, task_label, k):
        return rank_evidence_v2(claim, source, task_label, encoder, top_k=k)

    result = analyze_record(detector=detector, answer=answer,
                            source_info=source_info, task=task, question=question,
                            ranker=ranker, top_k=args.top_k,
                            source_id=source_id, response_id=response_id)
    result['baseline_model'] = args.model
    result['evidence_ranker'] = 'v2'
    result['ranking_score_note'] = 'Within-document relevance scores; not hallucination probabilities.'
    result = add_spanaware_verification(
        result, source_info, ranker,
        targeted_fallback=not args.no_targeted_fallback,
    )
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    counts = {}
    for review in result.get('claims_with_evidence', []):
        label = review.get('verification_status', 'missing')
        counts[label] = counts.get(label, 0) + 1
    print('Response', response_id, 'spans', len(result['baseline_spans']),
          'categories', counts, flush=True)
    print('Saved', output.resolve(), flush=True)


if __name__ == '__main__':
    main()
