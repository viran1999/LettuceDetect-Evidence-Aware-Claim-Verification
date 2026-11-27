#!/usr/bin/env python3
"""Data-driven IEEE-sized figures from recorded LettuceDetect relevance judgments.

This script DOES NOT run new retrieval, LettuceDetect, or AI labeling. Instead it
recalculates rankings metrics directly from row-level experiment outputs:

  30-example raw rank-1/2/3 relevance labels (TF-IDF / Semantic / Hybrid)
  88-claim paired per-claim outcomes (v1 and v2)
  547 individual claim-passage relevance judgments

NO PLOTTED RESULTS ARE HARDCODED. See README.md for evaluation limitations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import fmean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PRELIM_FILE = 'evidence_ranking_review_compact_rated.csv'
PAIRED_FILE = 'LettuceDetect_v1_v2_AI_paired_claims.csv'
JUDGMENT_FILE = 'LettuceDetect_v1_v2_AI_relevance_review.csv'

# Labels and rendering configuration, not hardcoded observations.
METHODS = ('TF-IDF', 'Semantic', 'Hybrid')
METRICS = [('Precision@1', 'p_at_1'), ('Hit@3', 'hit_at_3'),
           ('MRR@3', 'mrr_at_3'), ('Precision@3', 'p_at_3')]
TASK_ORDER = ('QA', 'Data2Txt', 'Summary')
FILE_STEMS = {
    1: 'fig01_preliminary_ranking',
    2: 'fig02_v1_v2_ranking',
    3: 'fig03_task_precision',
    4: 'fig04_relevance_judgments',
    5: 'fig05_paired_rank1_outcomes',
    6: 'fig06_metric_delta_intervals',
}

# IEEE single-column figures: plot type >= 10 pt BEFORE LaTeX insertion.
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 10,
    'savefig.dpi': 320,
})


def load_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f'Missing required experiment CSV: {path}')
    with path.open(newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        if not set(required) <= set(reader.fieldnames or []):
            raise ValueError(f'Incorrect columns in {path.name}; expected {required}, got {reader.fieldnames}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'Empty experiment CSV: {path}')
    return rows


def rating(raw: str | None, row_context: str) -> int:
    value = (raw or '').strip()
    if value not in {'0', '1', '0.0', '1.0'}:
        raise ValueError(f'Incomplete/invalid binary rating {value!r} at {row_context}')
    return int(float(value))


def load_preliminary(path: Path):
    rows = load_csv(path, ('task', 'response_id', 'method', 'example_number',
                           'useful_rank_1_0_or_1', 'useful_rank_2_0_or_1',
                           'useful_rank_3_0_or_1'))
    by_example = defaultdict(dict)
    for row in rows:
        example_key = (row['task'], row['response_id'], row['example_number'])
        method = row['method'].strip()
        if method not in METHODS:
            raise ValueError(f'Unexpected method {method!r}; expected {METHODS}')
        if method in by_example[example_key]:
            raise ValueError(f'Duplicate method {method} for {example_key}')
        by_example[example_key][method] = [
            rating(row.get(f'useful_rank_{i}_0_or_1'), f'{example_key} {method} rank {i}')
            for i in (1, 2, 3)
        ]
    complete = [example for example in by_example.values()
                if all(method in example for method in METHODS)]
    if len(complete) != len(by_example):
        raise ValueError(f'{len(by_example) - len(complete)} examples are missing a method; no silent exclusion.')
    if not complete:
        raise ValueError('No paired preliminary examples')
    out = {}
    for method in METHODS:
        triples = [example[method] for example in complete]
        out[method] = {
            'n': len(triples),
            'p_at_1': fmean(v[0] for v in triples),
            'hit_at_3': fmean(int(any(v)) for v in triples),
            'mrr_at_3': fmean(next((1.0 / (i + 1) for i, y in enumerate(v) if y), 0.0) for v in triples),
            'p_at_3': fmean(sum(v) / 3 for v in triples),
        }
    return out


def load_paired(path: Path):
    required = ('claim_id', 'task', 'source_id', 'v1_p_at_1', 'v2_p_at_1',
                'v1_hit_at_3', 'v2_hit_at_3', 'v1_mrr_at_3', 'v2_mrr_at_3',
                'v1_relevant_top_3', 'v2_relevant_top_3', 'winner_at_1')
    rows = load_csv(path, required)
    seen = set()
    for row in rows:
        claim_id = row['claim_id']
        if claim_id in seen:
            raise ValueError(f'Duplicate claim ID {claim_id}')
        seen.add(claim_id)
        if not row['source_id']:
            raise ValueError(f'Missing source_id for claim {claim_id}')
        for version in ('v1', 'v2'):
            for col in ('p_at_1', 'hit_at_3', 'mrr_at_3', 'relevant_top_3'):
                val = float(row[f'{version}_{col}'])
                if not 0 <= val <= (3 if col == 'relevant_top_3' else 1):
                    raise ValueError(f'Invalid metric {version}_{col}={val} for {claim_id}')
        # Verify the recorded rank-1 winner still agrees with the per-claim labels.
        p1 = float(row['v1_p_at_1']); p2 = float(row['v2_p_at_1'])
        expected = 'v2' if p2 > p1 else ('v1' if p1 > p2 else 'tie')
        if row['winner_at_1'].strip().lower() != expected:
            raise ValueError(f'Inconsistent winner_at_1 for {claim_id}')
    return rows


def score(row: dict[str, str], version: str, metric: str) -> float:
    """Compute an individual claim's metric from its rank outcomes."""
    if metric == 'p_at_3':
        return float(row[f'{version}_relevant_top_3']) / 3.0
    return float(row[f'{version}_{metric}'])


def aggregate(rows, version: str, metric: str) -> float:
    return fmean(score(row, version, metric) for row in rows)


def paired_metrics(rows):
    tasks = sorted({r['task'] for r in rows}, key=lambda t: (TASK_ORDER.index(t) if t in TASK_ORDER else len(TASK_ORDER), t))
    groups = {'ALL': rows}
    groups.update({t: [r for r in rows if r['task'] == t] for t in tasks})
    table = {}
    for group, items in groups.items():
        table[group] = {'n_claims': len(items), 'n_sources': len({r['source_id'] for r in items}),
                        'metrics': {metric: {version: aggregate(items, version, key) for version in ('v1', 'v2')}
                                    for metric, key in METRICS}}
    return table


def load_judgments(path: Path):
    rows = load_csv(path, ('claim_id', 'passage_id', 'useful_0_or_1_or_U'))
    labels = Counter(row['useful_0_or_1_or_U'].strip().upper() for row in rows)
    if set(labels) - {'0', '1', 'U'}:
        raise ValueError(f'Unexpected relevance labels: {dict(labels)}')
    if sum(labels.values()) != len(rows):
        raise AssertionError('Invalid relevance label counts')
    return labels


def source_cluster_ci(rows, metric: str, replicates: int, seed: int):
    """Match the original experiment's 95% source-cluster bootstrap procedure.

    Original review_and_score_ranking_v2.py uses sorted source IDs,
    random.Random(20261101), 5,000 resamples and indexed percentiles.
    """
    source_rows = defaultdict(list)
    for row in rows:
        source_rows[row['source_id']].append(row)
    ids = sorted(source_rows)
    if len(ids) < 2:
        raise ValueError('At least two source IDs needed for bootstrap')
    rng = random.Random(seed)
    sampled_deltas = []
    for _ in range(replicates):
        drawn = [row for _ in ids for row in source_rows[rng.choice(ids)]]
        sampled_deltas.append(aggregate(drawn, 'v2', metric) - aggregate(drawn, 'v1', metric))
    sampled_deltas.sort()
    return (sampled_deltas[int(0.025 * (replicates - 1))],
            sampled_deltas[int(0.975 * (replicates - 1))])


def decorate(ax, axis='y'):
    ax.grid(axis=axis, alpha=.25)
    ax.set_axisbelow(True)


def grouped(labels, a, b, version_labels=('Hybrid v1', 'Hybrid v2'), ylim=105):
    fig, ax = plt.subplots(figsize=(3.45, 2.7), layout='constrained')
    x = np.arange(len(labels), dtype=float)
    width = .36
    ax.bar(x - width / 2, a, width, label=version_labels[0])
    ax.bar(x + width / 2, b, width, label=version_labels[1])
    ax.set_xticks(x, labels)
    ax.set_ylabel('Retrieval score (%)')
    ax.set_ylim(0, ylim)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.24), frameon=False, ncol=2)
    decorate(ax)
    return fig


def figure_one(prelim):
    fig, ax = plt.subplots(figsize=(3.45, 2.8), layout='constrained')
    metrics = [('P@1', 'p_at_1'), ('Hit@3', 'hit_at_3'), ('MRR@3', 'mrr_at_3')]
    x = np.arange(len(metrics), dtype=float)
    width = .24
    for index, name in enumerate(METHODS):
        scores = [prelim[name][key] * 100 for _, key in metrics]
        ax.bar(x + (index - 1) * width, scores, width, label=('Cosine' if name == 'Semantic' else name))
    ax.set_xticks(x, [label for label, _ in metrics])
    ax.set_ylim(0, 65)
    ax.set_ylabel('Retrieval score (%)')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.24), ncol=3, frameon=False)
    decorate(ax)
    return fig


def figure_two(stats):
    pairs = [stats['ALL']['metrics'][label] for label, _ in METRICS]
    return grouped(['P@1', 'Hit@3', 'MRR@3', 'P@3'],
                   [m['v1'] * 100 for m in pairs], [m['v2'] * 100 for m in pairs])


def figure_three(stats):
    tasks = [task for task in stats if task != 'ALL']
    return grouped([f'{task}\n(n={stats[task]["n_claims"]})' for task in tasks],
                   [stats[t]['metrics']['Precision@1']['v1'] * 100 for t in tasks],
                   [stats[t]['metrics']['Precision@1']['v2'] * 100 for t in tasks])


def count_bars(labels, values, y_caption, ylim_multiplier=1.17):
    fig, ax = plt.subplots(figsize=(3.45, 2.55), layout='constrained')
    x = np.arange(len(labels))
    ax.bar(x, values, width=.65)
    ax.set_xticks(x, labels)
    ax.set_ylabel(y_caption)
    ax.set_ylim(0, max(values) * ylim_multiplier if max(values) else 1)
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * .025, str(v), ha='center', va='bottom', fontsize=10)
    decorate(ax)
    return fig


def figure_four(judgments):
    return count_bars(['Useful', 'Not useful', 'Uncertain'],
                      [judgments.get(v, 0) for v in ('1', '0', 'U')],
                      'Relevance judgments')


def figure_five(paired):
    wins = Counter(row['winner_at_1'].strip().lower() for row in paired)
    return count_bars(['v2 better', 'v1 better', 'Tied'],
                      [wins.get(v, 0) for v in ('v2', 'v1', 'tie')],
                      'Paired claims')


def figure_six(ci_data):
    fig, ax = plt.subplots(figsize=(3.45, 2.8), layout='constrained')
    x = np.arange(len(METRICS))
    deltas = np.array([ci_data[label]['delta'] * 100 for label, _ in METRICS])
    ci_lows = np.array([ci_data[label]['ci_low'] * 100 for label, _ in METRICS])
    ci_highs = np.array([ci_data[label]['ci_high'] * 100 for label, _ in METRICS])
    ax.errorbar(deltas, x, xerr=[deltas - ci_lows, ci_highs - deltas], fmt='o', capsize=4)
    ax.set_yticks(x, ['P@1', 'Hit@3', 'MRR@3', 'P@3'])
    ax.invert_yaxis()
    ax.axvline(0, linestyle='--', linewidth=1)
    ax.set_xlabel('v2 − v1 (percentage points)')
    ax.set_xlim(min(0, float(ci_lows.min())) - 2, float(ci_highs.max()) + 3)
    decorate(ax, 'x')
    return fig


def save_metrics(prelim, paired, stats, judgments, ci_data, data_paths, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        'description': 'Recalculated from source CSV rows; no plotted numerical results hardcoded.',
        'limitations': 'AI-assisted relevance labels; development data; retrieval not hallucination-detection accuracy.',
        'sources': {key: {'filename': value.name, 'sha256': sha256(value.read_bytes()).hexdigest()}
                    for key, value in data_paths.items()},
        'preliminary': prelim,
        'paired': stats,
        'relevance_judgments': dict(judgments),
        'winner_at_1': dict(Counter(row['winner_at_1'].strip().lower() for row in paired)),
        'bootstrap_95ci': ci_data,
    }
    with (output / 'recomputed_results.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    with (output / 'recomputed_paired_metrics.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['task', 'metric', 'n_claims', 'n_sources',
                                          'v1', 'v2', 'delta_v2_minus_v1',
                                          'delta_95ci_low', 'delta_95ci_high'])
        w.writeheader()
        for task, obj in stats.items():
            for label, key in METRICS:
                a = obj['metrics'][label]['v1']; b = obj['metrics'][label]['v2']
                lo = hi = ''
                if task == 'ALL':
                    lo, hi = ci_data[label]['ci_low'], ci_data[label]['ci_high']
                w.writerow({
                    'task': task, 'metric': key, 'n_claims': obj['n_claims'],
                    'n_sources': obj['n_sources'], 'v1': f'{a:.4f}', 'v2': f'{b:.4f}',
                    'delta_v2_minus_v1': f'{(b - a):.4f}',
                    'delta_95ci_low': f'{lo:.4f}' if lo != '' else '',
                    'delta_95ci_high': f'{hi:.4f}' if hi != '' else '',
                })

    with (output / 'recomputed_preliminary_metrics.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['method', 'n_examples', 'Precision@1', 'Hit@3', 'MRR@3', 'Precision@3'])
        w.writeheader()
        for method, obj in prelim.items():
            w.writerow({'method': method, 'n_examples': obj['n'],
                        **{label: f'{obj[key]:.4f}' for label, key in METRICS}})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    selection = ap.add_mutually_exclusive_group(required=True)
    selection.add_argument('--all', action='store_true', help='Recalculate data and generate all six plots')
    selection.add_argument('--figure', type=int, choices=FILE_STEMS, help='Generate one numbered plot')
    selection.add_argument('--validate-only', action='store_true', help='Recalculate/print metrics only')
    ap.add_argument('--data-dir', type=Path, default=Path('evaluation_data'),
                    help='Folder containing the three row-level experiment CSV files')
    ap.add_argument('--output-dir', type=Path, default=Path('figures'),
                    help='Figure image destination (PNG)')
    ap.add_argument('--metrics-dir', type=Path, default=Path('evaluation_metrics'),
                    help='Calculated metrics and SHA-256 provenance files')
    ap.add_argument('--bootstrap', type=int, default=5000,
                    help='Resamples for 95%% source-cluster CIs (default: 5000)')
    ap.add_argument('--seed', type=int, default=20261101,
                    help='Exact original review script bootstrap seed')
    args = ap.parse_args()
    if args.bootstrap < 100:
        ap.error('--bootstrap must be >= 100')

    files = {
        'preliminary': args.data_dir / PRELIM_FILE,
        'paired': args.data_dir / PAIRED_FILE,
        'judgments': args.data_dir / JUDGMENT_FILE,
    }
    prelim = load_preliminary(files['preliminary'])
    paired = load_paired(files['paired'])
    judgments = load_judgments(files['judgments'])
    stats = paired_metrics(paired)
    ci_data = {}
    for label, key in METRICS:
        lo, hi = source_cluster_ci(paired, key, args.bootstrap, args.seed)
        a = stats['ALL']['metrics'][label]['v1']
        b = stats['ALL']['metrics'][label]['v2']
        ci_data[label] = {'delta': b - a, 'ci_low': lo, 'ci_high': hi,
                          'bootstrap_resamples': args.bootstrap, 'seed': args.seed}
    save_metrics(prelim, paired, stats, judgments, ci_data, files, args.metrics_dir)

    print(f'Validated: {next(iter(prelim.values()))["n"]} preliminary paired examples, '
          f'{len(paired)} v1-v2 claims, {sum(judgments.values())} relevance judgments')
    for label, _ in METRICS:
        a, b = stats['ALL']['metrics'][label]['v1'], stats['ALL']['metrics'][label]['v2']
        print(f'{label:12}: Hybrid v1 {a:7.2%} | Hybrid v2 {b:7.2%} | delta {(b - a) * 100:+.2f} pp')
    print('Review-note: AI-assisted labels; this is evidence retrieval, not end-to-end hallucination accuracy.')

    if args.validate_only:
        print(f'Saved calculated results: {args.metrics_dir}')
        return

    options = range(1, 7) if args.all else [args.figure]
    for number in options:
        if number == 1: fig = figure_one(prelim)
        elif number == 2: fig = figure_two(stats)
        elif number == 3: fig = figure_three(stats)
        elif number == 4: fig = figure_four(judgments)
        elif number == 5: fig = figure_five(paired)
        elif number == 6: fig = figure_six(ci_data)
        else: raise AssertionError(number)
        target = args.output_dir / f'{FILE_STEMS[number]}.png'
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=320, bbox_inches='tight')
        plt.close(fig)
        print(f'Generated: {target.resolve()}')
    print(f'Calculated metrics and provenance: {args.metrics_dir.resolve()}')


if __name__ == '__main__':
    main()
