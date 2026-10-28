#!/usr/bin/env python3
"""main.py — Entry point for real-world QKRR regression experiments."""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments import exp_real_accuracy, ensure_reproducibility


DATASETS = ['household', 'nslkdd', 'gaze', 'no2', 'ise']


def main():
    parser = argparse.ArgumentParser(description='QKRR vs neural baselines on real-world data')
    parser.add_argument(
        '--experiment',
        choices=['all', 'real_accuracy'],
        default='all',
        help='Which experiment to run',
    )
    parser.add_argument('--dataset',
                        choices=DATASETS + ['all'],
                        default='all',
                        help='Real-world dataset (default: run all five benchmarks)')
    parser.add_argument('--results-dir',
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'),
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-samples', type=int, default=1000,
                        help='Max real-data samples')
    parser.add_argument('--n-epochs', type=int, default=100,
                        help='Training epochs for neural baselines')
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    seed = args.seed
    max_samples = args.max_samples
    n_epochs = args.n_epochs

    print(f"\n{'#'*80}")
    print(f"# QKRR vs neural baselines — real-world regression")
    print(f"# Datasets: {', '.join(datasets)} | Epochs: {n_epochs}")
    print(f"# Results: {results_dir}")
    print(f"{'#'*80}")

    ensure_reproducibility(seed)
    t0 = time.time()

    if args.experiment in ('all', 'real_accuracy'):
        for dataset in datasets:
            exp_real_accuracy(dataset, results_dir, max_samples=max_samples,
                              n_epochs=n_epochs, seed=seed)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Completed {len(datasets)} dataset(s) in {elapsed/60:.1f} minutes.")
    print(f"Results: {results_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
