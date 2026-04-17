#!/usr/bin/env python3
"""
main.py — Entry point for all experiments.

Usage
-----
  python main.py --experiment all          # run everything
  python main.py --experiment simulation   # 11 synthetic datasets
  python main.py --experiment online       # MNIST / Kaggle / Cybersecurity
  python main.py --experiment kernel       # UCI batch-kernel benchmark
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algos import warmup_jit
from experiments import (
    run_simulation,
    run_online_benchmark,
    run_kernel_benchmark_with_plots,
    ensure_reproducibility,
)


def main():
    parser = argparse.ArgumentParser(
        description='Run online-learning experiments.')
    parser.add_argument(
        '--experiment',
        choices=['simulation', 'online', 'kernel', 'all'],
        default='all',
        help='Which experiment to run (default: all).')
    parser.add_argument(
        '--data-dir',
        default=os.path.join(os.path.dirname(__file__), 'data'),
        help='Root data directory containing parquet files.')
    parser.add_argument(
        '--results-dir',
        default=os.path.join(os.path.dirname(__file__), 'results'),
        help='Directory to save results.')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility.')
    parser.add_argument(
        '--n-repeats', type=int, default=3,
        help='Number of repeated shuffles for kernel benchmark (default: 3).')
    parser.add_argument(
        '--gamma', default='tune',
        help='RBF gamma for kernel benchmark: "tune", "auto", or a float.')
    parser.add_argument(
        '--max-sv', type=int, default=500,
        help='Maximum support vectors for kernel algorithms (default: 500).')
    args = parser.parse_args()

    # Resolve paths
    data_dir = os.path.abspath(args.data_dir)
    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    ensure_reproducibility(args.seed)

    t0 = time.time()

    # ── Simulation ───────────────────────────────────────────────
    if args.experiment in ('simulation', 'all'):
        print("\n" + "#" * 80)
        print("# EXPERIMENT 1: Simulation (Online vs Kernel Online)")
        print("#" * 80)
        run_simulation(n_samples=2000, seed=args.seed, output_dir=results_dir)

    # ── Online Benchmark ─────────────────────────────────────────
    if args.experiment in ('online', 'all'):
        print("\n" + "#" * 80)
        print("# EXPERIMENT 2: Online Benchmark (MNIST / Kaggle / Cyber)")
        print("#" * 80)
        warmup_jit()
        run_online_benchmark(data_dir, output_dir=results_dir)

    # ── Kernel Benchmark ─────────────────────────────────────────
    if args.experiment in ('kernel', 'all'):
        print("\n" + "#" * 80)
        print("# EXPERIMENT 3: Batch Kernel Benchmark (UCI) with Plots")
        print("#" * 80)
        run_kernel_benchmark_with_plots(
            data_dir, output_dir=results_dir,
            n_repeats=args.n_repeats,
            gamma=args.gamma,
            max_sv=args.max_sv,
        )

    elapsed = time.time() - t0
    print(f"\nAll experiments completed in {elapsed/60:.1f} minutes.")
    print(f"Results saved to: {results_dir}")


if __name__ == '__main__':
    main()
