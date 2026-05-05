#!/usr/bin/env python3
"""
main.py — Entry point for all online regression experiments.

Usage
-----
  python main.py --experiment all        # run EXP1 + EXP2 + EXP3 + combined plot
  python main.py --experiment delayed    # EXP1: variable + fixed delays
  python main.py --experiment batched    # EXP2: pack/batched feedback
  python main.py --experiment corollary  # EXP3: concentrated vs perpetual delay

Options
-------
  --results-dir DIR    Directory for CSV outputs  [default: ./results]
  --max-samples N      Cap stream length (0 = no cap) [default: 50000]
  --seed INT           Random seed  [default: 42]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algos import warmup_jit
from experiments import (
    ensure_reproducibility,
    run_delayed_feedback,
    run_batched_feedback,
    run_corollary_test,
    make_combined_plot,
    print_summary,
)


def main():
    parser = argparse.ArgumentParser(
        description="Online regression experiments: CIRR, K-CIRR, Q-CIRR, QK-CIRR"
    )
    parser.add_argument(
        '--experiment',
        choices=['delayed', 'batched', 'corollary', 'all'],
        default='all',
        help='Which experiment to run (default: all)',
    )
    parser.add_argument(
        '--results-dir',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'),
        help='Directory to save CSV results',
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=50_000,
        help='Max stream length for EXP1/EXP2 (0 = no cap, uses full dataset)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    max_samples = args.max_samples if args.max_samples > 0 else None

    ensure_reproducibility(args.seed)

    # Trigger Numba JIT compilation before any timed experiment
    warmup_jit()

    t0 = time.time()
    df1 = df2 = df3 = None
    cumloss_exp1 = cumloss_exp2 = None
    cor_case1 = cor_case2 = None

    if args.experiment in ('delayed', 'all'):
        print("\n" + "#" * 80)
        print("# EXP1: Delayed Feedback  (CIRR / K-CIRR / Q-CIRR / QK-CIRR)")
        print("#" * 80)
        df1, cumloss_exp1 = run_delayed_feedback(results_dir, max_samples=max_samples)

    if args.experiment in ('batched', 'all'):
        print("\n" + "#" * 80)
        print("# EXP2: Batched Feedback  (CIRR / K-CIRR / Q-CIRR / QK-CIRR)")
        print("#" * 80)
        df2, cumloss_exp2 = run_batched_feedback(results_dir, max_samples=max_samples)

    if args.experiment in ('corollary', 'all'):
        print("\n" + "#" * 80)
        print("# EXP3: Corollary Test  (Concentrated vs Perpetual delay)")
        print("#" * 80)
        df3, cor_case1, cor_case2 = run_corollary_test(results_dir, seed=args.seed)

    if cumloss_exp1 is not None and cumloss_exp2 is not None and cor_case1 is not None:
        print("\n" + "#" * 80)
        print("# COMBINED PLOT for paper")
        print("#" * 80)
        combined = make_combined_plot(
            results_dir, cumloss_exp1, cumloss_exp2, cor_case1, cor_case2
        )
        # Also copy to paper directory
        import shutil
        paper_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', '..', 'book', 'delayed_kernel_quantum'
        )
        paper_dir = os.path.normpath(paper_dir)
        if os.path.isdir(paper_dir):
            shutil.copy2(combined, os.path.join(paper_dir, 'plot_combined.png'))
            print(f"  Copied -> {paper_dir}/plot_combined.png")

    if df1 is not None and df2 is not None:
        print_summary(df1, df2)

    elapsed = time.time() - t0
    print(f"\nAll experiments completed in {elapsed / 60:.1f} minutes.")
    print(f"Results saved to: {results_dir}")


if __name__ == '__main__':
    main()
