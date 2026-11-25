#!/usr/bin/env python3
"""Command-line entry point for the QIKRR PDE experiment suite."""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiments
from experiments import (exp1_accuracy, exp2_online, exp3_spectral,
                         exp4_batch_qae, exp5_scaling, exp6_exact_zz)


def main():
    parser = argparse.ArgumentParser(
        description='QIKRR physics-informed PDE experiments')
    parser.add_argument('--experiment', default='all',
                        choices=['all', 'exp1', 'exp2', 'exp3',
                                 'exp4', 'exp5', 'exp6'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--problem', default='all',
                        choices=['all', 'poisson1d', 'advecdiff1d',
                                 'poisson2d', 'poisson4d'],
                        help='EXP 2 problem; EXP 4/5 use Poisson-1D for all')
    parser.add_argument('--quick', action='store_true',
                        help='small smoke-test configuration; not for paper tables')
    parser.add_argument('--no-resume', action='store_true',
                        help='discard EXP 1/2 checkpoints and start a fresh run')
    parser.add_argument('--results-dir', default=None,
                        help='output directory (default: results; quick: results_quick)')
    args = parser.parse_args()
    if args.results_dir or args.quick:
        default_quick = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'results_quick')
        experiments.RESULTS = os.path.abspath(args.results_dir or default_quick)
        os.makedirs(experiments.RESULTS, exist_ok=True)
    seeds = tuple(args.seeds[:1]) if args.quick else tuple(args.seeds)
    t0 = time.perf_counter()

    if args.experiment in ('all', 'exp1'):
        exp1_accuracy(seeds=seeds, pinn_epochs=300 if args.quick else 4000,
                      resume=not (args.no_resume or args.quick))
    if args.experiment in ('all', 'exp2'):
        exp2_online(problem=args.problem, seeds=seeds,
                    T=60 if args.quick else 200,
                    delay=3 if args.quick else 10,
                    batch=10 if args.quick else 20,
                    n_coll=20 if args.quick else 60,
                    resume=not (args.no_resume or args.quick))
    if args.experiment in ('all', 'exp3'):
        if args.quick:
            exp3_spectral(problem='poisson2d', n=60, n_check=(20, 40, 60))
        else:
            exp3_spectral(problem='poisson2d')
            exp3_spectral(problem='poisson4d', tag='_poisson4d')
    cost_problem = 'poisson1d' if args.problem == 'all' else args.problem
    if args.experiment in ('all', 'exp4'):
        exp4_batch_qae(problem=cost_problem,
                       T=64 if args.quick else 256,
                       batch_grid=(1, 8, 32) if args.quick else (1, 4, 16, 64))
    if args.experiment in ('all', 'exp5'):
        exp5_scaling(problem=cost_problem,
                     n_coll_grid=(30, 60) if args.quick else
                     (60, 120, 240, 480, 960, 1920))
    if args.experiment in ('all', 'exp6'):
        exp6_exact_zz(n=60 if args.quick else 300,
                      n_check=(20, 40, 60) if args.quick else
                      (50, 100, 200, 300))
    print(f'\nDone in {(time.perf_counter() - t0) / 60:.1f} min. '
          f'Results are in {experiments.RESULTS}.')


if __name__ == '__main__':
    main()
