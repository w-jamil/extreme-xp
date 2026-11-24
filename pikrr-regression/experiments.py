#!/usr/bin/env python3
"""Reproducible PDE experiments for the QIKRR revision.

EXP 1 compares solution accuracy, fresh-point PDE residual, CPU time, linear
solves, and modelled gates in a same-kernel factorial design.  QIKRR/PI-IKRR/
PI-KRR share physics-informed hyperparameters; data-only QIKRR/IKRR/KRR share
KRR-selected hyperparameters.  EXP 2 studies causal online feedback; EXP 3 and
EXP 6 study spectra; EXP 4 studies measurement and solve-count models; EXP 5 compares
measured classical CPU time with modelled quantum gate counts.

No quantum hardware is used.  RFF and HHL results are simulations; exact ZZ
refers to closed-form kernel evaluation, not execution of a quantum circuit.
"""

import itertools
import json
import os
import time
import numpy as np
import pandas as pd

from pde import make_problems
from kernels import (QuantumKernel, RBFKernel, ZZKernel, build_pi_system)
from algos import (QIKRR, PIKRR, PIIKRR, IKRR, KRR, PINN, HHLSimulator,
                   CausalFullRefit)

RESULTS = os.environ.get(
    'QIKRR_RESULTS_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'))
os.makedirs(RESULTS, exist_ok=True)
N_REWEIGHT = 2


def _save(df, name):
    path = os.path.join(RESULTS, name)
    df.to_csv(path, index=False)
    print(f'  CSV -> {path}')
    return df


RFF_GRID = dict(n_qubits=[8], length_scale=[0.3, 0.5, 1.0, 2.0, 4.0],
                beta=[0.01, 0.03, 0.1], lam=[1e-6, 1e-4, 1e-2],
                pde_weight=[1.0, 10.0])
RBF_GRID = dict(length_scale=[0.3, 0.5, 1.0], lam=[1e-6, 1e-4, 1e-2],
                pde_weight=[1.0, 10.0])
ZZ_GRID_BY_DIM = {1: [4, 8], 2: [2, 4, 8], 4: [4, 8]}
ZZ_LAM = [1e-6, 1e-4]
ZZ_W = [1.0, 10.0, 100.0]


def _construct(kind, pde, seed, **kw):
    common = dict(seed=seed, **kw)
    if kind == 'pikrr_rff':
        return PIKRR(pde, kernel_kind='rff', **common)
    if kind == 'pikrr_rbf':
        return PIKRR(pde, kernel_kind='rbf', **common)
    if kind == 'pikrr_zz':
        return PIKRR(pde, kernel_kind='zz', **common)
    if kind == 'krr_rbf':
        return KRR(pde, kernel_kind='rbf',
                   **{k: v for k, v in common.items()
                      if k not in ('pde_weight', 'seed')})
    if kind == 'krr_rff':
        return KRR(pde, kernel_kind='rff',
                   **{k: v for k, v in common.items()
                      if k not in ('pde_weight', 'seed')})
    if kind == 'krr_zz':
        return KRR(pde, kernel_kind='zz',
                   **{k: v for k, v in common.items()
                      if k not in ('pde_weight', 'seed')})
    raise ValueError(kind)


def _tune(pde, Xd, yd, Xc, gc, kind, grid, seed):
    """Select hyperparameters without using the test grid.

    Classical PI-KRR selects settings shared by QIKRR, PI-IKRR, and PI-KRR.
    Data-only KRR is tuned only on validation solution error, and its settings
    are shared by data-only QIKRR, IKRR, and KRR; physics does not enter this
    model selection.
    """
    best, best_err = None, np.inf
    n_tr = max(4, int(0.8 * len(Xd)))
    Xtr, ytr, Xva, yva = Xd[:n_tr], yd[:n_tr], Xd[n_tr:], yd[n_tr:]
    rng = np.random.RandomState(seed + 999)
    Xv_c, gv_c = pde.sample_collocation(200, rng)
    g_scale = np.sqrt(np.mean(gv_c ** 2)) + 1e-12
    for vals in itertools.product(*grid.values()):
        kw = dict(zip(grid.keys(), vals))
        try:
            model = _construct(kind, pde, seed, **kw)
            model.fit(Xtr, ytr, Xc, gc)
            err = (np.linalg.norm(model.predict(Xva) - yva)
                   / (np.linalg.norm(yva) + 1e-12))
            if not kind.startswith('krr') and len(Xv_c):
                err += np.sqrt(np.mean((model.residual(Xv_c) - gv_c) ** 2)) / g_scale
            coef = getattr(model, 'coef_', None)
            if coef is None:
                coef = getattr(model, 'alpha_', np.zeros(1))
            if not np.isfinite(err) or np.linalg.norm(coef) > 1e4:
                continue
        except (np.linalg.LinAlgError, NotImplementedError, ValueError):
            continue
        if err < best_err:
            best, best_err = kw, err
    if best is None:
        raise RuntimeError(f'No stable hyperparameters for {kind} on {pde.name}')
    return best, best_err


def _metrics(model, method, problem, seed, Xt, yt, Xfresh, gfresh,
             total_time, build_time=0.0):
    pred = model.predict(Xt)
    res = model.residual(Xfresh) - gfresh
    return dict(problem=problem, method=method, seed=seed,
                rel_l2=np.linalg.norm(pred - yt) / (np.linalg.norm(yt) + 1e-12),
                res_rmse=np.sqrt(np.mean(res ** 2)),
                time_sec=total_time, build_sec=build_time,
                solve_sec=getattr(model, 'fit_time', np.nan),
                linear_solves=getattr(model, 'linear_solves', 0),
                modelled_gates=getattr(model, 'modelled_gates', 0.0),
                n_reweight=getattr(model, 'n_reweight', 1))


def exp1_accuracy(seeds=(0, 1, 2, 3, 4), noise_std=0.02,
                  n_in=20, n_bound=16, n_coll=120, n_test=900,
                  pinn_epochs=4000, resume=True):
    print('\n' + '=' * 76 + '\nEXP 1: factorial accuracy against PI-KRR and PI-IKRR benchmarks\n' + '=' * 76)
    checkpoint = os.path.join(RESULTS, 'exp1_checkpoint.csv')
    hp_checkpoint = os.path.join(RESULTS, 'exp1_hyperparams_checkpoint.csv')
    if resume and os.path.exists(checkpoint):
        rows = pd.read_csv(checkpoint).to_dict('records')
        hp_rows = (pd.read_csv(hp_checkpoint).to_dict('records')
                   if os.path.exists(hp_checkpoint) else [])
    else:
        rows, hp_rows = [], []
    completed = {(r['problem'], int(r['seed'])) for r in rows}
    for key, pde in make_problems().items():
        for seed in seeds:
            if (key, seed) in completed:
                print(f'  {key} seed {seed} resumed from checkpoint')
                continue
            rng = np.random.RandomState(seed)
            Xd, yd = pde.sample_data(n_in, noise_std, rng, n_boundary=n_bound)
            Xc, gc = pde.sample_collocation(n_coll, rng)
            Xt, yt = pde.test_grid(n_test)
            Xfresh, gfresh = pde.sample_collocation(400, rng)

            # Matched hyperparameters for each same-kernel factorial family.
            rff_hp, _ = _tune(pde, Xd, yd, Xc, gc, 'pikrr_rff', RFF_GRID, seed)
            rbf_hp, _ = _tune(pde, Xd, yd, Xc, gc, 'pikrr_rbf', RBF_GRID, seed)
            zz_grid = dict(n_qubits=ZZ_GRID_BY_DIM[pde.dim], lam=ZZ_LAM,
                           pde_weight=ZZ_W)
            zz_hp, _ = _tune(pde, Xd, yd, Xc, gc, 'pikrr_zz', zz_grid, seed)
            krr_rbf_hp, _ = _tune(pde, Xd, yd, Xc, gc, 'krr_rbf', RBF_GRID, seed)
            krr_rff_hp, _ = _tune(pde, Xd, yd, Xc, gc, 'krr_rff', RFF_GRID, seed)
            zz_krr_grid = dict(n_qubits=ZZ_GRID_BY_DIM[pde.dim], lam=ZZ_LAM)
            krr_zz_hp, _ = _tune(pde, Xd, yd, Xc, gc, 'krr_zz', zz_krr_grid, seed)

            specs = [
                ('QIKRR (exact ZZ)',
                 lambda: QIKRR(pde, kernel_kind='zz', n_reweight=N_REWEIGHT,
                               seed=seed, **zz_hp), True),
                ('PI-IKRR (exact ZZ)',
                 lambda: PIIKRR(pde, kernel_kind='zz', n_reweight=N_REWEIGHT,
                                seed=seed, **zz_hp), True),
                ('PI-KRR (exact ZZ)',
                 lambda: PIKRR(pde, kernel_kind='zz', seed=seed, **zz_hp), True),
                ('QIKRR (RBF)',
                 lambda: QIKRR(pde, kernel_kind='rbf', n_reweight=N_REWEIGHT,
                               seed=seed, **rbf_hp), True),
                ('PI-IKRR (RBF)',
                 lambda: PIIKRR(pde, kernel_kind='rbf', n_reweight=N_REWEIGHT,
                                seed=seed, **rbf_hp), True),
                ('PI-KRR (RBF)',
                 lambda: PIKRR(pde, kernel_kind='rbf', seed=seed, **rbf_hp), True),
                ('QIKRR (RFF sim.)',
                 lambda: QIKRR(pde, kernel_kind='rff', n_reweight=N_REWEIGHT,
                               seed=seed, **rff_hp), True),
                ('PI-IKRR (RFF sim.)',
                 lambda: PIIKRR(pde, kernel_kind='rff', n_reweight=N_REWEIGHT,
                                seed=seed, **rff_hp), True),
                ('PI-KRR (RFF sim.)',
                 lambda: PIKRR(pde, kernel_kind='rff', seed=seed, **rff_hp), True),
                ('QIKRR (RFF, no phys)',
                 lambda: QIKRR(pde, kernel_kind='rff', n_reweight=N_REWEIGHT,
                               seed=seed, **krr_rff_hp), False),
                ('IKRR (RFF, no phys)',
                 lambda: IKRR(pde, kernel_kind='rff', n_reweight=N_REWEIGHT,
                              seed=seed, **krr_rff_hp), False),
                ('KRR (RFF, no phys)',
                 lambda: KRR(pde, kernel_kind='rff', seed=seed,
                             **{k: v for k, v in krr_rff_hp.items()
                                if k != 'pde_weight'}), False),
                ('IKRR (exact ZZ, no phys)',
                 lambda: IKRR(pde, kernel_kind='zz', n_reweight=N_REWEIGHT,
                              seed=seed, **krr_zz_hp), False),
                ('KRR (exact ZZ, no phys)',
                 lambda: KRR(pde, kernel_kind='zz', seed=seed, **krr_zz_hp), False),
                ('IKRR (RBF, no phys)',
                 lambda: IKRR(pde, kernel_kind='rbf', n_reweight=N_REWEIGHT,
                              seed=seed, **krr_rbf_hp), False),
                ('KRR (RBF, no phys)',
                 lambda: KRR(pde, kernel_kind='rbf', seed=seed,
                             **{k: v for k, v in krr_rbf_hp.items()
                                if k != 'pde_weight'}), False),
            ]
            for method, factory, use_physics in specs:
                model = factory()
                t0 = time.perf_counter()
                if isinstance(model, KRR):
                    model.fit(Xd, yd)
                    build_time = 0.0
                elif use_physics:
                    model.fit(Xd, yd, Xc, gc)
                    build_time = model.build_time
                else:
                    model.fit(Xd, yd, Xd[:0], np.zeros(0))
                    build_time = model.build_time
                elapsed = time.perf_counter() - t0
                rows.append(_metrics(model, method, key, seed, Xt, yt,
                                     Xfresh, gfresh, elapsed, build_time))

            pinn = PINN(pde, seed=seed)
            t0 = time.perf_counter()
            pinn.fit(Xd, yd, Xc, gc, n_epochs=pinn_epochs)
            elapsed = time.perf_counter() - t0
            rows.append(_metrics(pinn, 'PINN', key, seed, Xt, yt,
                                 Xfresh, gfresh, elapsed))
            hp_rows.append(dict(problem=key, seed=seed,
                                rff_shared=json.dumps(rff_hp),
                                rbf_shared=json.dumps(rbf_hp),
                                zz_shared=json.dumps(zz_hp),
                                krr_rbf=json.dumps(krr_rbf_hp),
                                krr_rff=json.dumps(krr_rff_hp),
                                krr_zz=json.dumps(krr_zz_hp)))
            print(f'  {key} seed {seed} done')
            # Persist complete problem/seed units only.  A timed-out run can
            # therefore resume without mixing partial or duplicate results.
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
            pd.DataFrame(hp_rows).to_csv(hp_checkpoint, index=False)

    df = pd.DataFrame(rows)
    _save(df, 'exp1_pde_accuracy.csv')
    _save(pd.DataFrame(hp_rows), 'exp1_hyperparams.csv')
    agg = (df.groupby(['problem', 'method'], sort=False)
             .agg(rel_mean=('rel_l2', 'mean'), rel_std=('rel_l2', 'std'),
                  res_mean=('res_rmse', 'mean'), res_std=('res_rmse', 'std'),
                  t_mean=('time_sec', 'mean'), t_std=('time_sec', 'std'),
                  solves_mean=('linear_solves', 'mean'),
                  gates_mean=('modelled_gates', 'mean')).reset_index())
    _save(agg, 'exp1_pde_accuracy_summary.csv')

    kernel_methods = {
        'exact ZZ': dict(q='QIKRR (exact ZZ)', pii='PI-IKRR (exact ZZ)',
                         pi='PI-KRR (exact ZZ)', i='IKRR (exact ZZ, no phys)',
                         k='KRR (exact ZZ, no phys)'),
        'RBF': dict(q='QIKRR (RBF)', pii='PI-IKRR (RBF)',
                    pi='PI-KRR (RBF)', i='IKRR (RBF, no phys)',
                    k='KRR (RBF, no phys)'),
        'RFF': dict(q='QIKRR (RFF sim.)', pii='PI-IKRR (RFF sim.)',
                    pi='PI-KRR (RFF sim.)', i='IKRR (RFF, no phys)',
                    k='KRR (RFF, no phys)')}
    pairs = []
    for kernel, names in kernel_methods.items():
        pairs.extend([
            ('solver with physics', kernel, names['q'], names['pii']),
            ('iteration with physics', kernel, names['pii'], names['pi']),
            ('iteration without physics', kernel, names['i'], names['k']),
            ('physics without iteration', kernel, names['pi'], names['k']),
            ('physics with iteration', kernel, names['pii'], names['i']),
            ('overall QIKRR vs PI-KRR', kernel, names['q'], names['pi'])])
    pairs.append(('solver without physics', 'RFF',
                  'QIKRR (RFF, no phys)', 'IKRR (RFF, no phys)'))
    comparison = []
    for factor, kernel, method_name, benchmark_name in pairs:
        for (problem, seed), group in df.groupby(['problem', 'seed']):
            method_row = group[group.method == method_name].iloc[0]
            bench_row = group[group.method == benchmark_name].iloc[0]
            comparison.append(dict(
                comparison=factor, kernel=kernel, problem=problem, seed=seed,
                method=method_name, benchmark=benchmark_name,
                rel_l2_method=method_row.rel_l2,
                rel_l2_benchmark=bench_row.rel_l2,
                rel_l2_ratio=method_row.rel_l2 / max(bench_row.rel_l2, 1e-12),
                method_accuracy_win=method_row.rel_l2 < bench_row.rel_l2,
                res_method=method_row.res_rmse, res_benchmark=bench_row.res_rmse,
                res_ratio=method_row.res_rmse / max(bench_row.res_rmse, 1e-12),
                method_residual_win=method_row.res_rmse < bench_row.res_rmse,
                time_method=method_row.time_sec, time_benchmark=bench_row.time_sec,
                time_ratio_method_over_benchmark=(
                    method_row.time_sec / max(bench_row.time_sec, 1e-12)),
                modelled_gates_method=method_row.modelled_gates))
    comp = pd.DataFrame(comparison)
    _save(comp, 'exp1_factorial_comparisons.csv')
    print(agg.to_string(index=False))
    print('\nFactorial comparisons (method / matched benchmark):')
    print(comp.groupby(['comparison', 'kernel'])[
        ['rel_l2_ratio', 'res_ratio', 'time_ratio_method_over_benchmark',
         'method_accuracy_win', 'method_residual_win']].mean().to_string())
    return df


def exp2_online(problem='all', T=200, seeds=(0, 1, 2, 3, 4),
                noise_std=0.02, modes=('immediate', 'delayed', 'batched'),
                delay=10, batch=20, n_coll=60, n_test=900, resume=True):
    """Matched sequential factorial over four PDEs and three kernels.

    A fixed collocation set is available before each stream for the three
    physics-informed methods.  Data targets arrive causally and only released
    targets enter a full refit.  Every kernel has independently calibrated
    physics-informed and data-only hyperparameters, shared respectively by
    QIKRR/PI-IKRR/PI-KRR and QIKRR/IKRR/KRR.  No evaluated-stream target enters
    model selection, the clip bound, or the causal updates.
    """
    print('\n' + '=' * 76 +
          '\nEXP 2: four-PDE, three-kernel causal sequential factorial\n' +
          '=' * 76)
    all_problems = make_problems()
    problems = list(all_problems) if problem == 'all' else [problem]
    checkpoint = os.path.join(RESULTS, 'exp2_checkpoint.csv')
    curve_checkpoint = os.path.join(RESULTS, 'exp2_curves_checkpoint.csv')
    hp_checkpoint = os.path.join(RESULTS, 'exp2_hyperparams_checkpoint.csv')
    if resume and os.path.exists(checkpoint):
        rows = pd.read_csv(checkpoint).to_dict('records')
        curve_rows = pd.read_csv(curve_checkpoint).to_dict('records')
        hp_rows = pd.read_csv(hp_checkpoint).to_dict('records')
    else:
        rows, curve_rows, hp_rows = [], [], []
    completed = {(r['problem'], int(r['seed'])) for r in rows}

    for problem in problems:
        pde = all_problems[problem]
        Xt, yt = pde.test_grid(n_test)
        kernel_configs = {
            'RFF': dict(
                kind='rff', short='rff', tune_pi='pikrr_rff',
                tune_krr='krr_rff',
                pi_grid=dict(n_qubits=[8], length_scale=[0.5, 1.0, 2.0],
                             beta=[0.01, 0.03, 0.1], lam=[1e-6, 1e-4],
                             pde_weight=[1.0, 10.0])),
            'RBF': dict(
                kind='rbf', short='rbf', tune_pi='pikrr_rbf',
                tune_krr='krr_rbf',
                pi_grid=dict(length_scale=[0.3, 0.5, 1.0],
                             lam=[1e-6, 1e-4], pde_weight=[1.0, 10.0])),
            'exact ZZ': dict(
                kind='zz', short='zz', tune_pi='pikrr_zz',
                tune_krr='krr_zz',
                pi_grid=dict(n_qubits=ZZ_GRID_BY_DIM[pde.dim], lam=ZZ_LAM,
                             pde_weight=ZZ_W)),
        }
        for cfg in kernel_configs.values():
            cfg['krr_grid'] = dict(cfg['pi_grid'], pde_weight=[1.0])

        for seed in seeds:
            if (problem, seed) in completed:
                print(f'  {problem} seed {seed} resumed from checkpoint')
                continue
            rng = np.random.RandomState(seed)
            lo, hi = pde.domain[:, 0], pde.domain[:, 1]
            X = lo + (hi - lo) * rng.random((T, pde.dim))
            y = pde.solution(X) + noise_std * rng.standard_normal(T)
            Xc, gc = pde.sample_collocation(n_coll, rng)
            Xfresh, gfresh = pde.sample_collocation(400, rng)

            # Independent calibration prevents evaluated-stream leakage.
            tune_rng = np.random.RandomState(seed + 10000)
            X_tune = lo + (hi - lo) * tune_rng.random((T, pde.dim))
            y_tune = (pde.solution(X_tune) +
                      noise_std * tune_rng.standard_normal(T))
            Xc_tune, gc_tune = pde.sample_collocation(n_coll, tune_rng)
            Y = max(1.0, 1.2 * np.abs(y_tune).max())
            tuned = {}
            for kernel, cfg in kernel_configs.items():
                pi_hp, _ = _tune(pde, X_tune, y_tune, Xc_tune, gc_tune,
                                 cfg['tune_pi'], cfg['pi_grid'], seed)
                krr_hp, _ = _tune(pde, X_tune, y_tune, Xc_tune, gc_tune,
                                  cfg['tune_krr'], cfg['krr_grid'], seed)
                tuned[kernel] = (pi_hp, krr_hp)

            for kernel, cfg in kernel_configs.items():
                kind = cfg['kind']
                pi_hp, krr_hp = tuned[kernel]
                # Same-kernel hindsight references with matching physics access.
                pi_ref = PIKRR(pde, kernel_kind=kind, seed=seed, **pi_hp)
                pi_ref.fit(X, y, Xc, gc)
                pi_comp = float(np.sum((pi_ref.predict(X) - y) ** 2))
                krr_ref = KRR(
                    pde, kernel_kind=kind, seed=seed,
                    **{k: v for k, v in krr_hp.items() if k != 'pde_weight'})
                krr_ref.fit(X, y)
                krr_comp = float(np.sum((krr_ref.predict(X) - y) ** 2))

                specs = [
                    (f'QIKRR ({kernel})',
                     lambda kind=kind, hp=pi_hp: QIKRR(
                         pde, kernel_kind=kind, seed=seed,
                         n_reweight=N_REWEIGHT, **hp), True),
                    (f'PI-IKRR ({kernel})',
                     lambda kind=kind, hp=pi_hp: PIIKRR(
                         pde, kernel_kind=kind, seed=seed,
                         n_reweight=N_REWEIGHT, **hp), True),
                    (f'PI-KRR ({kernel})',
                     lambda kind=kind, hp=pi_hp: PIKRR(
                         pde, kernel_kind=kind, seed=seed, **hp), True),
                    (f'QIKRR ({kernel}, no phys)',
                     lambda kind=kind, hp=krr_hp: QIKRR(
                         pde, kernel_kind=kind, seed=seed,
                         n_reweight=N_REWEIGHT, **hp), False),
                    (f'IKRR ({kernel}, no phys)',
                     lambda kind=kind, hp=krr_hp: IKRR(
                         pde, kernel_kind=kind, seed=seed,
                         n_reweight=N_REWEIGHT, **hp), False),
                    (f'KRR ({kernel}, no phys)',
                     lambda kind=kind, hp=krr_hp: KRR(
                         pde, kernel_kind=kind, seed=seed,
                         **{k: v for k, v in hp.items()
                            if k != 'pde_weight'}), False),
                ]
                for method, factory, use_physics in specs:
                    for mode in modes:
                        model = factory()
                        learner = CausalFullRefit(
                            model,
                            Xc if use_physics else np.zeros((0, pde.dim)),
                            gc if use_physics else np.zeros(0), clip=Y)
                        mkw = (dict(delay=delay) if mode == 'delayed' else
                               dict(batch_size=batch) if mode == 'batched' else {})
                        preds = np.zeros(T)
                        t0 = time.perf_counter()
                        for t in range(T):
                            preds[t] = learner.start_round(X[t])
                            learner.observe(X[t], y[t], mode=mode, **mkw)
                        learner.flush()
                        elapsed = time.perf_counter() - t0
                        cum = np.cumsum((preds - y) ** 2)
                        curve_rows.extend(dict(
                            problem=problem, kernel=kernel, method=method,
                            mode=mode, seed=seed, t=t + 1, cum=cum[t])
                            for t in range(T))
                        comp = pi_comp if use_physics else krr_comp
                        final_pred = model.predict(Xt)
                        final_res = model.residual(Xfresh) - gfresh
                        hp = pi_hp if use_physics else krr_hp
                        rows.append(dict(
                            problem=problem, kernel=kernel, method=method,
                            mode=mode, seed=seed, physics=use_physics,
                            cum_loss=cum[-1], comp_loss=comp,
                            regret=cum[-1] - comp,
                            rel_l2=(np.linalg.norm(final_pred - yt) /
                                    (np.linalg.norm(yt) + 1e-12)),
                            res_rmse=np.sqrt(np.mean(final_res ** 2)),
                            time_sec=elapsed, refits=learner.refits,
                            linear_solves=learner.linear_solves,
                            modelled_gates=learner.modelled_gates,
                            hp=json.dumps(hp)))
                        print(f'  {problem} seed {seed} {method:31s} '
                              f'{mode:10s} loss {cum[-1]:8.3f} '
                              f'gap {cum[-1] - comp:8.3f} '
                              f'refits {learner.refits:3d}')
            hp_record = dict(problem=problem, seed=seed)
            for kernel, (pi_hp, krr_hp) in tuned.items():
                short = kernel_configs[kernel]['short']
                hp_record[f'{short}_physics_shared'] = json.dumps(pi_hp)
                hp_record[f'{short}_data_shared'] = json.dumps(krr_hp)
            hp_rows.append(hp_record)
            # Persist complete problem/seed units; partial runs are not resumed.
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
            pd.DataFrame(curve_rows).to_csv(curve_checkpoint, index=False)
            pd.DataFrame(hp_rows).to_csv(hp_checkpoint, index=False)

    df = pd.DataFrame(rows)
    curves = pd.DataFrame(curve_rows)
    _save(df, 'exp2_online_regret.csv')
    _save(curves, 'exp2_cumcurves.csv')
    _save(pd.DataFrame(hp_rows), 'exp2_hyperparams.csv')
    agg = (df.groupby(['problem', 'kernel', 'method', 'mode'], sort=False)
             .agg(cum_mean=('cum_loss', 'mean'), cum_std=('cum_loss', 'std'),
                  regret_mean=('regret', 'mean'), regret_std=('regret', 'std'),
                  rel_mean=('rel_l2', 'mean'), rel_std=('rel_l2', 'std'),
                  res_mean=('res_rmse', 'mean'), res_std=('res_rmse', 'std'),
                  time_mean=('time_sec', 'mean'),
                  refits_mean=('refits', 'mean'),
                  solves_mean=('linear_solves', 'mean')).reset_index())
    _save(agg, 'exp2_online_regret_summary.csv')

    pairs = []
    for kernel in ('RFF', 'RBF', 'exact ZZ'):
        pairs.extend([
            ('solver with physics', kernel,
             f'QIKRR ({kernel})', f'PI-IKRR ({kernel})'),
            ('iteration with physics', kernel,
             f'PI-IKRR ({kernel})', f'PI-KRR ({kernel})'),
            ('iteration without physics', kernel,
             f'IKRR ({kernel}, no phys)', f'KRR ({kernel}, no phys)'),
            ('physics without iteration', kernel,
             f'PI-KRR ({kernel})', f'KRR ({kernel}, no phys)'),
            ('physics with iteration', kernel,
             f'PI-IKRR ({kernel})', f'IKRR ({kernel}, no phys)'),
            ('overall QIKRR vs PI-KRR', kernel,
             f'QIKRR ({kernel})', f'PI-KRR ({kernel})'),
            ('solver without physics', kernel,
             f'QIKRR ({kernel}, no phys)', f'IKRR ({kernel}, no phys)'),
        ])
    comparison = []
    for factor, kernel, method_name, benchmark_name in pairs:
        kernel_df = df[df.kernel == kernel]
        for (problem, mode, seed), group in kernel_df.groupby(
                ['problem', 'mode', 'seed']):
            method_row = group[group.method == method_name].iloc[0]
            bench_row = group[group.method == benchmark_name].iloc[0]
            comparison.append(dict(
                comparison=factor, kernel=kernel, problem=problem, mode=mode,
                seed=seed, method=method_name, benchmark=benchmark_name,
                loss_method=method_row.cum_loss,
                loss_benchmark=bench_row.cum_loss,
                loss_ratio=method_row.cum_loss / max(bench_row.cum_loss, 1e-12),
                method_loss_win=method_row.cum_loss < bench_row.cum_loss,
                regret_method=method_row.regret,
                regret_benchmark=bench_row.regret,
                rel_l2_method=method_row.rel_l2,
                rel_l2_benchmark=bench_row.rel_l2,
                rel_l2_ratio=method_row.rel_l2 / max(bench_row.rel_l2, 1e-12),
                method_accuracy_win=method_row.rel_l2 < bench_row.rel_l2,
                res_method=method_row.res_rmse,
                res_benchmark=bench_row.res_rmse,
                res_ratio=method_row.res_rmse / max(bench_row.res_rmse, 1e-12),
                method_residual_win=method_row.res_rmse < bench_row.res_rmse,
                time_method=method_row.time_sec,
                time_benchmark=bench_row.time_sec,
                time_ratio_method_over_benchmark=(
                    method_row.time_sec / max(bench_row.time_sec, 1e-12)),
                refits_method=method_row.refits,
                refits_benchmark=bench_row.refits,
                solves_method=method_row.linear_solves,
                solves_benchmark=bench_row.linear_solves,
                modelled_gates_method=method_row.modelled_gates))
    comp = pd.DataFrame(comparison)
    _save(comp, 'exp2_factorial_comparisons.csv')
    print(agg.to_string(index=False))
    print('\nSequential factorial comparisons (method / matched benchmark):')
    print(comp.groupby(['comparison', 'kernel', 'mode'])[
        ['loss_ratio', 'rel_l2_ratio', 'res_ratio',
         'time_ratio_method_over_benchmark', 'method_loss_win',
         'method_accuracy_win', 'method_residual_win']].mean().to_string())
    return df


def exp3_spectral(problem='poisson2d', n=300,
                  n_check=(20, 50, 100, 200, 300), n_qubits=8,
                  length_scale=1.0, beta=0.1, rbf_ls=0.5,
                  matern_nu=1.0, matern_ls=0.5, tag=''):
    """Compare Gram spectra; this does not establish realised-regret bounds."""
    print('\n' + '=' * 76 + '\nEXP 3: spectra and effective dimension\n' + '=' * 76)
    from sklearn.gaussian_process.kernels import Matern
    pde = make_problems()[problem]
    rng = np.random.RandomState(7)
    lo, hi = pde.domain[:, 0], pde.domain[:, 1]
    X = lo + (hi - lo) * rng.random((n, pde.dim))
    lam = 1e-4
    qk = QuantumKernel(n_qubits=n_qubits, length_scale=length_scale,
                       beta=beta, operator_terms=pde.operator_terms)
    rk = RBFKernel(length_scale=rbf_ls, operator_terms=pde.operator_terms)
    rows = []
    for t in n_check:
        Xp = X[:t]
        eq = np.linalg.eigvalsh(qk.K(Xp, Xp))[::-1]
        er = np.linalg.eigvalsh(rk.K(Xp, Xp))[::-1]
        em = np.linalg.eigvalsh(Matern(nu=matern_nu,
                                      length_scale=matern_ls)(Xp))[::-1]
        dq = float(np.sum(eq / (eq + lam)))
        dr = float(np.sum(er / (er + lam)))
        dm = float(np.sum(em / (em + lam)))
        rows.append(dict(t=t, d_eff_qikrr_rff=dq, d_eff_rbf=dr,
                         d_eff_matern=dm))
        print(f'  t={t:4d}  d_eff RFF={dq:8.2f} RBF={dr:8.2f} Mat={dm:8.2f}')
    df = pd.DataFrame(rows)
    _save(df, f'exp3_effective_dimension{tag}.csv')
    spec = pd.DataFrame(dict(
        idx=np.arange(1, n + 1),
        eig_qikrr_rff=np.linalg.eigvalsh(qk.K(X, X))[::-1],
        eig_rbf=np.linalg.eigvalsh(rk.K(X, X))[::-1],
        eig_matern=np.linalg.eigvalsh(Matern(nu=matern_nu,
                                            length_scale=matern_ls)(X))[::-1]))
    _save(spec, f'exp3_spectra{tag}.csv')
    return df


def exp6_exact_zz(problem='poisson2d', n=300,
                  n_check=(50, 100, 200, 300), lam=1e-4, seed=7):
    """Verify finite support of the exact commuting ZZ kernel spectrum."""
    print('\n' + '=' * 76 + '\nEXP 6: exact commuting ZZ spectrum\n' + '=' * 76)
    pde = make_problems()[problem]
    rng = np.random.RandomState(seed)
    lo, hi = pde.domain[:, 0], pde.domain[:, 1]
    X = lo + (hi - lo) * rng.random((n, pde.dim))
    rows, spectra = [], {}
    for nq in (2, 4, 6, 8):
        if nq % pde.dim:
            continue
        zk = ZZKernel(n_qubits=nq, input_dim=pde.dim)
        ev = np.sort(np.linalg.eigvalsh(zk.K(X, X)))[::-1]
        spectra[nq] = ev
        for t in n_check:
            evt = np.sort(np.linalg.eigvalsh(zk.K(X[:t], X[:t])))[::-1]
            de = float(np.sum(evt / (evt + lam)))
            rows.append(dict(n_qubits=nq, t=t, d_eff=de,
                             rank_bound=(2 * (nq // pde.dim) + 1) ** pde.dim))
            print(f'  n={nq} t={t:4d} d_eff={de:8.2f}')
    deff = pd.DataFrame(rows)
    _save(deff, 'exp6_zz_deff.csv')
    sp = pd.DataFrame({'idx': np.arange(1, n + 1),
                       **{f'eig_zz_n{nq}': ev for nq, ev in spectra.items()}})
    _save(sp, 'exp6_zz_spectra.csv')
    return deff


def exp5_scaling(problem='poisson1d',
                 n_coll_grid=(60, 120, 240, 480, 960, 1920),
                 n_data=20, n_bound=16, lam=1e-4, hhl_eps=0.01,
                 n_reweight=N_REWEIGHT, seed=0):
    """Measure classical CPU time and separately evaluate quantum gate models.

    Seconds and gate counts are different units and are never divided to form
    a claimed hardware speedup.  Both quantum models include ``n_reweight``
    solves per refit.
    """
    print('\n' + '=' * 76 + '\nEXP 5: classical time and modelled quantum cost\n' + '=' * 76)
    pde = make_problems()[problem]
    rng = np.random.RandomState(seed)
    rows = []
    for n_coll in n_coll_grid:
        Xd, yd = pde.sample_data(n_data, 0.0, rng, n_boundary=n_bound)
        Xc, gc = pde.sample_collocation(n_coll, rng)
        rk = RBFKernel(length_scale=0.5, operator_terms=pde.operator_terms)
        t0 = time.perf_counter()
        A, rhs = build_pi_system(rk, Xd, yd, Xc, gc, lam)
        build_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        np.linalg.solve(A, rhs)
        solve_s = time.perf_counter() - t0

        # Estimate the 2-norm condition number through normal-equation power
        # iterations.  The Kansa matrix is nonsymmetric, so iterating A alone
        # would estimate an eigenvalue magnitude rather than a singular value.
        v = rng.standard_normal(A.shape[0])
        v /= np.linalg.norm(v)
        for _ in range(60):
            v = A.T @ (A @ v)
            v /= max(np.linalg.norm(v), 1e-300)
        lam_max = float(np.linalg.norm(A @ v))
        w = rng.standard_normal(A.shape[0])
        w /= np.linalg.norm(w)
        for _ in range(8):
            w = np.linalg.solve(A.T, np.linalg.solve(A, w))
            w /= max(np.linalg.norm(w), 1e-300)
        Aw = np.linalg.solve(A, w)
        lam_min = 1.0 / max(float(np.linalg.norm(Aw)), 1e-300)
        kappa = lam_max / max(lam_min, 1e-300)
        N = A.shape[0]
        hhl_one = np.log2(max(N, 2)) * kappa ** 2 * np.log2(1.0 / hhl_eps)
        qlsa_one = kappa * np.log2(max(kappa, 2)) ** 3 * np.log2(max(N, 2))
        rows.append(dict(N=N, n_coll=n_coll, classical_build_s=build_s,
                         classical_solve_s=solve_s, kappa=kappa,
                         lam_max=lam_max, lam_min=lam_min,
                         n_reweight=n_reweight,
                         modelled_hhl_gates=n_reweight * hhl_one,
                         modelled_qlsa_gates=n_reweight * qlsa_one))
        print(f'  N={N:5d} build {build_s:6.3f}s solve {solve_s:7.4f}s '
              f'kappa {kappa:9.2e} QIKRR-QLSA gates {n_reweight * qlsa_one:9.3e}')
    df = pd.DataFrame(rows)
    _save(df, 'exp5_scaling.csv')
    return df


def exp4_batch_qae(problem='poisson1d', T=256,
                   n_shots_grid=(1, 4, 16, 64, 256),
                   batch_grid=(1, 4, 16, 64), seed=0):
    """Synthetic error-versus-query scaling and causal batching solve counts."""
    print('\n' + '=' * 76 + '\nEXP 4: estimation and batching cost models\n' + '=' * 76)
    pde = make_problems()[problem]
    rng = np.random.RandomState(seed)
    Xd, yd = pde.sample_data(20, 0.02, rng, n_boundary=16)
    Xc, gc = pde.sample_collocation(120, rng)
    qk = QIKRR(pde, kernel_kind='rff', n_qubits=8, length_scale=1.0,
               beta=0.1, lam=1e-4, n_reweight=N_REWEIGHT, seed=seed)
    qk.fit(Xd, yd, Xc, gc)
    lo, hi = pde.domain[:, 0], pde.domain[:, 1]
    X = lo + (hi - lo) * rng.random((T, pde.dim))
    p_true = qk.predict(X)

    rows = []
    for S in n_shots_grid:
        direct = p_true + rng.standard_normal(T) / (2 * np.sqrt(S))
        rows.append(dict(protocol='direct Hadamard sampling',
                         budget_per_est=S,
                         rmse=float(np.sqrt(np.mean((direct - p_true) ** 2))),
                         total_queries=T * S))
    for S in n_shots_grid:
        qae = p_true + rng.standard_normal(T) / S
        rows.append(dict(protocol='QAE scaling model', budget_per_est=S,
                         rmse=float(np.sqrt(np.mean((qae - p_true) ** 2))),
                         total_queries=T * S))
    df = pd.DataFrame(rows)
    _save(df, 'exp4_error_vs_budget.csv')

    rows_b = []
    targets = pde.solution(X)
    for B in batch_grid:
        learner = QIKRR(pde, kernel_kind='rff', n_qubits=8, length_scale=1.0,
                        beta=0.1, lam=1e-4, n_reweight=N_REWEIGHT, seed=seed)
        for t in range(T):
            learner.start_round(X[t])
            learner.observe(X[t], targets[t], mode='batched', batch_size=B)
        learner.flush()
        rows_b.append(dict(batch=B, refits=learner.refits,
                           hhl_inversions=learner.linear_solves,
                           n_reweight=N_REWEIGHT,
                           total_gates=learner.modelled_gates))
    dfb = pd.DataFrame(rows_b)
    _save(dfb, 'exp4_gate_cost.csv')
    print(dfb.to_string(index=False))
    print(df.to_string(index=False))
    return df, dfb
