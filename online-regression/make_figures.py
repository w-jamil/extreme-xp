#!/usr/bin/env python3
"""Generate paper figures from the PI-KRR result CSV files."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('QIKRR_RESULTS_DIR', os.path.join(HERE, 'results'))
OUT = os.environ.get('QIKRR_FIGURE_DIR', os.path.join(HERE, 'figures'))
os.makedirs(OUT, exist_ok=True)


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches='tight')
    print('wrote', path)


def _ratio_plot(ax, df, title):
    """Geometric mean method/benchmark ratios by problem and kernel."""
    df = df.copy()
    df['variant'] = df.kernel
    df['log_ratio'] = np.log(np.clip(df.rel_l2_ratio, 1e-300, None))
    summary = (df.groupby(['problem', 'variant'])
               .agg(log_mean=('log_ratio', 'mean'),
                    log_std=('log_ratio', 'std')).reset_index())
    summary['ratio'] = np.exp(summary.log_mean)
    summary['log_std'] = summary.log_std.fillna(0)
    preferred = ['poisson1d', 'advecdiff1d', 'poisson2d', 'poisson4d']
    problems = [p for p in preferred if p in set(summary.problem)]
    problem_labels = {'poisson1d': 'Poisson-1D', 'advecdiff1d': 'AdvecDiff-1D',
                      'poisson2d': 'Poisson-2D', 'poisson4d': 'Poisson-4D'}
    variants = [v for v in ('exact ZZ', 'RBF', 'RFF')
                if v in set(summary.variant)]
    colors = {'exact ZZ': 'tab:orange', 'RBF': 'tab:blue', 'RFF': 'tab:green'}
    x = np.arange(len(problems))
    width = 0.8 / len(variants)
    for i, variant in enumerate(variants):
        sub = summary[summary.variant == variant].set_index('problem').loc[problems]
        lower = sub.ratio - sub.ratio * np.exp(-sub.log_std)
        upper = sub.ratio * np.exp(sub.log_std) - sub.ratio
        ax.bar(x + (i - (len(variants) - 1) / 2) * width, sub.ratio, width,
               yerr=np.vstack([lower, upper]), capsize=2,
               label=variant, color=colors[variant])
    ax.set_yscale('log')
    ax.axhline(1.0, color='black', ls='--', lw=1.0, label='parity')
    ax.set_xticks(x, [problem_labels.get(p, p) for p in problems], fontsize=8,
                  rotation=20, ha='right')
    ax.set_ylabel('rel-$L_2$ ratio\n(geometric mean)')
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, ncol=2)


def _factorial_data():
    csv = os.path.join(RES, 'exp1_factorial_comparisons.csv')
    if not os.path.exists(csv):
        print('missing', csv)
        return None
    return pd.read_csv(csv)


def fig_accuracy():
    df = _factorial_data()
    if df is None:
        return
    df = df[df.comparison == 'solver with physics']
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    _ratio_plot(ax, df, 'PI-KRR (rew., sim.)/PI-KRR (rew., direct)\n(below 1 favours the direct solve)')
    fig.tight_layout()
    _save(fig, 'accuracy_benchmark.pdf')


def fig_iteration():
    df = _factorial_data()
    if df is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    _ratio_plot(axes[0], df[df.comparison == 'iteration with physics'],
                '(a) (rew., direct)/(ridge, direct)')
    _ratio_plot(axes[1], df[df.comparison == 'iteration without physics'],
                '(b) (rew., direct)/(ridge, direct), no phys')
    fig.tight_layout()
    _save(fig, 'iteration_benchmark.pdf')


def fig_spectral():
    spec = pd.read_csv(os.path.join(RES, 'exp3_spectra.csv'))
    deff = pd.read_csv(os.path.join(RES, 'exp3_effective_dimension.csv'))
    zz_path = os.path.join(RES, 'exp6_zz_spectra.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    ax.semilogy(spec.idx, np.clip(spec.eig_qikrr_rff, 1e-18, None),
                lw=1.6, label='PI-KRR (RFF, rew., sim.)')
    if os.path.exists(zz_path):
        zz = pd.read_csv(zz_path)
        ax.semilogy(zz.idx, np.clip(zz.eig_zz_n8, 1e-18, None), lw=1.6,
                    color='tab:purple', label='exact ZZ, $n{=}8$')
    ax.semilogy(spec.idx, spec.eig_matern, lw=1.6, label='Matérn $\\nu{=}1$')
    ax.semilogy(spec.idx, spec.eig_rbf, lw=1.6, ls='--', label='RBF')
    ax.set_xlabel('eigenvalue index $i$')
    ax.set_ylabel('eigenvalue $\\lambda_i$')
    ax.legend(fontsize=7)
    ax.set_title('(a) Gram spectra', fontsize=9)

    ax = axes[1]
    ax.plot(deff.t, deff.d_eff_qikrr_rff, 'o-', lw=1.6,
            label='PI-KRR (RFF, rew., sim.)')
    zz_deff_path = os.path.join(RES, 'exp6_zz_deff.csv')
    if os.path.exists(zz_deff_path):
        zzd = pd.read_csv(zz_deff_path)
        styles = {2: ('v', 'tab:purple'), 4: ('^', 'tab:cyan'),
                  6: ('<', 'magenta'), 8: ('>', 'tab:brown')}
        for nq, (marker, color) in styles.items():
            group = zzd[zzd.n_qubits == nq]
            if len(group):
                ax.plot(group.t, group.d_eff, marker + '--', ms=3,
                        lw=1.3, color=color, label=f'exact ZZ $n={nq}$')
    ax.plot(deff.t, deff.d_eff_matern, 's-', lw=1.6, label='Matérn $\\nu{=}1$')
    ax.plot(deff.t, deff.d_eff_rbf, '^--', lw=1.6, label='RBF')
    ax.set_xlabel('number of points $t$')
    ax.set_ylabel('$d_{\\mathrm{eff}}(t,10^{-4})$')
    ax.legend(fontsize=6.5)
    ax.set_title('(b) effective dimension', fontsize=9)
    fig.tight_layout()
    _save(fig, 'spectral.pdf')


def fig_regret():
    path = os.path.join(RES, 'exp2_cumcurves.csv')
    if not os.path.exists(path):
        print('missing', path)
        return
    cur = pd.read_csv(path)
    cur = cur[cur.method == 'QIKRR (RFF)']
    preferred = ['poisson1d', 'advecdiff1d', 'poisson2d', 'poisson4d']
    problems = [p for p in preferred if p in set(cur.problem)]
    labels = {'poisson1d': 'Poisson-1D', 'advecdiff1d': 'AdvecDiff-1D',
              'poisson2d': 'Poisson-2D', 'poisson4d': 'Poisson-4D'}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), squeeze=False)
    styles = {'immediate': ('-', 'tab:blue'),
              'delayed': ('--', 'tab:orange'),
              'batched': ('-.', 'tab:green')}
    for ax, problem in zip(axes.ravel(), problems):
        sub = cur[cur.problem == problem]
        for mode, (ls, color) in styles.items():
            group = sub[sub['mode'] == mode]
            mean = group.groupby('t').cum.mean()
            std = group.groupby('t').cum.std().fillna(0)
            ax.plot(mean.index, mean.values, ls=ls, color=color,
                    lw=1.5, label=mode)
            ax.fill_between(mean.index, mean.values - std.values,
                            mean.values + std.values, color=color, alpha=0.15)
        ax.set_xlabel('round $t$')
        ax.set_ylabel('cumulative loss')
        ax.set_title(labels.get(problem, problem), fontsize=9)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('PI-KRR (RFF, rew., sim.) causal feedback protocols', fontsize=10)
    fig.tight_layout()
    _save(fig, 'regret_curves.pdf')


def fig_sequential_factorial():
    path = os.path.join(RES, 'exp2_factorial_comparisons.csv')
    if not os.path.exists(path):
        print('missing', path)
        return
    df = pd.read_csv(path)
    comparisons = [('overall QIKRR vs PI-KRR', '(rew., sim.)/(ridge, direct)'),
                   ('solver with physics', '(rew., sim.)/(rew., direct)')]
    modes = ['immediate', 'delayed', 'batched']
    preferred = ['poisson1d', 'advecdiff1d', 'poisson2d', 'poisson4d']
    problems = [p for p in preferred if p in set(df.problem)]
    problem_labels = {'poisson1d': 'Poisson-1D',
                      'advecdiff1d': 'AdvecDiff-1D',
                      'poisson2d': 'Poisson-2D', 'poisson4d': 'Poisson-4D'}
    kernels = [k for k in ('RFF', 'RBF', 'exact ZZ') if k in set(df.kernel)]
    colors = {'RFF': 'tab:green', 'RBF': 'tab:blue', 'exact ZZ': 'tab:orange'}
    fig, axes = plt.subplots(len(comparisons), len(modes),
                             figsize=(7.4, 4.8), squeeze=False)
    x = np.arange(len(problems))
    width = 0.8 / len(kernels)
    for row, (comparison, comparison_label) in enumerate(comparisons):
        for col, mode in enumerate(modes):
            ax = axes[row, col]
            sub = df[(df.comparison == comparison) & (df['mode'] == mode)].copy()
            sub['log_ratio'] = np.log(np.clip(sub.loss_ratio, 1e-300, None))
            summary = (sub.groupby(['problem', 'kernel'])
                       .log_ratio.agg(['mean', 'std']).reset_index())
            summary['ratio'] = np.exp(summary['mean'])
            summary['log_std'] = summary['std'].fillna(0)
            for i, kernel in enumerate(kernels):
                vals = summary[summary.kernel == kernel].set_index('problem').reindex(problems)
                lower = vals.ratio - vals.ratio * np.exp(-vals.log_std)
                upper = vals.ratio * np.exp(vals.log_std) - vals.ratio
                ax.bar(x + (i - (len(kernels) - 1) / 2) * width,
                       vals.ratio, width, yerr=np.vstack([lower, upper]),
                       capsize=2, label=kernel, color=colors[kernel])
            ax.set_yscale('log')
            ax.axhline(1.0, color='black', ls='--', lw=0.9)
            ax.set_xticks(x, [problem_labels.get(p, p) for p in problems],
                          fontsize=7, rotation=20, ha='right')
            ax.set_title(f'{comparison_label} — {mode}', fontsize=8)
            if col == 0:
                ax.set_ylabel('cumulative-loss ratio\n(geometric mean)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, ncol=3, loc='upper center',
               bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, 'sequential_factorial.pdf')


def fig_scaling():
    path = os.path.join(RES, 'exp5_scaling.csv')
    if not os.path.exists(path):
        print('missing', path)
        return
    df = pd.read_csv(path)
    N = df.N.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    ax.loglog(N, df.classical_solve_s, 'o-', color='tab:red',
              label='measured classical solve')
    cubic = df.classical_solve_s.iloc[0] * (N / N[0]) ** 3
    ax.loglog(N, cubic, '-.', color='gray', label='$N^3$ reference')
    ax.set_xlabel('system size $N$')
    ax.set_ylabel('CPU seconds')
    ax.legend(fontsize=7)
    ax.set_title('(a) measured classical time', fontsize=9)

    ax = axes[1]
    ax.loglog(N, df.modelled_qlsa_gates, 's--', color='tab:blue',
              label='modelled gates, measured $\\kappa$')
    k0 = df.kappa.iloc[0]
    n_reweight = int(df.n_reweight.iloc[0])
    fixed = n_reweight * k0 * np.log2(max(k0, 2)) ** 3 * np.log2(N)
    ax.loglog(N, fixed, ':', color='tab:green', label='modelled gates, $\\kappa$ fixed')
    ax.set_xlabel('system size $N$')
    ax.set_ylabel('gate-count model')
    ax.legend(fontsize=7)
    ax.set_title('(b) modelled PI-KRR cost', fontsize=9)
    fig.tight_layout()
    _save(fig, 'scaling.pdf')


def fig_deff4d():
    path = os.path.join(RES, 'exp3_effective_dimension_poisson4d.csv')
    if not os.path.exists(path):
        print('missing', path)
        return
    deff = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    ax.plot(deff.t, deff.d_eff_qikrr_rff, 'o-', lw=1.8,
            color='tab:blue', label='PI-KRR (RFF, rew., sim.)')
    ax.plot(deff.t, deff.d_eff_matern, 's-', lw=1.6,
            color='tab:orange', label='Matérn $\\nu{=}1$')
    ax.plot(deff.t, deff.d_eff_rbf, '^--', lw=1.6,
            color='tab:green', label='RBF')
    ax.plot(deff.t, deff.t, ':', color='gray', lw=1.0, label='$t$ (full rank)')
    ax.set_xlabel('number of points $t$ (Poisson-4D)')
    ax.set_ylabel('$d_{\\mathrm{eff}}(t,10^{-4})$')
    ax.legend(fontsize=7)
    ax.set_title('Effective dimension in $d=4$', fontsize=9)
    fig.tight_layout()
    _save(fig, 'deff4d.pdf')


if __name__ == '__main__':
    fig_accuracy()
    fig_iteration()
    fig_spectral()
    fig_regret()
    fig_sequential_factorial()
    fig_scaling()
    fig_deff4d()
