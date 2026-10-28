"""significance_tests_online.py — Statistical validation, ONLINE paradigm only.

Runs on the online benchmark CSV (Dataset,Algorithm,Precision,Recall,F1,FNR,FPR
for the 13 streaming datasets of Figure 4):

  1. Friedman test        — equal algorithm ranks across the 13 streams?
  2. Nemenyi post-hoc     — pairwise critical-difference p-values
  3. Wilcoxon signed-rank — OGC vs each baseline, Holm-corrected

Usage:
    python significance_tests_online.py --input results/online_benchmark_results.csv \
                                        --out results/significance
"""
import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

try:
    import scikit_posthocs as sp
    HAVE_SPH = True
except ImportError:
    HAVE_SPH = False
    print("WARNING: scikit-posthocs not installed — Nemenyi table skipped.")


def build_matrix(df, metric):
    piv = df.pivot_table(index='Dataset', columns='Algorithm',
                         values=metric, aggfunc='mean')
    complete = piv.dropna()
    if len(complete) < len(piv):
        dropped = set(piv.index) - set(complete.index)
        print(f"  [{metric}] dropped {len(dropped)} dataset(s) with missing "
              f"algorithm results: {sorted(dropped)}")
    return complete


def friedman_nemenyi(mat, alpha=0.05):
    algos = list(mat.columns)
    data = [mat[a].values for a in algos]
    stat, p = friedmanchisquare(*data)
    print(f"  Friedman chi2={stat:.3f}, p={p:.6f} "
          f"({'SIGNIFICANT' if p < alpha else 'not significant'} at α={alpha}, "
          f"n={len(mat)} datasets, k={len(algos)} algorithms)")

    if p >= alpha or not HAVE_SPH:
        return p, None

    nemenyi = sp.posthoc_nemenyi_friedman(mat.values)   # rows=datasets, cols=algos
    nemenyi.index = algos
    nemenyi.columns = algos
    assert nemenyi.shape == (len(algos), len(algos))
    return p, nemenyi.where(np.tril(np.ones(nemenyi.shape, dtype=bool), k=-1))


def wilcoxon_vs_ogc(mat, ogc='OGC', alpha=0.05):
    if ogc not in mat.columns:
        print(f"  [{ogc}] not in results — skipping Wilcoxon.")
        return pd.DataFrame()
    rows, pvals = [], []
    for base in mat.columns:
        if base == ogc:
            continue
        diff = (mat[ogc] - mat[base]).values
        diff = diff[~np.isnan(diff)]
        if len(diff) < 6:
            print(f"  skip {base}: only {len(diff)} paired datasets")
            continue
        try:
            stat, p = wilcoxon(diff, zero_method='wilcox', alternative='greater')
        except ValueError:
            stat, p = wilcoxon(diff + 1e-12, zero_method='wilcox',
                               alternative='greater')
        pvals.append(p)
        rows.append({'Comparison': f'{ogc} > {base}', 'n_datasets': len(diff),
                     'wins': int((diff > 0).sum()), 'losses': int((diff < 0).sum()),
                     'ties': int((diff == 0).sum()), 'mean_diff': float(diff.mean()),
                     'wilcoxon_stat': stat, 'p_raw': p})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out['p_holm'] = multipletests(out['p_raw'].values, alpha=alpha, method='holm')[1]
    out['significant'] = multipletests(out['p_raw'].values, alpha=alpha,
                                       method='holm')[0]
    return out.sort_values('p_holm')


def average_ranks(mat):
    return mat.rank(axis=1, ascending=False).mean().sort_values()


def critical_difference(n_datasets, n_algos, alpha=0.05):
    q = {0.05: 2.728, 0.1: 2.291}
    return q[alpha] * np.sqrt(n_algos * (n_algos + 1) / (6.0 * n_datasets))


def run_online_stats(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    print(f"\n{'='*60}\nONLINE PARADIGM — statistical tests "
          f"({df['Dataset'].nunique()} datasets, {df['Algorithm'].nunique()} "
          f"algorithms)\n{'='*60}")

    for metric in ['F1', 'Precision', 'Recall']:
        print(f"\n--- Metric: {metric} ---")
        mat = build_matrix(df, metric)
        if len(mat) < 5:
            print(f"  only {len(mat)} complete datasets — skipped")
            continue

        _, nemenyi = friedman_nemenyi(mat)
        if nemenyi is not None:
            print("\n  Nemenyi post-hoc p-values (lower triangle):")
            print(nemenyi.to_string(float_format=lambda v: f'{v:.4f}'))
            nemenyi.to_csv(os.path.join(out_dir, f'online_nemenyi_{metric.lower()}.csv'))

        w = wilcoxon_vs_ogc(mat)
        if len(w):
            print("\n  Wilcoxon signed-rank, OGC vs baselines (Holm-corrected):")
            print(w.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
            w.to_csv(os.path.join(out_dir, f'online_wilcoxon_{metric.lower()}.csv'),
                     index=False)

        print(f"\n  Average ranks ({metric}, lower=better):")
        print(average_ranks(mat).to_string(float_format=lambda v: f'{v:.2f}'))
        print(f"  Nemenyi CD (α=0.05) = "
              f"{critical_difference(len(mat), mat.shape[1]):.3f}")

    # degeneracy guard: flag any algorithm with recall==1 AND fpr==1 everywhere
    # for algo, sub in df.groupby('Algorithm'):
    #     if (sub['Recall'].eq(1).all()) and (sub['FPR'].eq(1).all()):
    #         print(f"\n  NOTE: {algo} predicts all-positive on every dataset "
    #               f"(recall=1, FPR=1) — degenerate, not evidence of learning.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='results/online_benchmark_results.csv')
    ap.add_argument('--out', default='results/')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    run_online_stats(args.input, args.out)
    print(f"\nDone — tables saved to {args.out}")


if __name__ == '__main__':
    main()