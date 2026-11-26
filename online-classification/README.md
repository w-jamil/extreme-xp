# Classification Experiments — Empirical Study

This README documents the classification experiments reported in the manuscript's **Empirical Study**. The study evaluates seven algorithms under different learning paradigms; its purpose is not to rank the paradigms themselves, since each is suited to a different data structure and learning scenario.

## Manuscript experiment map

| Experiment | Manuscript scope | Principal outputs |
|---|---|---|
| Synthetic imbalance study | 11 synthetic datasets, 3 class-imbalance ratios, 7 online and 7 kernel-online algorithms | `comparison_simulation.csv`, `imbalance_ratio_analysis.png` |
| Online public-data study | 12 cybersecurity streams and MIT-BIH: **13 datasets × 7 algorithms = 91 rows** | `online_benchmark_results.csv`, `online_benchmark_f1.png` |
| Batch public-data study | UNSW-NB15 and Kaggle Credit Fraud are the principal batch results reported in the manuscript | `batch_benchmark_results.csv`, `batch_benchmark_f1.png` |
| Continual learning | OpenML MNIST under Task-IL, Domain-IL, and Class-IL | `continual_learning_summary.csv` |
| Statistical validation | The 13 online benchmarks, 7 algorithms, and 6 OGC-versus-baseline comparisons per metric | `online_nemenyi_*.csv`, `online_wilcoxon_*.csv` |

The code can evaluate additional available datasets under some paradigms. Those exploratory evaluations are not automatically part of the manuscript's reported tables; use the experiment map above when selecting manuscript results.

## Algorithms

The seven compared algorithms are:

- **PA** — Passive-Aggressive classifier
- **Perceptron**
- **OGC** — Online Gradient Classifier
- **AROW** — Adaptive Regularisation of Weights
- **RDA** — Regularised Dual Averaging
- **SCW** — Soft Confidence-Weighted learning
- **AdaRDA** — Adaptive Regularised Dual Averaging

The synthetic study also evaluates the corresponding kernel-online versions: KPA, KPerceptron, KOGC, KAROW, KRDA, KSCW, and KAdaRDA. The batch experiment uses the batch-kernel implementations in `algos.py`, while retaining the seven algorithm names in its result tables.

For the manuscript configuration, PA, Perceptron, and OGC require no hyperparameter tuning. AROW uses regularisation parameter `r=0.1`; RDA and AdaRDA use regularisation parameter `1` in the online and batch experiments; and SCW uses a 90% confidence level (`eta=0.9`). Continual learning follows its separate experimental setup.

## Files

| File | Purpose |
|---|---|
| `main.py` | Command-line entry point for all experiment groups |
| `algos.py` | Algorithm implementations, online updates, metrics, and Numba warm-up |
| `sim_experiments.py` | Synthetic generators, imbalance study, and imbalance figure |
| `online_experiments.py` | Public-data online benchmark and online F1 figure |
| `batch_experiments.py` | Batch-kernel benchmark and batch F1 figure |
| `continual_experiments.py` | MNIST Task-IL, Domain-IL, and Class-IL experiments |
| `sig_tests.py` | Friedman, Nemenyi, and Holm-corrected Wilcoxon analyses |

## Installation

Install the Python dependencies with:

```bash
pip install numpy pandas scipy scikit-learn numba matplotlib pyarrow scikit-posthocs
```

`pyarrow` is required to read the Parquet files. `scikit-posthocs` is required for the Nemenyi tables; the other experiments and Wilcoxon tests do not depend on it. The first online or continual run may include Numba compilation time.

## Data preparation

Synthetic data are generated internally. The public-data experiments use the following layout by default:

```text
.
├── main.py
├── algos.py
├── sim_experiments.py
├── online_experiments.py
├── batch_experiments.py
├── continual_experiments.py
├── sig_tests.py
├── data/
│   ├── CreditFraud_kaggle.parquet
│   ├── UNSW_NB15_train.parquet
│   ├── UNSW_NB15_test.parquet
│   ├── MITBIH_Arrhythmia_train.parquet
│   ├── MITBIH_Arrhythmia_test.parquet
│   ├── Crypto_desktop.parquet
│   ├── Crypto_smartphone.parquet
│   ├── NonEnc_desktop.parquet
│   ├── NonEnc_smartphone.parquet
│   ├── OutFlash_desktop.parquet
│   ├── OutFlash_smartphone.parquet
│   ├── OutTLS_desktop.parquet
│   ├── OutTLS_smartphone.parquet
│   ├── P2P_desktop.parquet
│   ├── P2P_smartphone.parquet
│   ├── Phishing_desktop.parquet
│   └── Phishing_smartphone.parquet
└── results/
```

The continual-learning experiment downloads **OpenML MNIST (`mnist_784`, version 1)** separately, so its first run requires internet access or an existing OpenML cache.

### Public-data conventions

- **Cybersecurity:** each file requires `Time` or `timestamp` and `label` or `Class`. Records are chronologically ordered and aggregated by timestamp; a period is positive if any constituent record is an attack.
- **MIT-BIH:** the train and test files require `label`. They are processed as one train-then-test online stream, with metrics computed on the test portion.
- **UNSW-NB15:** the train and test files require `label` and matching feature columns. The manuscript describes 45 features and 257,673 instances.
- **Kaggle Credit Fraud:** the file requires `Class` or `label`; `Time` supplies the temporal ordering. The manuscript describes 284,807 transactions, including 492 frauds.
- **MNIST:** 70,000 images with 784 pixel features. Continual learning retains all ten digit classes and L2-normalises each sample independently.

Raw binary labels `0` and `1` are represented internally as `-1` and `+1`. Unless a dataset-specific rule states otherwise, `+1` is the positive class for precision, recall, F1, FNR, and FPR.

## Running the experiments

Run commands from the directory containing `main.py`.

### Complete suite

```bash
python main.py --experiment all
```

This runs `simulation`, `online`, `batch`, and `continual` in that order.

### Individual experiment groups

```bash
python main.py --experiment simulation
python main.py --experiment online
python main.py --experiment batch
python main.py --experiment continual
```

### Custom directories and seed

```bash
python main.py \
    --experiment all \
    --data-dir /path/to/data \
    --results-dir /path/to/results \
    --seed 42
```

The defaults are `--experiment all`, `--seed 42`, a `data/` directory beside `main.py`, and a `results/` directory beside `main.py`. The results directory is created automatically. Fixed output filenames are overwritten by subsequent runs, so use another `--results-dir` to preserve an earlier run.

## Experimental details and verified counts

### 1. Synthetic class-imbalance study

The 11 datasets are:

`Separable`, `OverlapGauss`, `Donut`, `Circles`, `Quadratic`, `HighCurve`, `Moons`, `SineWave`, `Spirals`, `SwissRoll`, and `XOR`.

The linearly separable dataset has five features, even though its visualisation uses two dimensions. The other datasets introduce overlapping classes, nonlinear boundaries, manifold structure, or midpoint feature transformations as described in the empirical section.

Each dataset is evaluated at three positive/negative class ratios:

- `50-50`: balanced classes;
- `70-30`: 70% positive and 30% negative;
- `90-10`: 90% positive and 10% negative.

The main entry point generates 2,000 samples per dataset. Every algorithm makes a single predict-before-update online pass; kernel-online counterparts use an RBF kernel with `gamma=1.0`.

The complete table contains **462 rows**:

```text
11 datasets × 3 imbalance ratios × 7 algorithms × 2 methods = 462
```

The figure reports median F1 across the synthetic datasets for each algorithm and imbalance ratio. Its top-three panels are ranked by median F1 in the `90-10` setting.

### 2. Online public-data study

The manuscript's online statistical comparison uses exactly **13 datasets**: the 12 desktop/smartphone cybersecurity streams and MIT-BIH. With seven algorithms, the corresponding result table contains **91 rows**.

Cybersecurity records are processed as evolving time series. MIT-BIH is processed in train-then-test order using predict-before-update online evaluation. The online figure reports positive-class F1 by dataset and algorithm.

If MNIST, UNSW-NB15, or Kaggle files are also present, the generic online runner can evaluate them, but they are not additional blocks in the manuscript's 13-dataset statistical table.

### 3. Batch public-data study

UNSW-NB15 and Kaggle Credit Fraud are the principal batch datasets discussed in the manuscript. All algorithms receive the same dataset splits. The saved batch CSV can contain additional exploratory datasets when their files are available; select `UNSW_NB15` and `CreditFraud_kaggle` for the manuscript's main batch comparison.

The supplied results reproduce the manuscript's highlighted OGC values after rounding:

| Dataset | Precision | Recall | F1 | Manuscript presentation |
|---|---:|---:|---:|---|
| UNSW-NB15 | 0.9980 | 0.9987 | 0.9984 | 0.998, 0.999, 0.998 |
| Kaggle Credit Fraud | 0.7215 | 0.7600 | 0.7403 | 0.722, 0.760, 0.740 |

### 4. MNIST continual learning

The seven online algorithms are evaluated on OpenML MNIST under:

- **Task-IL:** five isolated tasks defined by `(3,5)`, `(4,9)`, `(1,8)`, `(2,7)`, and `(0,6)`; task identity is available and each task has its own state.
- **Domain-IL:** the stream is divided into five sequential chunks with a shared model; forward transfer is evaluated read-only on future chunks.
- **Class-IL:** the same five digit pairs arrive sequentially, prediction uses every class observed so far, and task identity is not supplied.

The default seed list is `[42, 123, 256, 789, 1024]`. `MacroF1` is reported for Task-IL and Class-IL, and `FwdMacroF1` for Domain-IL. The summary also contains cumulative mistake rate `M_T`, which is lower-is-better and is not one of the F1 values in the manuscript table.

The manuscript's MNIST F1 table agrees with the supplied summary when rounded to three decimals:

| Algorithm | Task-IL | Domain-IL | Class-IL |
|---|---:|---:|---:|
| AROW | 0.958 | 0.853 | 0.958 |
| AdaRDA | 0.963 | 0.812 | 0.946 |
| OGC | 0.952 | 0.783 | 0.952 |
| PA | 0.953 | 0.809 | 0.953 |
| Perceptron | 0.958 | 0.837 | 0.958 |
| RDA | **0.965** | 0.790 | 0.948 |
| SCW | 0.960 | **0.866** | **0.960** |

## Metrics

For binary public-data experiments, with TP, FP, TN, and FN denoting confusion-matrix entries:

- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- **F1:** harmonic mean of precision and recall
- **FNR:** FN / (FN + TP) = 1 − recall
- **FPR:** FP / (FP + TN)

Precision, recall, and F1 are higher-is-better; FNR and FPR are lower-is-better. Because majority-class accuracy can be misleading under severe imbalance, the public-data discussion focuses on these positive-class metrics rather than overall accuracy. Accuracy remains available in the synthetic CSV as an additional diagnostic.

## Statistical validation

Run the analysis after generating the online results:

```bash
python sig_tests.py \
    --input results/online_benchmark_results.csv \
    --out results/
```

The input must contain `Dataset`, `Algorithm`, `Precision`, `Recall`, `F1`, `FNR`, and `FPR`. The script constructs complete dataset-by-algorithm matrices, so datasets with missing algorithm results are excluded. The manuscript analysis retains **13 datasets** and **7 algorithms**.

For each of F1, precision, and recall, it performs:

1. A Friedman omnibus test and average-rank calculation.
2. A Nemenyi post-hoc test when the Friedman result is significant and `scikit-posthocs` is installed.
3. Six paired one-sided Wilcoxon signed-rank tests of `OGC > baseline`.
4. Holm correction across those six comparisons for each metric.

The verified manuscript anchors are:

- Friedman F1 statistic: **χ²(6) = 60.94**, with **p = 2.9 × 10⁻¹¹**.
- Nemenyi critical difference at α = 0.05: **CD = 2.50** for `k=7`, `N=13`.
- OGC average F1 rank: **1.12**, followed by AdaRDA at **2.46** and AROW at **3.08**.
- Six comparisons × 13 datasets give **78 OGC-versus-baseline F1 comparisons**.
- With 13 complete blocks and six Holm-corrected comparisons, the minimum attainable adjusted p-value is `6 × 2⁻¹³ ≈ 0.000732`, or approximately **0.073%**, displayed as `<0.1%` in the manuscript.

The F1 W–L–T records are:

| Baseline | Average F1 rank | OGC W–L–T | Holm-adjusted Wilcoxon p |
|---|---:|---:|---:|
| AdaRDA | 2.46 | 12–1–0 | <0.1% |
| AROW | 3.08 | 13–0–0 | <0.1% |
| RDA | 4.58 | 13–0–0 | <0.1% |
| Perceptron | 4.62 | 12–0–1 | <0.1% |
| SCW | 5.46 | 13–0–0 | <0.1% |
| PA | 6.69 | 13–0–0 | <0.1% |

For recall, the supplied tests support significant OGC advantages over AROW and PA, but not over Perceptron, RDA, or AdaRDA. SCW's recall result must be interpreted together with its near-degenerate positive-class predictions and very low precision on Kaggle Fraud, as explained in the manuscript.

For custom paths:

```bash
python sig_tests.py \
    --input /path/to/results/online_benchmark_results.csv \
    --out /path/to/results/
```

## Output files

| Output | Contents |
|---|---|
| `comparison_simulation.csv` | 462-row synthetic online/kernel-online F1 and accuracy table |
| `imbalance_ratio_analysis.png` | Four-panel median-F1 imbalance analysis |
| `online_benchmark_results.csv` | Online precision, recall, F1, FNR, and FPR; 91 rows for the manuscript's 13 datasets |
| `online_benchmark_f1.png` | Online F1 comparison used in the empirical section |
| `batch_benchmark_results.csv` | Batch metrics for all successfully evaluated datasets; filter to the manuscript scope when reporting |
| `batch_benchmark_f1.png` | Batch F1 comparison |
| `continual_learning_summary.csv` | Mean ± standard-deviation continual-learning summary |
| `online_nemenyi_f1.csv` | Nemenyi F1 p-values, when generated |
| `online_nemenyi_precision.csv` | Nemenyi precision p-values, when generated |
| `online_nemenyi_recall.csv` | Nemenyi recall p-values, when generated |
| `online_wilcoxon_f1.csv` | OGC-versus-baseline F1 tests, W–L–T, and Holm correction |
| `online_wilcoxon_precision.csv` | Corresponding precision tests |
| `online_wilcoxon_recall.csv` | Corresponding recall tests |

Nemenyi CSVs are conditional outputs: they are absent if the Friedman test is not significant, too few complete datasets are available, or `scikit-posthocs` is not installed. Friedman statistics and average ranks are printed to the console rather than saved as standalone CSV files.

## Reproducibility notes

- `main.py` initialises the common random state before dispatching experiments. `--seed` directly controls synthetic generation and the simulation's imbalance/outlier transformations.
- Some protocols use fixed internal seeds, including deterministic epoch shuffling and the continual-learning seed list. Changing `--seed` therefore does not reconfigure every experiment-specific seed.
- The current generic online-benchmark call sets SCW to `eta=0.5`, whereas the empirical section states a 90% confidence level (`eta=0.9`). Set the online SCW confidence to `eta=0.9` when aligning a fresh run with that manuscript configuration.
- Exact numerical reproduction also depends on package versions, Numba/BLAS behaviour, input contents and ordering, and the OpenML download. Archive package versions with the results.
- Always verify the retained dataset count before statistical analysis. The manuscript's statistical numbers specifically assume 13 complete online datasets, seven algorithms, and six Holm-corrected comparisons per metric.