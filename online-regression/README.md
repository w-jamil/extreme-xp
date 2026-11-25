# QIKRR — Physics-Informed PDE Experiments

Experimental code for the paper *Quantum Iterative Kernel Ridge Regression for Physics-Informed Learning under Delayed and Batched Feedback*.

The suite uses **QIKRR** consistently throughout and separates predictive performance from computational cost:

1. **Factorial predictive comparisons.** Matched same-kernel methods separate iterative reweighting, physics constraints, and simulated solver/noise. In particular, QIKRR is compared with **PI-IKRR** at the same two-solve iterative regularisation, PI-IKRR is compared with non-iterative **PI-KRR**, and data-only **IKRR** is compared with **KRR**.
2. **Computational cost.** Measured non-quantum CPU time and modelled fault-tolerant quantum gate/query counts are reported separately. Because HHL is simulated by linear algebra, these experiments do **not** demonstrate a measured quantum wall-clock speedup.

## Method matrix

| Method | Kernel | Physics constraints | Regularisation and solver |
|---|---|---:|---|
| QIKRR (exact ZZ) | exact commuting ZZ | yes | two-solve iterative reweighting + simulated HHL |
| PI-IKRR (exact ZZ) | exact commuting ZZ | yes | two-solve iterative reweighting + non-quantum direct solves |
| PI-KRR (exact ZZ) | exact commuting ZZ | yes | one non-quantum direct solve |
| QIKRR (RBF) | Gaussian RBF | yes | two-solve iterative reweighting + simulated HHL |
| PI-IKRR (RBF) | Gaussian RBF | yes | two-solve iterative reweighting + non-quantum direct solves |
| PI-KRR (RBF) | Gaussian RBF | yes | one non-quantum direct solve |
| QIKRR (RFF sim.) | finite-dimensional RFF surrogate | yes | two-solve iterative reweighting + simulated HHL |
| PI-IKRR (RFF sim.) | finite-dimensional RFF surrogate | yes | two-solve iterative reweighting + non-quantum direct solves |
| PI-KRR (RFF sim.) | finite-dimensional RFF surrogate | yes | one non-quantum direct solve |
| QIKRR (RFF, no phys) | RFF surrogate | no | two-solve iterative reweighting + simulated HHL |
| IKRR (RFF, no phys) | RFF surrogate | no | two-solve iterative reweighting + non-quantum direct solves |
| KRR (RFF, no phys) | RFF surrogate | no | one non-quantum direct solve |
| IKRR (exact ZZ, no phys) | exact commuting ZZ | no | two-solve iterative reweighting + non-quantum direct solves |
| KRR (exact ZZ, no phys) | exact commuting ZZ | no | one non-quantum direct solve |
| IKRR (RBF, no phys) | Gaussian RBF | no | two-solve iterative reweighting + non-quantum direct solves |
| KRR (RBF, no phys) | Gaussian RBF | no | one non-quantum direct solve |
| PINN | neural representation | yes | Adam |

For each kernel, the factorial comparisons are:

- **Solver/noise with physics:** QIKRR / PI-IKRR.
- **Iteration with physics:** PI-IKRR / PI-KRR.
- **Iteration without physics:** IKRR / KRR.
- **Physics without iteration:** PI-KRR / KRR.
- **Physics with iteration:** PI-IKRR / IKRR.
- **Overall:** QIKRR / PI-KRR.
- **Solver/noise without physics:** QIKRR-RFF / IKRR-RFF.

A ZZ-versus-RBF comparison alone measures kernel inductive bias, not a quantum-solver advantage. The appropriate simulated-solver control is the same-kernel, same-iteration QIKRR/PI-IKRR comparison.

QIKRR, PI-IKRR, and IKRR use `n_reweight=2`: an initial isotropic solve followed by one fixed-operator reweighted solve. PI-KRR and KRR use one solve. Offline fits—including the full causal refits in EXP 2—define the feature coordinates through the eigenbasis of the training-functional Gram matrix; native incremental RFF updates use the explicit RFF coordinates. If \(w_e\) is the previous primal coefficient vector, the next operator is \(G_e=\operatorname{diag}(\max(|w_e|,\delta))\), with \(\delta=10^{-3}\). The implementation reweights these primal feature coefficients—not the dual Kansa coefficients—so the solved systems match the paper's fixed-\(G_e\) formulation.

Physics-informed QIKRR/PI-IKRR/PI-KRR triples share hyperparameters selected by non-quantum PI-KRR. Data-only QIKRR/IKRR/KRR triples share hyperparameters selected by non-quantum KRR using validation solution error alone. These are controlled common-setting comparisons, not claims of global hyperparameter optimality.

## Experiments

- **EXP 1 — PDE solution accuracy and residuals.** Four manufactured problems: Poisson-1D, advection–diffusion-1D, Poisson-2D, and Poisson-4D. Reports relative \(L_2\) error on a dense test grid, PDE-residual RMSE on 400 fresh collocation points, CPU time, linear-solve counts, and modelled gates. Five seeds and 17 methods give **340 raw rows**. Raw and mean±std CSVs are saved, together with per-seed matched factorial comparisons.
- **EXP 2 — Causal sequential learning on all four PDEs and all three kernels.** For each RFF, RBF, and exact-ZZ kernel, the suite runs QIKRR, PI-IKRR, PI-KRR, and data-only QIKRR, IKRR, and KRR under immediate, delayed (\(d=10\)), and batched (\(B=20\)) feedback on \(T=200\) noisy observations. A fixed set of 60 collocation points is available before the stream for physics-informed models; data-only models omit it. Hyperparameters and the clip bound come from an independently generated calibration stream. Same-kernel hindsight references have matching physics access. Four problems, three kernels, six methods, three modes, and five seeds give **1,080 raw rows**, plus **1,260 matched factorial comparisons**. Reports cumulative loss, reference gap, final solution error/residual, runtime, refits, and linear-solve counts.
- **EXP 3 — Spectra and effective dimension.** Compares the RFF surrogate, RBF, and Matérn (\(\nu=1\)) Gram matrices on Poisson-2D and Poisson-4D inputs. These are spectral diagnostics, not realised-regret lower bounds.
- **EXP 4 — Estimation and batching models.** Synthetic direct-Hadamard-sampling error scaling (\(S^{-1/2}\)) versus QAE scaling (\(S^{-1}\)), and causal batching refit/inversion counts. This is a scaling simulation, not a circuit-level QAE implementation.
- **EXP 5 — non-quantum time and modelled quantum cost.** Measures non-quantum Kansa-system assembly/solve time and evaluates HHL/QLSA gate models at measured condition numbers. CPU seconds and gate counts are plotted separately and are never converted into a hardware speedup claim.
- **EXP 6 — Exact commuting ZZ spectrum.** Evaluates the closed-form ZZ fidelity kernel and verifies its finite-support spectrum and bounded effective dimension.

## Verified five-seed factorial outcome

The completed EXP 1 run contains **340 rows**: four problems, five seeds, and 17 methods. The factorial file contains **380 rows**: 19 comparison/kernel groups, each with 20 problem/seed pairs.

- **PI-IKRR versus PI-KRR:** PI-IKRR wins **35/60** accuracy comparisons and **26/60** residual comparisons. Exact ZZ gives a geometric-mean error ratio of **0.968** and 14/20 accuracy wins, but its residual ratio is 1.010. RBF and RFF geometric-mean error ratios are above one. Iteration is therefore not uniformly superior with physics.
- **IKRR versus KRR:** IKRR wins **40/60** accuracy comparisons and **31/60** residual comparisons, with aggregated geometric-mean ratios **0.965** and **0.979**. RFF-IKRR has lower mean error on all four problems, but RBF-IKRR is worse on Poisson-2D and Poisson-4D, and the median IKRR/KRR time ratio is **2.23**. The results do not support an “IKRR beats KRR in all ways” claim.
- **QIKRR versus PI-IKRR:** at matched iterative regularisation, QIKRR wins only **13/60** accuracy and **8/60** residual comparisons. Exact-ZZ QIKRR has no wins in either metric. The simulated HHL perturbations degrade accuracy relative to the non-quantum two-solve control.
- **Physics ablations:** PI-KRR/KRR and PI-IKRR/IKRR each win **60/60** residual comparisons, and respectively win **54/60** and **53/60** accuracy comparisons. Collocation also enlarges the system and increases measured runtime.
- **Timing:** the median QIKRR/PI-KRR time ratio is **1.40**, and the matched two-solve QIKRR/PI-IKRR median is **1.01**. There is no measured wall-clock speed advantage.

## Verified five-seed sequential outcome

The completed EXP 2 run contains **1,080 raw rows** and **1,260 matched comparisons**, covering all four PDEs, all three kernels, six methods per kernel, three feedback modes, and five seeds. All ratios below are geometric means of per-seed cumulative-loss ratios; values below one favour QIKRR.

- **QIKRR versus PI-KRR:** QIKRR wins only **22/180** matched comparisons, with aggregate ratio **1.0118**. By problem, the ratios are **1.0065** (Poisson-1D), **0.9986** (AdvecDiff-1D), **1.0176** (Poisson-2D), and **1.0247** (Poisson-4D), with respectively **1/45**, **12/45**, **3/45**, and **6/45** wins. The former Poisson-1D advantage therefore does not persist in the matched same-kernel design.
- **QIKRR versus PI-IKRR:** QIKRR wins only **14/180** comparisons, with aggregate ratio **1.0079**. Aggregating over problems and kernels gives ratios **1.0210**, **1.0019**, and **1.0011** under immediate, delayed, and batched feedback. There is no matched sequential-loss advantage for the simulated solver.
- **Largest isolated gain:** AdvecDiff-1D with RFF and immediate feedback has QIKRR/PI-KRR ratio **0.9641**, but its matched QIKRR/PI-IKRR ratio is **1.0314**. The gain over non-iterative PI-KRR is therefore associated with iterative regularisation, not simulated HHL.
- **Feedback staleness:** delayed feedback increases mean cumulative loss relative to immediate feedback in **72/72** problem/kernel/method groups; batched feedback also increases loss in **72/72** groups. Immediate, delayed, and batched feedback require respectively 200, 191, and 10 refits; the two-solve methods therefore require 400, 382, and 20 solves.
- **Runtime:** the median QIKRR/PI-KRR ratio is **1.45**, and the matched QIKRR/PI-IKRR median is **1.01**. These are measured non-quantum CPU-time ratios and do not indicate quantum speedup.

See `../empirical.tex` for the manuscript-ready interpretation and limitations.

## Files

| File | Contents |
|---|---|
| `pde.py` | Manufactured PDEs, exact solutions, right-hand sides, and samplers |
| `kernels.py` | RFF, RBF, and exact ZZ kernels with analytic derivative blocks |
| `algos.py` | QIKRR, PI-IKRR, PI-KRR, IKRR, KRR, PINN, causal updates, and HHL cost simulator |
| `experiments.py` | EXP 1–6, validation-only tuning, factorial comparisons, CSV export |
| `make_figures.py` | Paper figures from `results/*.csv` |
| `main.py` | Command-line entry point |


## Usage

Install the dependencies:

```bash
pip install numpy scipy pandas matplotlib scikit-learn torch
```

Run the full suite (EXP 1 and EXP 2 save checkpoints after every completed problem/seed and resume automatically after interruption):

```bash
python main.py
```

Start EXP 1 afresh instead of resuming:

```bash
python main.py --experiment exp1 --no-resume
```

Run selected experiments:

```bash
python main.py --experiment exp1 --seeds 0 1 2 3 4
python main.py --experiment exp2  # all four PDEs and all three kernels
python main.py --experiment exp2 --problem advecdiff1d  # one selected PDE
python main.py --experiment exp3
python main.py --experiment exp4
python main.py --experiment exp5
python main.py --experiment exp6
```

Use a small smoke test before a full run. Quick runs write to `results_quick/`, so they cannot overwrite paper CSVs or full-run checkpoints:

```bash
python main.py --experiment all --quick
```

Use `--results-dir PATH` to select another output location. Set `QIKRR_RESULTS_DIR=PATH` when running `make_figures.py` to read results from that location.

Generate figures:

```bash
python make_figures.py
```

By default figures are written to `./figures/`. Set `QIKRR_FIGURE_DIR` to override the destination.

## Important outputs

| Output | Purpose |
|---|---|
| `results/exp1_pde_accuracy.csv` | Per-seed accuracy, residual, timing, solve, and gate results |
| `results/exp1_pde_accuracy_summary.csv` | Mean±std summary for the paper table |
| `results/exp1_factorial_comparisons.csv` | Per-seed matched ratios, wins, and timing for all factorial comparisons |
| `results/exp1_hyperparams.csv` | Shared and method-specific selected hyperparameters |
| `results/exp2_online_regret.csv` | Four-PDE, three-kernel sequential loss/regret and solve accounting |
| `results/exp2_online_regret_summary.csv` | Mean±std sequential summaries by problem, kernel, method, and feedback mode |
| `results/exp2_cumcurves.csv` | Per-round cumulative-loss trajectories for every problem, kernel, method, mode, and seed |
| `results/exp2_factorial_comparisons.csv` | Per-seed same-kernel sequential loss, accuracy, residual, and timing ratios |
| `results/exp2_hyperparams.csv` | Independently calibrated shared sequential hyperparameters |
| `results/exp3_*.csv`, `results/exp6_*.csv` | Spectral and effective-dimension results |
| `results/exp4_*.csv`, `results/exp5_scaling.csv` | Estimation, batching, and cost-model results |
| `figures/accuracy_benchmark.pdf` | Same-kernel, same-iteration QIKRR/PI-IKRR accuracy ratios |
| `figures/iteration_benchmark.pdf` | PI-IKRR/PI-KRR and IKRR/KRR accuracy ratios |
| `figures/regret_curves.pdf` | Four-PDE QIKRR-RFF cumulative-loss trajectories under all feedback modes |
| `figures/sequential_factorial.pdf` | Cross-problem QIKRR/PI-KRR and QIKRR/PI-IKRR sequential loss ratios |

## Simulation and reporting limitations

- The RFF backend is a finite-dimensional surrogate, not an exact simulation of a quantum circuit.
- The exact ZZ backend evaluates a closed-form fidelity kernel and its analytic derivatives; no quantum circuit is executed.
- `HHLSimulator` performs `numpy.linalg.solve`, adds optional precision-scaled noise, and accumulates the gate model \(\log_2(n)\kappa^2\log_2(1/\epsilon)\). For nonsymmetric Kansa matrices, a quantum implementation would require a standard Hermitian dilation or an equivalent non-Hermitian linear-system construction; the model suppresses its constant overhead.
- Iterative reweighting performs `n_reweight=2` linear solves per refit by default. Both refits and actual linear solves are reported; batching reduces refits by the batch factor while the constant reweighting multiplier remains visible.
- Quantum gate estimates depend on conditioning and input-model assumptions. They are not wall-clock measurements and do not establish an unconditional end-to-end advantage.
- Solver/noise claims must use the **same-kernel QIKRR/PI-IKRR comparison**; iterative-regression claims must use **PI-IKRR/PI-KRR** with physics or **IKRR/KRR** without physics. Kernel and physics effects must not be attributed to quantum computation.
- Common hyperparameters provide controlled comparisons, not globally optimised performance for any method.

No external datasets are required. All observations and collocation points are generated from the manufactured PDEs, and seeds make the runs reproducible.
