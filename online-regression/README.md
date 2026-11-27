# Physics-Informed Kernel Regularised Regression

Experimental code for the paper *Physics-Informed Kernel Regularised Regression*. Method names below follow the paper's (kernel, regularisation, solver) configuration naming.

The suite separates predictive performance from computational cost:

1. **Factorial predictive comparisons.** Matched same-kernel methods separate iterative reweighting, physics constraints, and simulated solver/noise. In particular, the reweighted simulated-solve configuration is compared with the reweighted direct-solve configuration at the same two-solve iterative regularisation, the reweighted direct-solve configuration is compared with the non-iterative ridge configuration, and data-only **IKRR** is compared with **KRR**.
2. **Computational cost.** Measured non-quantum CPU time and modelled fault-tolerant quantum gate/query counts are reported separately. Because HHL is simulated by linear algebra, these experiments do **not** demonstrate a measured quantum wall-clock speedup.

## Method matrix

Configurations are named by a **(kernel, regularisation, solver)** tuple: `rew.` = two-solve iterative reweighting, `ridge` = one isotropic solve, `sim.` = simulated HHL, `direct` = non-quantum direct solve, and `ZZ`/`RBF`/`RFF` = the exact commuting-ZZ, Gaussian, and random-Fourier-feature kernels. Physics constraints are on unless the label is a data-only `IKRR`/`KRR` configuration.

| Method | Kernel | Physics constraints | Regularisation and solver |
|---|---|---:|---|
| PI-KRR (ZZ, rew., sim.) | exact commuting ZZ | yes | two-solve iterative reweighting + simulated HHL |
| PI-KRR (ZZ, rew., direct) | exact commuting ZZ | yes | two-solve iterative reweighting + non-quantum direct solves |
| PI-KRR (ZZ, ridge, direct) | exact commuting ZZ | yes | one non-quantum direct solve |
| PI-KRR (RBF, rew., sim.) | Gaussian RBF | yes | two-solve iterative reweighting + simulated HHL |
| PI-KRR (RBF, rew., direct) | Gaussian RBF | yes | two-solve iterative reweighting + non-quantum direct solves |
| PI-KRR (RBF, ridge, direct) | Gaussian RBF | yes | one non-quantum direct solve |
| PI-KRR (RFF, rew., sim.) | finite-dimensional RFF surrogate | yes | two-solve iterative reweighting + simulated HHL |
| PI-KRR (RFF, rew., direct) | finite-dimensional RFF surrogate | yes | two-solve iterative reweighting + non-quantum direct solves |
| PI-KRR (RFF, ridge, direct) | finite-dimensional RFF surrogate | yes | one non-quantum direct solve |
| IKRR (RFF, rew., sim.) | RFF surrogate | no | two-solve iterative reweighting + simulated HHL |
| IKRR (RFF, rew., direct) | RFF surrogate | no | two-solve iterative reweighting + non-quantum direct solves |
| KRR (RFF, ridge, direct) | RFF surrogate | no | one non-quantum direct solve |
| IKRR (ZZ, rew., direct) | exact commuting ZZ | no | two-solve iterative reweighting + non-quantum direct solves |
| KRR (ZZ, ridge, direct) | exact commuting ZZ | no | one non-quantum direct solve |
| IKRR (RBF, rew., direct) | Gaussian RBF | no | two-solve iterative reweighting + non-quantum direct solves |
| KRR (RBF, ridge, direct) | Gaussian RBF | no | one non-quantum direct solve |
| PINN | neural representation | yes | Adam |

For each kernel, the factorial comparisons are:

- **Solver/noise with physics:** (rew., sim.) / (rew., direct).
- **Iteration with physics:** (rew., direct) / (ridge, direct).
- **Iteration without physics:** IKRR / KRR.
- **Physics without iteration:** (ridge, direct, phys) / (ridge, direct, no phys).
- **Physics with iteration:** (rew., direct, phys) / (rew., direct, no phys).
- **Overall:** (rew., sim.) / (ridge, direct).
- **Solver/noise without physics:** IKRR (RFF, rew., sim.) / IKRR (RFF, rew., direct).

A ZZ-versus-RBF comparison alone measures kernel inductive bias, not a quantum-solver advantage. The appropriate simulated-solver control is the same-kernel, same-iteration (rew., sim.)/(rew., direct) comparison.

The reweighted configurations — the `rew.` rows above — use `n_reweight=2`: an initial isotropic solve followed by one fixed-operator reweighted solve. The ridge configurations use one solve. Offline fits—including the full causal refits in EXP 2—define the feature coordinates through the eigenbasis of the training-functional Gram matrix; native incremental RFF updates use the explicit RFF coordinates. If \(w_e\) is the previous primal coefficient vector, the next operator is \(G_e=\operatorname{diag}(\max(|w_e|,\delta))\), with \(\delta=10^{-3}\). The implementation reweights these primal feature coefficients—not the dual Kansa coefficients—so the solved systems match the paper's fixed-\(G_e\) formulation.

Physics-informed triples — (rew., sim.), (rew., direct), and (ridge, direct) on one kernel — share hyperparameters selected by the non-quantum (ridge, direct) configuration. Data-only triples share hyperparameters selected by the non-quantum KRR configuration using validation solution error alone. These are controlled common-setting comparisons, not claims of global hyperparameter optimality.

## Experiments

- **EXP 1 — PDE solution accuracy and residuals.** Four manufactured problems: Poisson-1D, advection–diffusion-1D, Poisson-2D, and Poisson-4D. Reports relative \(L_2\) error on a dense test grid, PDE-residual RMSE on 400 fresh collocation points, CPU time, linear-solve counts, and modelled gates. Five seeds and 17 methods give **340 raw rows**. Raw and mean±std CSVs are saved, together with per-seed matched factorial comparisons.
- **EXP 2 — Causal sequential learning on all four PDEs and all three kernels.** For each RFF, RBF, and exact-ZZ kernel, the suite runs the physics-informed configurations (rew., sim.), (rew., direct), and (ridge, direct) together with their data-only counterparts under immediate, delayed (\(h=10\)), and batched (\(B=20\)) feedback on \(T=200\) noisy observations. A fixed set of 60 collocation points is available before the stream for physics-informed models; data-only models omit it. Hyperparameters and the clip bound come from an independently generated calibration stream. Same-kernel hindsight references have matching physics access. Four problems, three kernels, six methods, three modes, and five seeds give **1,080 raw rows**, plus **1,260 matched factorial comparisons**. Reports cumulative loss, reference gap, final solution error/residual, runtime, refits, and linear-solve counts.
- **EXP 3 — Spectra and effective dimension.** Compares the RFF surrogate, RBF, and Matérn (\(\nu=1\)) Gram matrices on Poisson-2D and Poisson-4D inputs. These are spectral diagnostics, not realised-regret lower bounds.
- **EXP 4 — Estimation and batching models.** Synthetic direct-Hadamard-sampling error scaling (\(S^{-1/2}\)) versus QAE scaling (\(S^{-1}\)), and causal batching refit/inversion counts. This is a scaling simulation, not a circuit-level QAE implementation.
- **EXP 5 — non-quantum time and modelled quantum cost.** Measures non-quantum Kansa-system assembly/solve time and evaluates HHL/QLSA gate models at measured condition numbers. CPU seconds and gate counts are plotted separately and are never converted into a hardware speedup claim.
- **EXP 6 — Exact commuting ZZ spectrum.** Evaluates the closed-form ZZ fidelity kernel and verifies its finite-support spectrum and bounded effective dimension.
- **EXP-R1/R2 — Real-world regression benchmarks (in `exp/`).** Five data with no governing equation — Household Power, NSL-KDD, Gaze, NO₂, and Istanbul Stock Exchange (ISE) — comparing the quantum-kernel configuration PI-KRR (quant., ridge, sim.) with the kernel-ridge family and six neural baselines. These require external data and their own entry points; see *Real-world regression benchmarks* below.

## Verified five-seed factorial outcome

The completed EXP 1 run contains **340 rows**: four problems, five seeds, and 17 methods. The factorial file contains **380 rows**: 19 comparison/kernel groups, each with 20 problem/seed pairs.

- **Iteration with physics, (rew., direct)/(ridge, direct):** the reweighted configuration wins **35/60** accuracy comparisons and **26/60** residual comparisons. Exact ZZ gives a geometric-mean error ratio of **0.968** and 14/20 accuracy wins, but its residual ratio is 1.010. RBF and RFF geometric-mean error ratios are above one. Iteration is therefore not uniformly superior with physics.
- **Iteration without physics, IKRR/KRR:** IKRR wins **40/60** accuracy comparisons and **31/60** residual comparisons, with aggregated geometric-mean ratios **0.965** and **0.979**. RFF-IKRR has lower mean error on all four problems, but RBF-IKRR is worse on Poisson-2D and Poisson-4D, and the median IKRR/KRR time ratio is **2.23**. The results do not support an “IKRR beats KRR in all ways” claim.
- **Simulated versus direct solve at matched reweighting, (rew., sim.)/(rew., direct):** the simulated-solve configuration wins only **13/60** accuracy and **8/60** residual comparisons. On exact ZZ it has no wins in either metric. The simulated HHL perturbations degrade accuracy relative to the non-quantum two-solve control.
- **Physics ablations:** (ridge, direct, phys)/(ridge, direct, no phys) and (rew., direct, phys)/(rew., direct, no phys) each win **60/60** residual comparisons, and respectively win **54/60** and **53/60** accuracy comparisons. Collocation also enlarges the system and increases measured runtime.
- **Timing:** the median (rew., sim.)/(ridge, direct) time ratio is **1.40**, and the matched two-solve (rew., sim.)/(rew., direct) median is **1.01**. There is no measured wall-clock speed advantage.

## Verified five-seed sequential outcome

The completed EXP 2 run contains **1,080 raw rows** and **1,260 matched comparisons**, covering all four PDEs, all three kernels, six methods per kernel, three feedback modes, and five seeds. All ratios below are geometric means of per-seed cumulative-loss ratios; values below one favour the simulated-solve configuration.

- **(rew., sim.) versus (ridge, direct):** the simulated-solve configuration wins only **22/180** matched comparisons, with aggregate ratio **1.0118**. By problem, the ratios are **1.0065** (Poisson-1D), **0.9986** (AdvecDiff-1D), **1.0176** (Poisson-2D), and **1.0247** (Poisson-4D), with respectively **1/45**, **12/45**, **3/45**, and **6/45** wins. The former Poisson-1D advantage therefore does not persist in the matched same-kernel design.
- **(rew., sim.) versus (rew., direct):** the simulated-solve configuration wins only **14/180** comparisons, with aggregate ratio **1.0079**. Aggregating over problems and kernels gives ratios **1.0210**, **1.0019**, and **1.0011** under immediate, delayed, and batched feedback. There is no matched sequential-loss advantage for the simulated solver.
- **Largest isolated gain:** AdvecDiff-1D with RFF and immediate feedback has (rew., sim.)/(ridge, direct) ratio **0.9641**, but its matched (rew., sim.)/(rew., direct) ratio is **1.0314**. The gain over the non-iterative ridge configuration is therefore associated with iterative regularisation, not simulated HHL.
- **Feedback staleness:** delayed feedback increases mean cumulative loss relative to immediate feedback in **72/72** problem/kernel/method groups; batched feedback also increases loss in **72/72** groups. Immediate, delayed, and batched feedback require respectively 200, 191, and 10 refits; the two-solve methods therefore require 400, 382, and 20 solves.
- **Runtime:** the median (rew., sim.)/(ridge, direct) ratio is **1.45**, and the matched (rew., sim.)/(rew., direct) median is **1.01**. These are measured non-quantum CPU-time ratios and do not indicate quantum speedup.

See the empirical section of the paper for the manuscript-ready interpretation and limitations.

## Real-world regression benchmarks

The suite in `exp/` reproduces the paper's real-world table: five regression data with no governing equation — Household Power, NSL-KDD, Gaze, NO₂, and Istanbul Stock Exchange (ISE). Because no equation is available, the physics constraints are switched off and the quantum-kernel configuration reduces to PI-KRR (quant., ridge, sim.). It is compared with with six neural baselines: a PINN-style MLP (run without physics), MLP, DeepMLP, ResMLP, CNN1D, and LSTM.

### Data setup

The loaders read from `exp/data/` (created on first run if missing):

- **Gaze, NO₂, ISE:** no automatic download. Copy the tracked files from `../data/` into `exp/data/`: `gaze_train.inputs`, `gaze_train.targets`, `gaze_valid.inputs`, `gaze_valid.targets`, `NO2.dat`, and `ISE.rda`. Reading `ISE.rda` additionally requires `pip install pyreadr`.
- **Household Power:** downloaded automatically from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) on first use (~127 MB). To run offline, place `household_power_consumption.txt` in `exp/data/` yourself.
- **NSL-KDD:** downloaded automatically on first use. To run offline, place `nslkdd_train.csv` and `nslkdd_test.csv` in `exp/data/`.

### Running

```bash
cd exp
python main.py                    # EXP-R1 on all five datasets: quantum-kernel configuration + neural baselines
python main.py --dataset gaze     # a single dataset
python run_kernels.py             # EXP-R2: KRR/IKRR (and the coinciding PI rows) on all five datasets
```

`main.py` accepts `--dataset {household,nslkdd,gaze,no2,ise,all}`, `--max-samples` (default 1000, the paper setting), `--n-epochs` (default 100), `--seed` (default 42), and `--results-dir`. `run_kernels.py` takes no arguments and uses the same sample cap, seed, and train/test split, so its rows are directly comparable. Inputs and targets are standardised; Household Power and NSL-KDD are reduced to three principal components; Household Power and ISE use a time-ordered 80/20 split and the other three a random one.

EXP-R1 writes `expR1_accuracy_<dataset>.csv` (test RMSE, \(R^2\), and training time for the quantum-kernel configuration, PINN, and the five neural baselines); EXP-R2 writes `expR2_kernels_<dataset>.csv` (the kernel-ridge rows). Both are written to `exp/results/` by default. The completed runs backing the paper's table are included there.

## Files

| File | Contents |
|---|---|
| `pde.py` | Manufactured PDEs, exact solutions, right-hand sides, and samplers |
| `kernels.py` | RFF, RBF, and exact ZZ kernels with analytic derivative blocks |
| `algos.py` | Algorithm classes for the PI-KRR family (ridge/reweighted, with/without physics, direct/simulated solves), the data-only KRR/IKRR variants, PINN, causal updates, and the HHL cost simulator |
| `experiments.py` | EXP 1–6, validation-only tuning, factorial comparisons, CSV export |
| `make_figures.py` | Paper figures from `results/*.csv` |
| `main.py` | Command-line entry point for the PDE suite |
| `exp/` | Real-world benchmark suite: data loaders, quantum-kernel and kernel-ridge regressors, neural baselines, and its own `main.py` / `run_kernels.py` entry points |


## Usage

Install the dependencies:

```bash
pip install numpy scipy pandas matplotlib scikit-learn torch
```

Run the full PDE suite (EXP 1 and EXP 2 save checkpoints after every completed problem/seed and resume automatically after interruption):

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
| `exp/results/expR1_accuracy_*.csv` | Real-world RMSE, \(R^2\), and training time for the quantum-kernel configuration, PINN, and neural baselines |
| `exp/results/expR2_kernels_*.csv` | Real-world kernel-ridge rows (KRR/IKRR and the coinciding PI rows) |
| `figures/accuracy_benchmark.pdf` | Same-kernel, same-iteration (rew., sim.)/(rew., direct) accuracy ratios |
| `figures/iteration_benchmark.pdf` | (rew., direct)/(ridge, direct) accuracy ratios with and without physics |
| `figures/regret_curves.pdf` | Four-PDE PI-KRR (RFF, rew., sim.) cumulative-loss trajectories under all feedback modes |
| `figures/sequential_factorial.pdf` | Cross-problem (rew., sim.)/(ridge, direct) and (rew., sim.)/(rew., direct) sequential loss ratios |

### Legacy labels in archived outputs

The archived CSVs, the `QIKRR_RESULTS_DIR`/`QIKRR_FIGURE_DIR` environment variables, and the class names in `algos.py` and `exp/algos.py` predate the paper's configuration naming and keep the original short labels. The `method` column of the PDE-suite CSVs maps to the paper configurations as follows (for kernel `K` ∈ {exact ZZ, RBF, RFF}):

- `QIKRR (K)` → PI-KRR (K, rew., sim.); `QIKRR (RFF, no phys)` → IKRR (RFF, rew., sim.)
- `PI-IKRR (K)` → PI-KRR (K, rew., direct)
- `PI-KRR (K)` → PI-KRR (K, ridge, direct)
- `IKRR (K, no phys)` → IKRR (K, rew., direct)
- `KRR (K, no phys)` → KRR (K, ridge, direct)

The real-world CSVs in `exp/results/` use their own legacy labels: `QKRR` is PI-KRR (quant., ridge, sim.), and `KRR`, `IKRR`, `PI-KRR`, `PI-IKRR` are the paper's KRR (RBF, ridge, direct), IKRR (RBF, rew., direct), PI-KRR (RBF, ridge, direct), and PI-KRR (RBF, rew., direct) rows.

## Simulation and reporting limitations

- The RFF backend is a finite-dimensional surrogate, not an exact simulation of a quantum circuit.
- The exact ZZ backend evaluates a closed-form fidelity kernel and its analytic derivatives; no quantum circuit is executed.
- `HHLSimulator` performs `numpy.linalg.solve`, adds optional precision-scaled noise, and accumulates the gate model \(\log_2(n)\kappa^2\log_2(1/\epsilon)\). For nonsymmetric Kansa matrices, a quantum implementation would require a standard Hermitian dilation or an equivalent non-Hermitian linear-system construction; the model suppresses its constant overhead.
- Iterative reweighting performs `n_reweight=2` linear solves per refit by default. Both refits and actual linear solves are reported; batching reduces refits by the batch factor while the constant reweighting multiplier remains visible.
- Quantum gate estimates depend on conditioning and input-model assumptions. They are not wall-clock measurements and do not establish an unconditional end-to-end advantage.
- Solver/noise claims must use the **same-kernel (rew., sim.)/(rew., direct) comparison**; iterative-regression claims must use **(rew., direct)/(ridge, direct)** with physics or **IKRR/KRR** without physics. Kernel and physics effects must not be attributed to quantum computation.
- Common hyperparameters provide controlled comparisons, not globally optimised performance for any method.

The PDE suite (EXP 1–6) requires no external data: all observations and collocation points are generated from the manufactured PDEs, and seeds make the runs reproducible. The real-world suite in `exp/` does require the data described above.