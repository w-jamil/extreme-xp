# QKRR vs Neural Baselines — Real-World Regression

Experimental code for the real-world regression benchmarks in the paper *Quantum Kernel Ridge Regression for Physics-Informed Learning under Delayed and Batched Feedback*.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all five real-world benchmarks (default)
python main.py

# Run a single dataset
python main.py --dataset household

# View the paper results table
jupyter notebook example.ipynb
```

## Datasets

| Dataset | Files | Split |
|---|---|---|
| Household | `data/household_power_consumption.txt` | Time-ordered 80/20 (first 80% train) |
| NSL-KDD | `data/nslkdd_train.csv`, `data/nslkdd_test.csv` | Random 80/20 |
| Gaze | `data/gaze_train.*`, `data/gaze_valid.*` | Random 80/20 |
| NO₂ | `data/NO2.dat` | Random 80/20 |
| ISE | `data/ISE.rda` | Time-ordered 80/20 (daily returns) |

## Output

`main.py` writes one CSV per dataset to `./results/`:

- `expR1_accuracy_{dataset}.csv` — Test-set RMSE and $R^2$ for all methods.

## Tuning

QKRR hyperparameters for each dataset were selected by 3-fold cross-validation. The final tuned values are stored in `_qkrr_for_dataset()` in `experiments.py`. To re-run the search:

```bash
python tune_qkrr.py --dataset gaze
python tune_qkrr_fine.py --dataset gaze --best "n_qubits=8,length_scale=10.0,beta=0.05,lambda_reg=1e-5"
```

## Notes on Quantum Simulation

- **Quantum kernel:** Simulated via high-dimensional structured feature map ($D = 2^{n_{\text{qubits}}}$) with exponential spectral decay.
- **HHL:** Simulated by classical inversion + condition-number-proportional noise + gate-count tracking.
- These are standard practices for quantum ML benchmarking on classical hardware.
