# Constraint-Aware Machine Learning

Constraint-aware machine learning derives the learning algorithm from a constrained optimisation problem. This repository contains experiments spanning three learning paradigms — batch, continual and online — with ongoing research into delayed and batched feedback constraints in online learning.

---

## Repository Structure

```
online-classificaton/  Core scripts for the primary research work
- dash/                Flask-based interactive simulation dashboard
- data/                Data (parquet files)
- results/             Output CSVs and plots
caml-initial-study/    Phase 1 "Initial Study" experiments (Deliverable 3.2)
```

Please download data from [this](https://drive.google.com/file/d/1jixn6DH4HKCH4Yw_2wT5h-S-teBvJET-/view?usp=sharing) link and add the parquet files to online-classificaton/data/

## Research Experiments

All benchmarking scripts are in the `online-classificaton/` directory.

### Running All Experiments
```bash
cd online-classificaton
python main.py --experiment all  # or: python main.py
```

### Simulation
Compares standard against kernel-based classifiers on 11 synthetic datasets.
```bash
cd online-classificaton
python main.py --experiment simulation
```

### Benchmark


## Classification

Online learning experiments on MNIST, Kaggle Credit Fraud, and Cybersecurity datasets.
```bash
cd online-classificaton
python main.py --experiment online
```

---

## Regression — Delayed & Batched Feedback

Online learning regression experiments studying CIRR, K-CIRR, Q-CIRR and QK-CIRR under delayed and batched feedback constraints.

### Running All Experiments
```bash
cd online-regression
pip install -r requirements.txt
python main.py --experiment all --max-samples 5000 --results-dir results
```

### Individual Experiments
```bash
python main.py --experiment delayed   # EXP1: UCI Household Power, variable delay
python main.py --experiment batched   # EXP2: NSL-KDD, pack sizes K=1..50
python main.py --experiment corollary # EXP3: concentrated vs perpetual delay
```

Results (CSV tables + plots) are written to `online-regression/results/`.

> **Data:** NSL-KDD CSVs are included in `online-regression/data/`. The UCI Household Power Consumption dataset (`household_power_consumption.txt`, ~127 MB) must be downloaded separately from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) and placed in `online-regression/data/`.

---

## Interactive Dashboard

A Flask application for visualising online classification simulations.

```bash
cd online-classificaton/dash
python app.py
```

Then open `http://127.0.0.1:5000` in a browser.

---

## Initial Study

The `caml-initial-study/` directory preserves the foundational codebase for Deliverable 3.2 (D3.2). This study established the experimental groundwork.

---

## Prerequisites

- Python 3.8+
- Install dependencies: `pip install -r online-classificaton/requirements.txt`
  - Required: `numpy`, `pandas`, `scikit-learn`, `numba`, `scipy`, `matplotlib`, `flask`, `pyarrow`

---

## Related working papers

* **Jamil, W.** and Bouchachia, A. Online Gradient-based Learning. *In progress.*
* **Jamil, W.** and Bouchachia, A. Online learning with Delayed & Batched feedback. *In progress.*

---
