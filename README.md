# Constraint-Aware Machine Learning

Constraint-aware machine learning derives the learning algorithm from a constrained optimisation problem. This repository contains experiments spanning three learning paradigms — batch, continual, and online — with ongoing research into online learning under delayed and batched feedback.

---

## Repository Structure

```
research-experiments/    Core scripts for the primary research work
experiment-dashboard/    Flask-based interactive visualisation tool
caml-initial-study/      Phase 1 "Initial Study" experiments (Deliverable 3.2)
```

---

## Research Experiments

All benchmarking scripts are in the `research-experiments/` directory.

### Kernel vs. Standard Simulation
Compares standard against kernel-based classifiers on simulated data in an online setting.
```bash
cd research-experiments
python comparison_simulation.py
```

### Online Learning
Online data processing and adaptive learning experiments on published datasets.
```bash
cd research-experiments
python online.py
```

### Kernel-Based Learning
Experiments on high-dimensional kernel mappings for non-linear decision boundaries on published datasets.
```bash
cd research-experiments
python kernel_experiments.py
```

---

## Interactive Dashboard

A Flask application for visualising experiment results interactively.

```bash
cd experiment-dashboard
python app.py
```

Then open `http://127.0.0.1:5000` in a browser.

---

## Initial Study (Phase 1)

The `caml-initial-study/` directory contains the codebase for **Deliverable 3.2**, which established the baseline for constraint-aware online learning. See `caml-initial-study/README.md` for setup and historical results.

---

## Prerequisites

- Python 3.8+
- Install dependencies: `pip install -r requirements.txt`
  - Required: `numpy`, `scikit-learn`, `pandas`, `flask`, `matplotlib`, `scipy`, `seaborn`
- Docker recommended for isolated execution

---

## Citation

> **Jamil, W.** and Bouchachia, A. Online Gradient-based Learning. *In progress.*

---
