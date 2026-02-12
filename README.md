# Extreme-XP: Machine Learning Experiment Suite

A comprehensive machine learning experiment suite featuring **Batch Learning**, **Continual Learning**, and **Online Learning** approaches. This suite implements multiple machine learning paradigms.

## Experiment Types

### 1. OnlineToBatch Learning Protocol
**Location**: `batch/` directory

Novel hybrid approach that combines online learning benefits with batch validation:
- **Algorithms**: Regression and classification algorithms
- **Approach**: Epoch-based training with data shuffling for stochasticity
- **Validation**: Held-out data to select optimal model weights
- **Datasets**: Simulated + Financial + cybersecurity + Image datasets for experimentation

### 2. Continual Learning
**Locations**: `cl_case1/` and `cl_case2/` directories

Sequential learning with catastrophic forgetting mitigation:
- **Algorithms**: 7 algorithms (PassiveAggressive, Perceptron, GradientLearning, AROW, RDA, SCW, AdaRDA)
- **Case 1**: Task-agnostic continual learning approach
- **Case 2**: Sliding window continual learning paradigm  
- **Datasets**: All RBD24 cybersecurity datasets processed sequentially

### 3. Online Learning  
**Location**: `online/` directory

Online data processing:
- **Algorithms**: 7 algorithms (PassiveAggressive, Perceptron, GradientLearning, AROW, RDA, SCW, AdaRDA)
- **Approach**: Adaptive learning from data streams
- **Datasets**: All RBD24 cybersecurity datasets + Kaggle credit card fraud


## How to Run Experiments

Navigate to the `master-experiment-suite/` directory and use the provided scripts:

**Run all experiments sequentially:**
```bash
cd master-experiment-suite/
bash run_all.sh
```

**Run specific experiment types:**
```bash
bash run_batch.sh      # OnlineToBatch learning (RBD24 + MNIST)
bash run_continual.sh  # Continual learning (both cases)  
bash run_online.sh     # Online learning (RBD24 + Kaggle)
```

## Results and Analysis

Results are automatically saved to respective `results/` directories:
- **OnlineToBatch**: `batch/results/`
  - `onlinetobatch_individual_results.csv` - RBD24 cybersecurity results
  - `mnist_binary_even_odd.csv` - MNIST binary classification results
- **Continual Learning**: `cl_case1/results/` and `cl_case2/results/`
  - `cl_case1_results.csv` - Task-agnostic continual learning
  - `cl_case2_results.csv` - Sliding window continual learning
- **Online Learning**: `online/results/`
  - `online_results.csv` - RBD24 + Kaggle fraud detection results


### Directory Structure
```
extreme-xp/
├── README.md                         # Comprehensive project documentation
├── .gitignore                       # Git ignore patterns for datasets/results
└── master-experiment-suite/
    ├── algorithmic_allocation       # Scripts (experiment)
    ├── regression_experiments       # Scripts and csv files (experiment results)
    ├── .gitignore                   # Additional ignore patterns
    ├── docker-compose.yaml          # Docker orchestration for all experiments
    ├── run_all.sh                   # Run all experiments sequentially
    ├── run_batch.sh                 # OnlineToBatch experiments only
    ├── run_continual.sh             # Continual learning experiments
    ├── run_online.sh                # Online learning experiments
    ├── batch/                       # OnlineToBatch Learning
    │   ├── Dockerfile               # Docker container configuration
    │   ├── requirements.txt         # Python dependencies
    │   ├── src/
    │   │   ├── batch_sim.py        # RBD24 cybersecurity experiments
    │   │   ├── mnist_sim.py        # MNIST binary classification
    │   │   ├── hybrid_sim.py       # Hybrid learning approaches
    │   │   ├── algorithms.py       # Algorithm implementations
    │   │   └── data_handler.py     # Data loading utilities
    │   ├── results/                # Generated experiment results
    │   └── cyber/                  # RBD24 datasets (auto-downloaded)
    ├── cl_case1/                   # Continual Learning Case 1 
    │   ├── Dockerfile               # Docker container configuration
    │   ├── requirements.txt         # Python dependencies
    │   ├── src/
    │   │   ├── case1_sim.py        # Main continual learning experiment
    │   │   ├── algorithms.py       # Algorithm implementations
    │   │   ├── data_handler.py     # Data loading utilities
    │   │   ├── data_processor.py   # Task generation utilities
    │   │   ├── evaluation.py       # Performance evaluation
    │   │   └── [additional experiment files]
    │   ├── results/                # Generated experiment results
    │   └── cyber/                  # RBD24 datasets (auto-downloaded)
    ├── cl_case2/                   # Continual Learning Case 2
    │   ├── Dockerfile               # Docker container configuration
    │   ├── requirements.txt         # Python dependencies
    │   ├── src/
    │   │   ├── case2_sim.py        # Sliding window continual learning
    │   │   └── [supporting files]
    │   ├── results/                # Generated experiment results
    │   └── cyber/                  # RBD24 datasets (auto-downloaded)
    └── online/                     # Online Learning
        ├── Dockerfile               # Docker container configuration
        ├── requirements.txt         # Python dependencies
        ├── src/
        │   ├── online.py           # Main online learning experiment
        │   └── data_handler.py     # Data loading + Kaggle integration
        ├── results/                # Generated experiment results
        ├── cyber/                  # RBD24 datasets (auto-downloaded)
        └── data/                   # Kaggle datasets (auto-downloaded)
```

## Prerequisites

- **Python 3.8+** with virtual environment
- **Docker and docker-compose** (for continual and online learning experiments)
- **Internet connection** (for automatic dataset downloads)

The experiments will automatically download required datasets:
- RBD24 cybersecurity datasets from Zenodo
- Kaggle credit card fraud dataset (requires kagglehub)
- MNIST dataset from OpenML

## Dataset Overview

- **RBD24**: 12 cybersecurity datasets with temporal patterns and user behavior
- **Kaggle Credit Card Fraud**: European credit card transaction dataset with fraud labels
- **MNIST**: Handwritten digit images converted to binary classification tasks (even vs odd digits)

All datasets are automatically downloaded and processed during experiment execution.

