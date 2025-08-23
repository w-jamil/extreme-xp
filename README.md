# Batch, Continual and Online Learning Evaluation

A machine learning experiment suite featuring **Batch Learning**, **Continual Learning**, and **Online Learning** approaches.

## Overview

This suite implements and compares multiple machine learning paradigms using RBD24 datasets. 

## Experiment Types

### 1. OnlineToBatch Protocol (Default)
**Location**: `batch/src/batch_sim.py`

Novel hybrid approach that combines online learning benefits with batch validation:
- Trains 7 algorithms: PassiveAggressive, Perceptron, GradientLearning, AROW, RDA, SCW, AdaRDA
- Uses epoch-based training with data shuffling for stochasticity
- Validates on held-out data to select optimal model weights
- Weighted majority voting ensemble optimized for recall
- Processes all 12 RBD24 cybersecurity datasets

### 2. Continual Learning
**Location**: `cl_case1/` and `cl_case2/` directories

Sequential learning with catastrophic forgetting mitigation

### 3. Online Learning  
**Location**: `online/` directory

Real-time adaptive learning from data streams

##  Data and Algorithms

### Datasets
- **RBD24**: 12 real cybersecurity datasets (desktop/smartphone variants)
  - OutTLS, NonEnc, Phishing, OutFlash, P2P, Crypto
  - Automatic download from Zenodo if not present

### OnlineToBatch Algorithms
1. **PassiveAggressive** - Margin-based online learning
2. **Perceptron** - Classic classifier  
3. **GL** - Gradient Learning
4. **AROW** - Adaptive Regularization of Weight Vectors
5. **RDA** - Regularized Dual Averaging
6. **SCW** - Soft Confidence-Weighted learning
7. **AdaRDA** - Adaptive Regularized Dual Averaging

### Ensemble Method
- **WeightedMajorityVoter**: Combines algorithms using recall-based weights
- Optimizes for False Negative minimization
- Critical for cybersecurity threat detection

## Results and Analysis

Results are automatically saved to `master-experiment-suite/batch/results/`:
- `onlinetobatch_individual_results.csv` - Individual algorithm performance
- `onlinetobatch_ensemble_comparison.csv` - Ensemble vs best individual comparison

### Key Metrics
- **FNR** (False Negative Rate) - Primary optimization target
- **FPR** (False Positive Rate) - Secondary consideration
- **Recall** - Inverse of FNR, maximized for threat detection
- **Accuracy** - Overall correctness

## Architecture

### OnlineToBatch Protocol Flow
1. **Data Loading** - RBD24 datasets with user-based splitting
2. **Training Loop** - For each algorithm and epoch:
   - Shuffle training data
   - Train algorithm incrementally  
   - Validate on held-out data
   - Save best weights based on recall
3. **Ensemble Creation** - WeightedMajorityVoter with recall-based weighting
4. **Evaluation** - Test on unseen data, compare individual vs ensemble

### Directory Structure
```
extreme-xp/
├── README.md                         # Comprehensive project documentation
└── master-experiment-suite/
    ├── .gitignore                   # Git ignore patterns
    ├── docker-compose.yaml          # Docker orchestration
    ├── run_all.sh                   # Run all experiments sequentially
    ├── run_batch.sh                 # OnlineToBatch experiments  
    ├── run_continual.sh             # Continual learning experiments
    ├── run_online.sh                # Online learning experiments
    ├── batch/                       # OnlineToBatch experiments
    │   ├── Dockerfile               # Docker container configuration
    │   ├── requirements.txt         # Python dependencies
    │   ├── src/
    │   │   ├── batch_sim.py        # Main OnlineToBatch experiment
    │   │   └── data_handler.py     # RBD24 data loading utilities
    │   ├── results/                # Experiment output files
    │   └── cyber/                  # RBD24 datasets (auto-downloaded)
    ├── cl_case1/                   # Continual Learning Case 1
    │   ├── Dockerfile               # Docker container configuration
    │   ├── requirements.txt         # Python dependencies
    │   ├── src/
    │   │   ├── case1_sim.py        # Main continual learning experiment
    │   │   ├── algorithms.py       # Algorithm implementations
    │   │   ├── data_handler.py     # Data loading utilities
    │   │   ├── evaluation.py       # Performance evaluation
    │   │   └── [other experiment files]
    │   ├── results/                # Experiment output files
    │   └── cyber/                  # RBD24 datasets (auto-downloaded)
    ├── cl_case2/                   # Continual Learning Case 2
    │   ├── Dockerfile               # Docker container configuration
    │   ├── requirements.txt         # Python dependencies
    │   ├── src/                    # Source code and utilities
    │   ├── results/                # Experiment output files
    │   └── cyber/                  # RBD24 datasets (auto-downloaded)
    └── online/                     # Online Learning experiments
        ├── Dockerfile               # Docker container configuration
        ├── requirements.txt         # Python dependencies
        ├── src/
        │   ├── online.py           # Main online learning experiment
        │   └── data_handler.py     # Data loading utilities
        ├── results/                # Experiment output files
        └── cyber/                  # RBD24 datasets (auto-downloaded)
```

## Technical Details

### OnlineToBatch Innovation
- **Problem**: Online algorithms sensitive to data order, batch algorithms ignore temporal patterns
- **Solution**: Hybrid approach with controlled stochasticity via epoch shuffling
- **Validation**: Hold-out data prevents overfitting while optimizing for recall
- **Ensemble**: Weighted combination based on individual algorithm strengths


## Key Features

- Implementation with 7 algorithms
- Real datasets (RBD24 - 12 variants)  
- False Negative optimization via recall-focused ensemble
- Documentation and clean code structure
- Comprehensive results with detailed FNR/FPR analysis
- Automatic data handling with Zenodo downloads
---

Sequential learning scenarios:
- Case 1: Standard continual learning with forgetting analysis
- Case 2: Advanced continual adaptation
- Catastrophic forgetting mitigation techniques
- Per-task performance tracking

### 4. Online Learning
**Location**: `online/` directory

Real-time adaptive learning:
- Stream-based algorithm updates
- Immediate adaptation to new threats
- Minimal computational overhead
- Suitable for production deployment

### Experiments

**Run all experiments:**
```bash
bash run_all.sh
```

**Run specific experiment:**
```bash
bash run_batch.sh      # Batch learning
bash run_continual.sh  # Continual learning  
bash run_online.sh     # Online learning
```

