# Batch, Continual and Online Learning Evaluation

A machine learning experiment suite featuring **Batch Learning**, **Continual Learning**, and **Online Learning** approaches. This suite implements multiple machine learning paradigms using RBD24 datasets. 

## Experiment Types

### 1. OnlineToBatch Protocol
**Location**: `batch/src/batch_sim.py`

Novel hybrid approach that combines online learning benefits with batch validation:
- Trains 7 algorithms: PassiveAggressive, Perceptron, GradientLearning, AROW, RDA, SCW, AdaRDA
- Uses epoch-based training with data shuffling for stochasticity
- Validates on held-out data to select optimal model weights
- Weighted majority voting ensemble optimized for recall
- Processes all 12 RBD24 cybersecurity datasets

### 2. Continual Learning
**Location**: `cl_case1/` and `cl_case2/` directories

Novel Sequential learning with catastrophic forgetting mitigation

- 7 algorithms: PassiveAggressive, Perceptron, GradientLearning, AROW, RDA, SCW, AdaRDA
- Task-agnostic continual learning approaches
- Formulating sliding window in continual learning paradigm  
- Processes all 12 RBD24 cybersecurity datasets collectively and individually


### 3. Online Learning  
**Location**: `online/` directory

Comparing novel algorithm with state-of-the-art

- 7 algorithms: PassiveAggressive, Perceptron, GradientLearning, AROW, RDA, SCW, AdaRDA
- Adaptive learning from data streams  
- Processes all 12 RBD24 cybersecurity datasets


### Ensemble Method
- **WeightedMajorityVoter**: Combines algorithms using recall-based weights
- Optimizes for False Negative minimization
- Critical for cybersecurity threat detection

## Results and Analysis

Results are automatically saved to `master-experiment-suite/batch/results/`:
- `onlinetobatch_individual_results.csv` - Individual algorithm performance
- `onlinetobatch_ensemble_comparison.csv` - Ensemble vs best individual comparison


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

## Experiments

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

