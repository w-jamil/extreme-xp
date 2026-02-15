# Extreme-XP: Machine Learning Experiment Suite

A comprehensive machine learning experiment suite featuring **Batch Learning**, **Continual Learning**, and **Online Learning** approaches. This suite implements multiple machine learning paradigms on simulated, financial, cybersecurity and image dataset(s).

## Experiment Types

### 1. OnlineToBatch Learning Protocol
**Location**: `batch/` directory

Novel hybrid approach that combines online learning benefits with batch validation:
- **Algorithms**: Regression and classification algorithms
- **Approach**: Epoch-based training with data shuffling for stochasticity
- **Validation**: Held-out data to select optimal model weights

### 2. Continual Learning
**Locations**: `cl_case1/` and `cl_case2/` directories

Sequential learning with catastrophic forgetting mitigation:
- **Algorithms**: Regression and classification algorithms
- **Case 1**: Task-agnostic continual learning approach
- **Case 2**: Sliding window continual learning paradigm  


### 3. Online Learning  
**Location**: `online/` directory

Online data processing:
- **Algorithms**: Regression and classification algorithms
- **Approach**: Adaptive learning from data streams



## How to Run Experiments

Navigate to the `master-experiment-suite/` directory and use the provided scripts:

**Run all experiments sequentially:**
```bash
cd master-experiment-suite/
bash run_all.sh
```

**Run specific experiment types:**
```bash
bash run_batch.sh      
bash run_continual.sh  
bash run_online.sh    
```

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
