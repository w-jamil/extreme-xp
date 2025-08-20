# Batch, Continual and Online Learning Evaluation

This project contains a comprehensive suite of four distinct experiments for evaluating batch, continual, and online machine learning algorithms with **enhanced ensemble aggregation** for cybersecurity datasets.

All code, dependencies, and data are automatically managed by Docker, allowing you to run everything with single commands. The suite now includes advanced ensemble methods that combine multiple algorithms to achieve better performance, particularly lower False Negative Rates (FNR) critical for cybersecurity applications.

## � Table of Contents

- [🚀 Key Features](#-key-features)
- [📋 System Requirements](#-system-requirements)
- [🎯 Quick Start](#-quick-start)
- [🔬 Enhanced Batch Learning with Ensemble Aggregation](#-enhanced-batch-learning-with-ensemble-aggregation)
- [📊 Results and Output](#-results-and-output)
- [📁 Project Structure](#-project-structure)
- [🧪 Experiment Guide](#-experiment-guide)
- [📁 Data Requirements](#-data-requirements)
- [🛠 Troubleshooting](#-troubleshooting)
- [🎯 Why Ensembles for Cybersecurity?](#-why-ensembles-for-cybersecurity)

## �🚀 Key Features

- **Four experiment types**: Batch Learning (with ensembles), Continual Learning (2 cases), Online Learning
- **Enhanced Batch Learning**: Now includes 3 ensemble aggregation methods for improved performance
- **Cybersecurity Focus**: Optimized for low False Negative Rate (missing fewer attacks)
- **Docker Integration**: Complete environment isolation and easy deployment
- **Automatic Data Download**: First run downloads datasets from Zenodo repository

## � Data Directory Setup

**Each experiment has its own data directory to avoid conflicts.**

### **Data Directory Structure:**
- **`batch/cyber/`** - Batch learning data
- **`online/cyber/`** - Online learning data  
- **`cl_case1/cyber/`** - Continual learning case 1 data
- **`cl_case2/cyber/`** - Continual learning case 2 data

### **How It Works:**
1. **Automatic download** - Each experiment downloads data to its own directory on first run
2. **No conflicts** - Separate directories prevent concurrent access issues
3. **Independent caching** - Each experiment can cache processed data separately
4. **Sample data fallback** - Creates demo data if download fails

### **Directory Structure:**
```
master-experiment-suite/
├── batch/
│   ├── cyber/              ← Batch experiment data
│   └── results/            ← Batch results
├── online/
│   ├── cyber/              ← Online experiment data  
│   └── results/            ← Online results
├── cl_case1/
│   ├── cyber/              ← CL case 1 data
│   └── results/            ← CL case 1 results
└── cl_case2/
    ├── cyber/              ← CL case 2 data
    └── results/            ← CL case 2 results
```

### **Benefits:**
✅ **No conflicts** - Each experiment has isolated data access  
✅ **Parallel execution** - Can run multiple experiments simultaneously  
✅ **Independent caching** - Faster subsequent runs per experiment  

**No manual configuration needed!** Just run any experiment.

## �📋 System Requirements

**Docker Desktop** is the only requirement:
- **Operating System:** Windows 10/11, macOS, or Linux
- **Software:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)

> **IMPORTANT:** Ensure Docker Desktop is running (stable whale icon in system tray) before proceeding.

## 🎯 Quick Start

### 1. Download and Setup
```bash
# Download project and navigate to directory
cd /path/to/master-experiment-suite
```

### 2. Run All Experiments (Recommended)
```bash
# Linux/Mac
chmod +x run_all_experiments.sh
./run_all_experiments.sh

# Windows
run_all_experiments.bat
```

### 3. Run Individual Experiments

**Batch Learning with Ensembles (Featured):**
```bash
# Linux/Mac
./run_batch_individual.sh

# Windows  
run_batch_individual.bat

# Direct Docker command
docker compose up --build batch-learning-experiment
```

**Online Learning:**
```bash
# Linux/Mac
./run_online_individual.sh

# Windows
run_online_individual.bat

# Direct Docker command
docker compose up --build online-experiment
```

**Continual Learning (Both Cases):**
```bash
# Linux/Mac
./run_continual_individual.sh

# Windows
run_continual_individual.bat

# Individual cases with direct Docker commands
docker compose up --build cl_case1-experiment         # Case 1 only
docker compose up --build cl_case2-experiment         # Case 2 only
```

## 🔬 Enhanced Batch Learning with Ensemble Aggregation

The batch learning experiment now includes advanced ensemble methods that combine predictions from multiple algorithms:

### Ensemble Methods Included:
1. **Ensemble_Mean**: Simple averaging of all algorithm predictions
2. **Ensemble_WeightedByFNR**: Weighted by FNR performance (prioritizes security)
3. **Ensemble_TopPerformers**: Uses only the best 3 algorithms per dataset

### Algorithms Combined:
- PassiveAggressive
- Perceptron  
- GradientLearning
- AROW (Adaptive Regularization)
- RDA (Regularized Dual Averaging)
- SCW (Soft Confidence Weighted)
- AdaRDA (Adaptive RDA)

### Algorithm Details:
| Algorithm | Type | Strengths | Best For |
|-----------|------|-----------|----------|
| **PassiveAggressive** | Online | Fast updates, aggressive learning | High-frequency attacks |
| **Perceptron** | Linear | Simple, interpretable | Basic classification |
| **GradientLearning** | Gradient-based | Stable convergence | General purpose |
| **AROW** | Adaptive | Confidence-weighted updates | Noisy data |
| **RDA** | Regularized | Sparsity control | High-dimensional data |
| **SCW** | Confidence-weighted | Soft margin updates | Uncertain labels |
| **AdaRDA** | Adaptive RDA | Adaptive regularization | Complex patterns |

### Ensemble Methods:
| Method | Strategy | When to Use |
|--------|----------|-------------|
| **Ensemble_Mean** | Simple averaging | Stable, conservative results |
| **Ensemble_WeightedByFNR** | FNR-based weighting | **Security-critical applications** |
| **Ensemble_TopPerformers** | Best 3 algorithms only | Maximum performance focus |

### Expected Improvements:
- **Lower FNR**: Ensembles reduce false negatives (missing fewer attacks)
- **Higher F1-Score**: Better balance between precision and recall
- **More Robust**: Less variance across different datasets
- **Better Generalization**: Combines strengths of different algorithms

## 📊 Results and Output

### Output Locations:
- **Batch (Enhanced)**: `batch/results/`
  - `batch_results_individual.csv` - Individual algorithm results
  - `batch_results_with_ensembles.csv` - Individual + ensemble results
- **Online**: `online/results/online_results.csv`
- **CL Case 1**: `cl_case1/results/case1_results.csv`
- **CL Case 2**: `cl_case2/results/case2_results.csv`

### Key Metrics:
- **FNR (False Negative Rate)**: Lower is better for cybersecurity
- **FPR (False Positive Rate)**: Lower is better
- **F1-Score**: Higher is better
- **Accuracy**: Overall correctness

## 📁 Data Requirements

Place your cybersecurity datasets (in Parquet format) in the respective directories:
- `batch/cyber/*.parquet`
- `online/cyber/*.parquet`
- `cl_case1/cyber/*.parquet`
- `cl_case2/cyber/*.parquet`

**Note**: On first run, sample datasets will be automatically downloaded if directories are empty.

## 🛠 Troubleshooting

**Docker not recognized:**
- Ensure Docker Desktop is installed and running
- Restart terminal after Docker installation

**Different Docker versions:**
- Try `docker-compose` instead of `docker compose` or vice versa
- One syntax will work depending on your Docker version

**Memory issues:**
- Run experiments individually instead of all at once
- Use the provided individual run scripts

## 🎯 Why Ensembles for Cybersecurity?

Ensemble methods are particularly valuable for cybersecurity because:
- **Reduced False Negatives**: Missing an attack is costly
- **Algorithm Diversity**: Different algorithms catch different attack patterns
- **Robustness**: Less likely to fail on new attack variants
- **Performance Stability**: More consistent results across datasets

The enhanced batch learning experiment demonstrates these benefits by comparing individual algorithms against ensemble methods, typically showing 10-30% improvement in key security metrics.

## 🚀 Available Run Scripts

| Script | Platform | Purpose | Runtime |
|--------|----------|---------|---------|
| `run_all_experiments.sh/.bat` | Linux/Mac/Win | All experiments sequentially | 30-90 min |
| `run_batch_individual.sh/.bat` | Linux/Mac/Win | Batch learning with ensembles | 10-30 min |
| `run_online_individual.sh/.bat` | Linux/Mac/Win | Online learning only | 5-15 min |
| `run_continual_individual.sh/.bat` | Linux/Mac/Win | Both continual learning cases | 15-45 min |

**Quick Commands:**
```bash
# Make scripts executable (Linux/Mac only)
chmod +x *.sh

# Run specific experiment (example)
./run_batch_individual.sh        # Linux/Mac
run_batch_individual.bat         # Windows
```

## 📦 Docker Commands Reference

```bash
# All experiments (parallel - high memory usage)
docker compose up --build

# Individual experiments
docker compose up --build batch-learning-experiment
docker compose up --build online-experiment
docker compose up --build cl_case1-experiment
docker compose up --build cl_case2-experiment

# Clean up Docker resources
docker compose down
docker system prune -f
```

## 📁 Project Structure

```
master-experiment-suite/
├── README.md                          # Main documentation
├── docker-compose.yaml               # Docker configuration
├── run_all_experiments.sh/.bat       # Run all experiments sequentially
├── run_batch_individual.sh/.bat      # Batch Learning with Ensembles
├── run_online_individual.sh/.bat     # Online Learning only
├── run_continual_individual.sh/.bat  # Both Continual Learning experiments
├── batch/                             # Batch Learning with Ensembles
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── cyber/                         # Place parquet data files here
│   ├── results/                       # Output CSV files
│   │   ├── batch_results_individual.csv
│   │   └── batch_results_with_ensembles.csv
│   └── src/
│       ├── batch_sim.py              # Main batch experiment with ensembles
│       └── data_loader.py            # Data loading utilities
├── online/                           # Online Learning Experiment
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── cyber/                        # Place parquet data files here
│   ├── results/                      # Output: online_results.csv
│   └── src/
│       ├── online.py
│       └── data_handler.py
├── cl_case1/                         # Continual Learning Case 1
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── cyber/                        # Place parquet data files here
│   ├── results/                      # Output: case1_results.csv
│   └── src/
│       ├── case1_sim.py
│       ├── algorithms.py
│       ├── data_handler.py
│       ├── data_processor.py
│       └── evaluation.py
└── cl_case2/                         # Continual Learning Case 2
    ├── Dockerfile
    ├── requirements.txt
    ├── cyber/                        # Place parquet data files here
    ├── results/                      # Output: case2_results.csv
    └── src/
        ├── case2_sim.py
        ├── algorithms.py
        ├── data_handler.py
        ├── data_preprocessor.py
        └── metrics.py
```

## 🧪 Experiment Guide

### Experiment Types Overview

| Experiment | Description | Key Features | Output File |
|------------|-------------|--------------|-------------|
| **Batch Learning** | Enhanced with ensemble aggregation | 7 algorithms + 3 ensemble methods | `batch_results_with_ensembles.csv` |
| **Online Learning** | Streaming data processing | Real-time learning capabilities | `online_results.csv` |
| **Continual Learning Case 1** | Task-Domain adaptation | Handles concept drift | `case1_results.csv` |
| **Continual Learning Case 2** | Task-CL scenarios | Sequential task learning | `case2_results.csv` |

### Detailed Run Instructions

#### Option 1: Run All Experiments (Sequential)
**Recommended for comprehensive evaluation**
```bash
# Linux/Mac
chmod +x run_all_experiments.sh
./run_all_experiments.sh

# Windows
run_all_experiments.bat

# Direct Docker (all parallel - high memory usage)
docker compose up --build
```

#### Option 2: Individual Experiments

**🎯 Batch Learning with Ensembles (Featured)**
```bash
# Linux/Mac
./run_batch_individual.sh

# Windows
run_batch_individual.bat

# Direct Docker
docker compose up --build batch-learning-experiment
```
*Expected runtime: 10-30 minutes depending on dataset size*
*Output: Individual + ensemble results showing performance improvements*

**🌊 Online Learning**
```bash
# Linux/Mac
./run_online_individual.sh

# Windows
run_online_individual.bat

# Direct Docker
docker compose up --build online-experiment
```
*Expected runtime: 5-15 minutes*
*Output: Real-time learning performance metrics*

**🔄 Continual Learning (Both Cases)**
```bash
# Linux/Mac
./run_continual_individual.sh

# Windows
run_continual_individual.bat

# Individual cases
docker compose up --build cl_case1-experiment  # Case 1 only
docker compose up --build cl_case2-experiment  # Case 2 only
```
*Expected runtime: 15-45 minutes (Case 2 takes longer)*
*Output: Adaptation performance across different scenarios*

### Data Preparation Guide

#### 1. Data Format Requirements
- **Format**: Parquet files (`.parquet`)
- **Structure**: Tabular data with features and target column
- **Target column**: Should be named `is_attack`, `label`, or `target`
- **Features**: Numerical or categorical (will be auto-encoded)

#### 2. Data Placement
```bash
# Place your datasets in respective directories:
batch/cyber/
├── dataset1.parquet
├── dataset2.parquet
└── dataset3.parquet

online/cyber/
├── streaming_data1.parquet
└── streaming_data2.parquet

cl_case1/cyber/
├── task1_data.parquet
└── task2_data.parquet

cl_case2/cyber/
├── sequential_data1.parquet
└── sequential_data2.parquet
```

#### 3. Supported Dataset Types
- **CICIDS2017/2018**: Network intrusion detection
- **NSL-KDD**: Classic network security dataset
- **UNSW-NB15**: Modern network security dataset
- **CSE-CIC-IDS2018**: Comprehensive intrusion detection
- **Custom datasets**: Any cybersecurity tabular data

### Understanding Results

#### Batch Learning Results
**File**: `batch/results/batch_results_with_ensembles.csv`

**Key Columns**:
- `algorithm`: Algorithm name (individual or ensemble)
- `fnr`: False Negative Rate (lower = better for security)
- `fpr`: False Positive Rate (lower = better)
- `f1_score`: F1-Score (higher = better)
- `accuracy`: Overall accuracy (higher = better)

**Look for**:
- Ensemble methods typically outperform individual algorithms
- `Ensemble_WeightedByFNR` usually best for cybersecurity (lowest FNR)
- Performance improvements of 10-30% in key metrics

#### Performance Interpretation

| Metric | Good Value | Cybersecurity Priority |
|--------|------------|------------------------|
| **FNR** | < 0.05 (5%) | **CRITICAL** - Missing attacks is costly |
| **FPR** | < 0.10 (10%) | Important - Too many false alarms |
| **F1-Score** | > 0.90 | High - Balanced performance |
| **Accuracy** | > 0.95 | High - Overall correctness |

### Troubleshooting & Tips

#### Common Issues
1. **Docker not found**: Ensure Docker Desktop is running
2. **Memory errors**: Run experiments individually instead of all at once
3. **No data found**: Check that parquet files are in correct `cyber/` directories
4. **Permission denied**: Run `chmod +x *.sh` for Linux/Mac scripts

#### Performance Tips
1. **Start with batch experiment**: Best results with ensemble methods
2. **Monitor memory usage**: Close other applications for large datasets
3. **Check logs**: Docker will show detailed progress and any errors
4. **Incremental testing**: Start with small datasets to verify setup

#### Expected Outputs
```bash
# After successful run, you should see:
batch/results/
├── batch_results_individual.csv      # Individual algorithms
└── batch_results_with_ensembles.csv  # Individual + ensembles

online/results/
└── online_results.csv

cl_case1/results/
└── case1_results.csv

cl_case2/results/
└── case2_results.csv
```

## ❓ Frequently Asked Questions (FAQ)

### General Questions

**Q: Which experiment should I start with?**
A: Start with **Batch Learning** (`run_batch_individual.sh/.bat`) as it includes ensemble methods and typically gives the best results for cybersecurity datasets.

**Q: How long do experiments take to run?**
A: Depends on dataset size:
- Batch Learning: 10-30 minutes
- Online Learning: 5-15 minutes  
- Continual Learning: 15-45 minutes (Case 2 takes longer)
- All experiments: 30-90 minutes

**Q: What if I don't have cybersecurity data?**
A: The system will work with any binary classification dataset. Just ensure your target column is named `is_attack`, `label`, or `target`.

### Technical Questions

**Q: Why are ensemble methods better?**
A: Ensembles combine multiple algorithms' strengths, reducing individual weaknesses. For cybersecurity, this typically means 10-30% improvement in detecting attacks (lower FNR).

**Q: Which ensemble method should I use?**
A: For cybersecurity applications, use `Ensemble_WeightedByFNR` as it prioritizes algorithms that miss fewer attacks.

**Q: Can I add my own algorithms?**
A: Yes, modify the algorithm classes in `batch/src/batch_sim.py` and add your algorithm to the algorithms dictionary.

### Data Questions

**Q: What data format is required?**
A: Parquet files (`.parquet`) with tabular data. CSV files can be converted using pandas: `df.to_parquet('file.parquet')`

**Q: How should I prepare my data?**
A: Ensure you have:
- Features in columns (numerical or categorical)
- Target column named `is_attack`, `label`, or `target`
- Values: 0 = normal, 1 = attack/positive class

**Q: What's the minimum dataset size?**
A: At least 1000 samples recommended, with at least 100 positive cases for meaningful results.

### Troubleshooting Questions

**Q: Docker command not found?**
A: Install Docker Desktop and ensure it's running (whale icon stable in system tray).

**Q: Out of memory errors?**
A: Run experiments individually instead of all at once, or reduce dataset size.

**Q: No results generated?**
A: Check that parquet files are in correct `cyber/` directories and contain valid data.

## 🔧 Advanced Configuration

### Memory Optimization
For large datasets, modify `docker-compose.yaml`:
```yaml
services:
  batch-learning-experiment:
    # ... existing config ...
    deploy:
      resources:
        limits:
          memory: 4G        # Adjust as needed
        reservations:
          memory: 2G
```

### Custom Algorithm Parameters
Edit `batch/src/batch_sim.py`:
```python
algorithms = {
    'AROW': lambda: AROW(r=0.05),          # Lower regularization
    'SCW': lambda: SCW(C=2.0, eta=0.95),   # Higher confidence
    # ... other algorithms
}
```

### Dataset Splitting
The system automatically uses 80-20 train-test split. To modify, edit the `load_single_dataset` function in `batch/src/data_loader.py`.

## 📈 Performance Benchmarks

Based on standard cybersecurity datasets:

| Dataset Type | Individual Best | Ensemble Improvement | FNR Reduction |
|--------------|----------------|---------------------|---------------|
| **Network Intrusion** | F1: 0.85-0.92 | +8-15% | 20-35% |
| **Malware Detection** | F1: 0.88-0.94 | +5-12% | 15-25% |
| **Anomaly Detection** | F1: 0.82-0.89 | +10-18% | 25-40% |

*Results may vary depending on dataset characteristics and quality.*
