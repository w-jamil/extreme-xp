#!/bin/bash

# This script runs all experiments sequentially to avoid resource conflicts.
# It ensures that one experiment finishes before the next one starts.

echo "======================================================"
echo "EXTREME-XP: Master Experiment Suite - All Experiments"
echo "======================================================"
echo "This suite will run:"
echo "1. OnlineToBatch Learning (RBD24 + MNIST)"
echo "2. Continual Learning Case 1 (Task-agnostic)"
echo "3. Continual Learning Case 2 (Sliding window)"
echo "4. Online Learning (RBD24 + Kaggle)"
echo ""

# Check if Docker is available
if ! command -v docker-compose &> /dev/null; then
    echo "⚠️  docker-compose is not installed or not in PATH."
    echo "Only OnlineToBatch experiments will run (they don't require Docker)."
    echo ""
fi

echo "======================================================"
echo "Starting Experiment 1: OnlineToBatch Learning"
echo "======================================================"
cd batch/src

echo "1.1: Running RBD24 Cybersecurity experiments..."
/home/wjamil/Documents/venv/bin/python batch_sim.py

echo ""
echo "1.2: Running MNIST Binary Classification experiment..."
/home/wjamil/Documents/venv/bin/python mnist_sim.py

cd ../..

echo ""
echo "======================================================"
echo "Starting Experiment 2: Continual Learning Case 1"
echo "======================================================"
if [ -d "cl_case1" ] && command -v docker-compose &> /dev/null; then
    echo "Running Continual Learning Case 1 (Task-agnostic approach)..."
    if sudo docker-compose up --build --exit-code-from cl_case1-experiment cl_case1-experiment; then
        echo "✓ Continual Learning Case 1 completed successfully."
    else
        echo "✗ Continual Learning Case 1 failed or was interrupted."
    fi
else
    echo "⚠️  CL Case 1 directory not found or docker-compose not available"
fi

echo ""
echo "======================================================"
echo "Starting Experiment 3: Continual Learning Case 2"
echo "======================================================"
if [ -d "cl_case2" ] && command -v docker-compose &> /dev/null; then
    echo "Running Continual Learning Case 2 (Sliding window approach)..."
    if sudo docker-compose up --build --exit-code-from cl_case2-experiment cl_case2-experiment; then
        echo "✓ Continual Learning Case 2 completed successfully."
    else
        echo "✗ Continual Learning Case 2 failed or was interrupted."
    fi
else
    echo "⚠️  CL Case 2 directory not found or docker-compose not available"
fi

echo ""
echo "======================================================"
echo "Starting Experiment 4: Online Learning"
echo "======================================================"
if [ -d "online" ] && command -v docker-compose &> /dev/null; then
    echo "Running Online Learning experiment (Streaming approach)..."
    if sudo docker-compose up --build --exit-code-from online-experiment online-experiment; then
        echo "✓ Online Learning experiment completed successfully."
    else
        echo "✗ Online Learning experiment failed or was interrupted."
    fi
else
    echo "⚠️  Online directory not found or docker-compose not available"
fi

echo ""
echo "======================================================"
echo "All Available Experiments Complete!"
echo "======================================================"
echo "Results locations:"
echo "- OnlineToBatch experiments: ./batch/results/"
echo "  * RBD24 cybersecurity: onlinetobatch_individual_results.csv"
echo "  * MNIST: mnist_binary_even_odd.csv"
echo "- Continual Learning Case 1: ./cl_case1/results/cl_case1_results.csv"
echo "- Continual Learning Case 2: ./cl_case2/results/cl_case2_results.csv"
echo "- Online Learning: ./online/results/online_results.csv"
echo "  * Includes RBD24 + Kaggle credit card fraud datasets"
echo "======================================================"