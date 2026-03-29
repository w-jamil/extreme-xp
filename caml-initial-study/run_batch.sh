#!/bin/bash

echo "======================================================"
echo "EXTREME-XP: OnlineToBatch Learning Experiments"
echo "======================================================"
echo "Running OnlineToBatch Protocol:"
echo "- 7 algorithms with weighted majority voting"
echo "- Focus on minimizing false negatives"
echo "- Comprehensive ensemble analysis"
echo "- RBD24 cybersecurity datasets + MNIST binary classification"
echo ""

echo "======================================================"
echo "Experiment 1: RBD24 Cybersecurity Datasets"
echo "======================================================"
cd batch/src
echo "Running batch simulation with RBD24 cybersecurity datasets..."
/home/wjamil/Documents/venv/bin/python batch_sim.py

echo ""
echo "======================================================"
echo "Experiment 2: MNIST Binary Classification"
echo "======================================================"
echo "Running MNIST even/odd digit classification experiment..."
/home/wjamil/Documents/venv/bin/python mnist_sim.py
cd ../..

echo ""
echo "======================================================"
echo "OnlineToBatch Learning Experiments Complete!"
echo "======================================================"
echo "Results saved to: ./batch/results/"
echo "Files generated:"
echo "- RBD24 Cybersecurity: onlinetobatch_individual_results.csv"
echo "- MNIST: mnist_binary_even_odd.csv"
echo "======================================================"