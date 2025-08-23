#!/bin/bash

echo "======================================================"
echo "EXTREME-XP: Batch Learning Experiments"
echo "======================================================"
echo "Choose your experiment type:"
echo ""
echo "1. OnlineToBatch Protocol (Recommended)"
echo "   - 7 algorithms with weighted majority voting"
echo "   - Focus on minimizing false negatives"
echo "   - Comprehensive ensemble analysis"
echo ""
echo "2. AROW Comparison"
echo "   - Plain AROW vs AROW + RF Ensemble"
echo "   - Data denoising and feature selection"
echo "   - FNR/FPR cybersecurity analysis"
echo ""

read -p "Enter your choice (1 or 2, default=1): " choice
choice=${choice:-1}

if [ "$choice" = "1" ]; then
    echo "======================================================"
    echo "Running OnlineToBatch Protocol Experiment"
    echo "======================================================"
    cd batch/src
    python batch_sim.py online-to-batch
    cd ../..
elif [ "$choice" = "2" ]; then
    echo "======================================================"
    echo "Running AROW Comparison Experiment"
    echo "======================================================"
    cd batch/src
    python batch_sim.py
    cd ../..
else
    echo "Invalid choice. Running default OnlineToBatch experiment..."
    cd batch/src
    python batch_sim.py online-to-batch
    cd ../..
fi

echo ""
echo "======================================================"
echo "Batch Learning Experiment Complete!"
echo "======================================================"
echo "Results saved to: ./batch/results/"
echo "Files generated:"
echo "- onlinetobatch_individual_results.csv"
echo "- onlinetobatch_ensemble_comparison.csv"
echo "======================================================"