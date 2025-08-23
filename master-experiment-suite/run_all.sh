#!/bin/bash

# This script runs all experiments sequentially to avoid resource conflicts.
# It ensures that one experiment finishes before the next one starts.

echo "======================================================"
echo "EXTREME-XP: Master Experiment Suite - All Experiments"
echo "======================================================"

echo "======================================================"
echo "Starting Experiment 1: OnlineToBatch Protocol"
echo "======================================================"
cd batch/src
/home/wjamil/Documents/venv/bin/python batch_sim.py
cd ../..

echo ""
echo "======================================================"
echo "Starting Experiment 2: Continual Learning Case 1"
echo "======================================================"
if [ -d "cl_case1" ]; then
    echo "Continual Learning Case 1 - requires Docker setup"
    echo "Run: docker compose up --build --exit-code-from cl_case1-experiment cl_case1-experiment"
else
    echo "CL Case 1 directory not found or not configured for direct execution"
fi

echo ""
echo "======================================================"
echo "Starting Experiment 3: Continual Learning Case 2"
echo "======================================================"
if [ -d "cl_case2" ]; then
    echo "Continual Learning Case 2 - requires Docker setup"
    echo "Run: docker compose up --build --exit-code-from cl_case2-experiment cl_case2-experiment"
else
    echo "CL Case 2 directory not found or not configured for direct execution"
fi

echo ""
echo "======================================================"
echo "Starting Experiment 4: Online Learning"
echo "======================================================"
if [ -d "online" ]; then
    echo "Online Learning - requires Docker setup"
    echo "Run: docker compose up --build --exit-code-from online-experiment online-experiment"
else
    echo "Online directory not found or not configured for direct execution"
fi

echo ""
echo "======================================================"
echo "All available experiments complete."
echo "======================================================"
echo "Results locations:"
echo "- OnlineToBatch experiments: ./batch/results/"
echo "- Other experiments: Check respective directories"
echo "======================================================"