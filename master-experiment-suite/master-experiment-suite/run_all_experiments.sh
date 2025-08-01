#!/bin/bash

# This script runs all Docker Compose experiments sequentially to avoid memory issues.
# It ensures that one experiment finishes before the next one starts.

echo "======================================================"
echo "Starting Experiment 1: CL CASE 2"
echo "======================================================"
docker compose up --build --exit-code-from cl_case2-experiment cl_case2-experiment

echo ""
echo "======================================================"
echo "Starting Experiment 2: CL CASE 1"
echo "======================================================"
docker compose up --build --exit-code-from cl_case1-experiment cl_case1-experiment

echo ""
echo "======================================================"
echo "Starting Experiment 3: Online"
echo "======================================================"
docker compose up --build --exit-code-from online-experiment online-experiment

echo ""
echo "======================================================"
echo "Starting Experiment 4: Batch Learning"
echo "======================================================"
docker compose up --build --exit-code-from batch-learning-experiment batch-learning-experiment

echo ""
echo "======================================================"
echo "All experiments complete."
echo "Check the 'results' folder in each experiment directory for the output."
echo "======================================================"