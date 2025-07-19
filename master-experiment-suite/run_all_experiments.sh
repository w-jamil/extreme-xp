#!/bin/bash

# This script runs all Docker Compose experiments sequentially to avoid memory issues.
# It ensures that one experiment finishes before the next one starts.

echo "======================================================"
echo "Starting Experiment 1: Task-CL"
echo "======================================================"
docker compose up --build --exit-code-from task-cl-experiment task-cl-experiment

echo ""
echo "======================================================"
echo "Starting Experiment 2: Task-Domain-CL"
echo "======================================================"
docker compose up --build --exit-code-from task-domain-cl-experiment task-domain-cl-experiment

echo ""
echo "======================================================"
echo "Starting Experiment 3: Online (Correct Scaling)"
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