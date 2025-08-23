#!/bin/bash

echo "======================================================"
echo "Running CONTINUAL LEARNING Experiments (Both Cases)"
echo "======================================================"
echo "This will run both continual learning experiments sequentially"
echo ""

echo "======================================================"
echo "Starting Continual Learning Case 1"
echo "======================================================"
docker compose up --build --exit-code-from cl_case1-experiment cl_case1-experiment

echo ""
echo "======================================================"
echo "Starting Continual Learning Case 2"
echo "======================================================"
docker compose up --build --exit-code-from cl_case2-experiment cl_case2-experiment

echo ""
echo "======================================================"
echo "CONTINUAL LEARNING Experiments Complete!"
echo "======================================================"
echo "Results saved to:"
echo "- ./cl_case1/results/case1_results.csv"
echo "- ./cl_case2/results/case2_results.csv"
echo "======================================================"
