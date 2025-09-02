#!/bin/bash

echo "======================================================"
echo "EXTREME-XP: Continual Learning Experiments"
echo "======================================================"
echo "Running Continual Learning Protocols:"
echo "- Sequential learning with catastrophic forgetting mitigation"
echo "- Case 1: Task-agnostic learning approach"
echo "- Case 2: Sliding window continual learning"
echo "- RBD24 cybersecurity datasets"
echo ""

echo "======================================================"
echo "Starting Continual Learning Case 1"
echo "======================================================"
if sudo docker-compose up --build --exit-code-from cl_case1-experiment cl_case1-experiment; then
    echo "✓ Continual Learning Case 1 completed successfully."
else
    echo "✗ Continual Learning Case 1 failed or was interrupted."
fi

echo ""
echo "======================================================"
echo "Starting Continual Learning Case 2"  
echo "======================================================"
if sudo docker-compose up --build --exit-code-from cl_case2-experiment cl_case2-experiment; then
    echo "✓ Continual Learning Case 2 completed successfully."
else
    echo "✗ Continual Learning Case 2 failed or was interrupted."
fi

echo ""
echo "======================================================"
echo "Continual Learning Experiments Complete!"
echo "======================================================"
echo "Results saved to:"
echo "- ./cl_case1/results/cl_case1_results.csv"
echo "- ./cl_case2/results/cl_case2_results.csv"
echo "======================================================"
