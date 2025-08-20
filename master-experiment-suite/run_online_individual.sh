#!/bin/bash

echo "======================================================"
echo "Running ONLINE LEARNING Experiment (Individual)"
echo "======================================================"
echo "This will run only the online learning experiment"
echo ""

# Run only the online experiment
docker compose up --build --exit-code-from online-experiment online-experiment

echo ""
echo "======================================================"
echo "ONLINE LEARNING Experiment Complete!"
echo "======================================================"
echo "Results saved to: ./online/results/online_results.csv"
echo "======================================================"
