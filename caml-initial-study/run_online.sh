#!/bin/bash

echo "======================================================"
echo "EXTREME-XP: Online Learning Experiments"
echo "======================================================"
echo "Running Online Learning Protocol:"
echo "- 7 algorithms in streaming fashion"
echo "- RBD24 cybersecurity datasets + Kaggle credit fraud"
echo "- Prequential evaluation methodology"
echo ""

# Clean up any existing containers first
echo "Cleaning up existing containers..."
sudo docker-compose down --remove-orphans

# Build the image
echo "Building online experiment image..."
sudo docker-compose build online-experiment

# Run only the online experiment using docker-compose run (avoids container recreation issues)
echo "Starting online learning experiment..."
sudo docker-compose run --rm online-experiment python online.py

echo ""
echo "======================================================"
echo "Online Learning Experiments Complete!"
echo "======================================================"
echo "Results saved to: ./online/results/online_results.csv"
echo "======================================================"
