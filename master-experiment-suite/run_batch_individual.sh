#!/bin/bash

echo "======================================================"
echo "Running BATCH LEARNING Experiment (Individual)"
echo "======================================================"
echo "This will run only the batch learning experiment with ensemble aggregation"
echo ""

# Run only the batch learning experiment
docker compose up --build --exit-code-from batch-learning-experiment batch-learning-experiment

echo ""
echo "======================================================"
echo "BATCH LEARNING Experiment Complete!"
echo "======================================================"
echo "Results saved to: ./batch/results/"
echo "Files generated:"
echo "- batch_results_individual.csv (individual algorithms only)"
echo "- batch_results_with_ensembles.csv (individual + ensemble results)"
echo "======================================================"
