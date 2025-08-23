#!/usr/bin/env python3
"""
Quick test runner for OnlineToBatch experiment
"""

import sys
import os
sys.path.append('/home/wjamil/Documents/extreme-xp/master-experiment-suite/cl_case1/src')

from online_to_batch_experiment import OnlineToBatchExperiment

def main():
    print("Starting OnlineToBatch experiment...")
    
    # Create experiment with efficient settings
    experiment = OnlineToBatchExperiment(
        data_dir='/home/wjamil/Documents/extreme-xp/master-experiment-suite/cl_case1/cyber',
        results_dir='/home/wjamil/Documents/extreme-xp/master-experiment-suite/cl_case1/src/results',
        epochs=3,  # Small number for efficiency
        validation_size=0.15,
        test_size=0.25
    )
    
    experiment.run_experiment()

if __name__ == "__main__":
    main()
