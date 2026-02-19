"""
Visualize all simulated datasets to show their non-linearities.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, '/home/wjamil/Documents/classification_experiments/online/src')

from simulation_kernel_vs_linear import (
    generate_linear_data,
    generate_overlapping_gaussians,
    generate_donut,
    generate_concentric_circles,
    generate_quadratic_boundary,
    generate_high_curvature,
    generate_moons,
    generate_sine_wave,
    generate_spirals,
    generate_swiss_roll,
    generate_xor,
    ensure_reproducibility
)

# Ensure reproducibility
ensure_reproducibility(seed=42)

# Dataset configurations
datasets = [
    ('Linear_5D', generate_linear_data, {'n_features': 5, 'noise': 0.1}, 'Linear'),
    ('Overlapping_Gaussians', generate_overlapping_gaussians, {'separation': 1.5, 'noise': 0.2}, 'Non-Linear-Moderate'),
    ('Donut', generate_donut, {'noise': 0.1}, 'Non-Linear-Moderate'),
    ('Circles', generate_concentric_circles, {'noise': 0.1}, 'Non-Linear-High'),
    ('Quadratic', generate_quadratic_boundary, {'noise': 0.1}, 'Non-Linear-High'),
    ('HighCurvature', generate_high_curvature, {'noise': 0.1}, 'Non-Linear-High'),
    ('Moons', generate_moons, {'noise': 0.1}, 'Non-Linear-VeryHigh'),
    ('SineWave', generate_sine_wave, {'noise': 0.1}, 'Non-Linear-VeryHigh'),
    ('Spirals', generate_spirals, {'noise': 0.1, 'n_turns': 2}, 'Non-Linear-VeryHigh'),
    ('SwissRoll', generate_swiss_roll, {'noise': 0.1}, 'Non-Linear-VeryHigh'),
    ('XOR', generate_xor, {'noise': 0.1}, 'Non-Linear-VeryHigh'),
]

# Create figure with subplots (4 rows x 3 cols = 12 subplots, we use 11)
fig, axes = plt.subplots(4, 3, figsize=(16, 14))
axes = axes.flatten()

for idx, (name, gen_func, kwargs, data_type) in enumerate(datasets):
    ax = axes[idx]
    
    # Generate data
    X, y = gen_func(n_samples=2000, **kwargs)
    
    # For high-dimensional data (>2D), project to 2D using PCA
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        X = PCA(n_components=2).fit_transform(X)
    
    # Plot
    pos_mask = y == 1
    neg_mask = y == -1
    
    ax.scatter(X[pos_mask, 0], X[pos_mask, 1], c='blue', alpha=0.6, s=20, label='Class +1')
    ax.scatter(X[neg_mask, 0], X[neg_mask, 1], c='red', alpha=0.6, s=20, label='Class -1')
    
    ax.set_title(f'{name}\n({data_type})', fontsize=10, fontweight='bold')
    ax.set_xlabel('Feature 1', fontsize=8)
    ax.set_ylabel('Feature 2', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    if idx == 0:
        ax.legend(loc='upper right', fontsize=8)

# Hide the last unused subplot
axes[-1].set_visible(False)

plt.tight_layout()
import os
output_dir = '/home/wjamil/Documents/classification_experiments/online/results'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'datasets_visualization.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved to: {output_file}")
