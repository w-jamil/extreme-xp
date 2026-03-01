import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import random
import os

# ============================================================================
# REPRODUCIBILITY: Set all random seeds
# ============================================================================
def ensure_reproducibility(seed=42):
    """Set all random seeds for complete reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.default_rng(seed)
    print(f"✓ Random seed set to {seed} for reproducibility")

# --- Configuration: All simulation parameters are here ---
N_SAMPLES = 5000 
N_FEATURES = 5 
NOISE_LEVEL = 0.1
DELAYS_TO_TEST = [0, 5, 10, 20, 50] 
PACK_SIZES_TO_TEST = [1, 5, 10, 20, 50] 

# --- Data Generation ---
def generate_data(n_samples, n_features, noise):
    """Generates a dataset with some noise."""
    print(f"Generating a dataset with {n_samples} samples and {n_features} features...")
    np.random.seed(42) # for reproducibility
    w_true = np.random.randn(n_features)
    X = np.random.rand(n_samples, n_features) * 2 - 1
    y_true = X @ w_true + np.random.randn(n_samples) * noise
    print("Data generation complete.")
    return X, y_true

# --- Algorithm Implementation ---
class CIRR:

    def __init__(self, n_features, total_samples):
        self.n_features = n_features
        # State variables from the R function
        self.b = np.zeros(n_features)
        self.A = np.zeros((n_features, n_features))
        self.w = np.ones(n_features)
        # Regularization 'a' is defined based on total samples, as in the R code
        self.a = 1.0 / total_samples if total_samples > 0 else 1.0
        self.name = "CIRR"

    def predict(self, x):
        """Makes a prediction based on the current state."""
        # 1. Create D matrix based on current weights `w`
        D_diag_sqrt = np.sqrt(np.abs(self.w))
        D_outer = np.outer(D_diag_sqrt, D_diag_sqrt) # This is `D` in R

        # 2. Update covariance matrix for prediction calculation (following R code structure)
        temp_A = self.A + np.outer(x, x)
        
        # 3. Calculate inverse and update weights for prediction
        try:
            # CRITICAL CORRECTION: Use * for element-wise multiplication to match R's D * At
            InvA = np.linalg.inv(np.diag([self.a] * self.n_features) + D_outer * temp_A) 
            # CRITICAL CORRECTION: Use * for element-wise multiplication to match R's D * InvA
            AAt = D_outer * InvA
            prediction_weights = AAt.T @ self.b # A.T @ b is equivalent to crossprod(A, b) for vectors/matrices
        except np.linalg.LinAlgError:
            # Fallback for singular matrix
            prediction_weights = self.w 
            
        return np.dot(prediction_weights, x)

    def update(self, x, y):
        """Updates the internal state using the true label."""
        # 1. Update covariance matrix `A` and target vector `b`
        self.A += np.outer(x, x)
        self.b += y * x

        # 2. Re-calculate D based on the *previous* step's weights `self.w`
        D_diag_sqrt = np.sqrt(np.abs(self.w))
        D_outer = np.outer(D_diag_sqrt, D_diag_sqrt) # This is `D` in R

        # 3. Re-calculate the final weights for the *next* time step
        try:
            # CRITICAL CORRECTION: Use * for element-wise multiplication to match R's D * self.A
            InvA = np.linalg.inv(np.diag([self.a] * self.n_features) + D_outer * self.A) 
            # CRITICAL CORRECTION: Use * for element-wise multiplication to match R's D * InvA
            AAt = D_outer * InvA
            self.w = AAt.T @ self.b # A.T @ b is equivalent to crossprod(A, b) for vectors/matrices
        except np.linalg.LinAlgError:
            # Suppress warning as it's common for this type of inverse calculation with certain data
            pass # Keep old weights

# --- Simulation Runner for Delayed Feedback ---
def run_simulation_with_delay(algorithm_class, X_data, y_data, delay, **kwargs):
    """
    Runs a full simulation, handling delayed feedback.
    """
    n_samples, n_features = X_data.shape
    algorithm = algorithm_class(n_features=n_features, total_samples=n_samples, **kwargs)
    
    predictions = np.zeros(n_samples)
    feedback_buffer = collections.deque()

    print(f"  Running simulation for {algorithm.name} with delay = {delay}...")

    current_losses = [] # To calculate cumulative loss step-by-step
    for t in range(n_samples):
        x_t, y_t = X_data[t], y_data[t]
        
        # 1. Make a prediction using the current state
        prediction_t = algorithm.predict(x_t)
        predictions[t] = prediction_t

        # Calculate current step's loss
        step_loss = (prediction_t - y_t)**2
        current_losses.append(step_loss)
        
        # 2. Add the data to the buffer to await feedback
        feedback_buffer.append((x_t, y_t))
        
        # 3. If the buffer is full, receive delayed feedback and update the model
        if len(feedback_buffer) > delay:
            x_delayed, y_delayed = feedback_buffer.popleft()
            algorithm.update(x_delayed, y_delayed)
            
    # Calculate final performance metrics
    rmse = np.sqrt(mean_squared_error(y_data, predictions))
    r2 = r2_score(y_data, predictions)
    cumulative_loss = np.cumsum(current_losses)
    # Replace zeros or very small numbers with a tiny positive value for log plotting (if applicable)
    cumulative_loss[cumulative_loss <= 0] = np.finfo(float).eps 
    
    return {
        'predictions': predictions,
        'cumulative_loss': cumulative_loss,
        'performance': {'RMSE': rmse, 'R2': r2}
    }

# --- Simulation Runner for Variable Delays (per-sample delay) ---
def run_simulation_with_variable_delay(algorithm_class, X_data, y_data, delay_sequence, **kwargs):
    """
    Runs simulation where each sample has its own delay specified in delay_sequence.
    delay_sequence[t] = d_t means sample t's feedback arrives at time t + d_t.
    
    This enables testing Corollary (concurrent-staleness):
    - Case 1: d_1 = D_T, d_t = 0 for t > 1 (single large delay, then recovery)
    - Case 2: d_t = d for all t (constant small delay, perpetual staleness)
    """
    n_samples, n_features = X_data.shape
    algorithm = algorithm_class(n_features=n_features, total_samples=n_samples, **kwargs)
    
    predictions = np.zeros(n_samples)
    # pending_feedback[t] = (x_t, y_t, arrival_time) - when feedback for sample t arrives
    pending_feedback = []
    
    total_delay = sum(delay_sequence)
    print(f"  Running simulation with variable delays (total D_T = {total_delay})...")

    current_losses = []
    for t in range(n_samples):
        x_t, y_t = X_data[t], y_data[t]
        
        # 1. Process any feedback that has arrived by time t
        still_pending = []
        for (x_s, y_s, arrival_time) in pending_feedback:
            if arrival_time <= t:
                algorithm.update(x_s, y_s)
            else:
                still_pending.append((x_s, y_s, arrival_time))
        pending_feedback = still_pending
        
        # 2. Make a prediction using the current state
        prediction_t = algorithm.predict(x_t)
        predictions[t] = prediction_t
        
        # Calculate current step's loss
        step_loss = (prediction_t - y_t)**2
        current_losses.append(step_loss)
        
        # 3. Schedule this sample's feedback for future arrival
        d_t = delay_sequence[t] if t < len(delay_sequence) else 0
        arrival_time = t + d_t + 1  # +1 because feedback arrives AFTER prediction
        pending_feedback.append((x_t, y_t, arrival_time))
    
    # Process any remaining feedback after the main loop
    for (x_s, y_s, arrival_time) in pending_feedback:
        if arrival_time <= n_samples:
            algorithm.update(x_s, y_s)
            
    # Calculate final performance metrics
    rmse = np.sqrt(mean_squared_error(y_data, predictions))
    r2 = r2_score(y_data, predictions)
    cumulative_loss = np.cumsum(current_losses)
    cumulative_loss[cumulative_loss <= 0] = np.finfo(float).eps 
    
    return {
        'predictions': predictions,
        'cumulative_loss': cumulative_loss,
        'performance': {'RMSE': rmse, 'R2': r2},
        'total_delay': total_delay
    }

# --- Simulation Runner for Pack/Batch Updates ---
def run_simulation_with_packs(algorithm_class, X_data, y_data, pack_size, **kwargs):
    """
    Runs a simulation where data arrives in packs (batches).
    The model predicts for the whole pack, then updates with the pack's data.
    """
    n_samples, n_features = X_data.shape
    algorithm = algorithm_class(n_features=n_features, total_samples=n_samples, **kwargs)
    
    predictions = np.zeros(n_samples)
    
    print(f"  Running simulation for {algorithm.name} with pack_size = {pack_size}...")
    
    # Iterate through the data in chunks of size `pack_size`
    for t in range(0, n_samples, pack_size):
        # Define the current pack
        X_pack = X_data[t : t + pack_size]
        y_pack = y_data[t : t + pack_size]
        
        # 1. Make predictions for the entire pack using the *current* model state
        for i in range(len(X_pack)):
            pred_idx = t + i
            predictions[pred_idx] = algorithm.predict(X_pack[i])
            
        # 2. After all predictions for the pack are made, update the model
        #    with the data from the pack
        for i in range(len(X_pack)):
            algorithm.update(X_pack[i], y_pack[i])
            
    # Calculate final performance metrics
    rmse = np.sqrt(mean_squared_error(y_data, predictions))
    r2 = r2_score(y_data, predictions)
    
    cumulative_loss = np.cumsum((predictions - y_data)**2)
    # Replace zeros or very small numbers with a tiny positive value for log plotting (if applicable)
    cumulative_loss[cumulative_loss <= 0] = np.finfo(float).eps 
    
    return {
        'predictions': predictions,
        'cumulative_loss': cumulative_loss,
        'performance': {'RMSE': rmse, 'R2': r2}
    }

# --- Main Execution and Plotting ---
if __name__ == '__main__':
    ensure_reproducibility(seed=42)
    # Generate consistent data for all runs
    X_data, y_data = generate_data(N_SAMPLES, N_FEATURES, NOISE_LEVEL)
    
    # --- DELAYED FEEDBACK SIMULATION ---
    delay_results = {}
    print("\n--- Starting Continuous Delayed Feedback Simulations ---")
    for delay in DELAYS_TO_TEST:
        delay_results[delay] = run_simulation_with_delay(CIRR, X_data, y_data, delay=delay)

    # --- PACKS (BATCH) SIMULATION ---
    pack_results = {}
    print("\n--- Starting Pack (Batch) Simulations ---")
    for k in PACK_SIZES_TO_TEST:
        pack_results[k] = run_simulation_with_packs(CIRR, X_data, y_data, pack_size=k)

    # --- Corollary Test: Concentrated vs Perpetual Delays (SAME D_T) ---
    # This properly tests Corollary (concurrent-staleness) with IDENTICAL total delay
    # Case 1: Concentrated delays (few large, recovers) vs Case 2: Perpetual delay (never recovers)
    print("\n--- Starting Corollary Test: Concentrated vs Perpetual Delays (Same D_T) ---")
    
    # Case 2: Perpetual delay of 10 for ALL samples
    # D_T = 10 * N_SAMPLES = 10 * 5000 = 50,000
    PERPETUAL_DELAY = 10
    D_T = PERPETUAL_DELAY * N_SAMPLES  # 50,000
    
    # Case 1: Concentrated delays to match D_T = 50,000
    # Use 500 samples with delay=100 each → 500 * 100 = 50,000
    NUM_CONCENTRATED = 500
    DELAY_PER_SAMPLE = D_T // NUM_CONCENTRATED  # 100
    
    # Case 1: Concentrated delays at the start, then immediate feedback
    # d_1 = d_2 = ... = d_500 = 100, d_t = 0 for t > 500
    # Algorithm suffers initial burst but RECOVERS at t ≈ 600
    delay_seq_case1 = [DELAY_PER_SAMPLE] * NUM_CONCENTRATED + [0] * (N_SAMPLES - NUM_CONCENTRATED)
    case1_result = run_simulation_with_variable_delay(CIRR, X_data, y_data, delay_seq_case1)
    
    # Case 2: Perpetual delay of 10 for ALL samples
    # Algorithm NEVER recovers - always 10 steps behind
    print(f"  Running simulation for {CIRR.__name__} with perpetual delay = {PERPETUAL_DELAY}...")
    case2_result = run_simulation_with_delay(CIRR, X_data, y_data, delay=PERPETUAL_DELAY)

    # --- Plot the results (three separate plots with white background) ---
    plt.style.use('default')  # White background
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['legend.title_fontsize'] = 9
    
    # Define distinct color maps for clarity
    delay_colors = plt.cm.tab10(np.arange(len(DELAYS_TO_TEST))) 
    pack_colors = plt.cm.tab10(np.arange(len(PACK_SIZES_TO_TEST)))
    line_styles = ['-', '--', ':', '-.', '-'] 

    # --- Create combined subplot figure (3 rows, 1 column) ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    start_t = 2
    
    # Subplot 1: Delayed Feedback
    ax1 = axes[0]
    for i, (delay, result_data) in enumerate(delay_results.items()):
        ax1.plot(range(start_t, N_SAMPLES), result_data['cumulative_loss'][start_t:], 
                 label=f'd={delay}', 
                 color=delay_colors[i], linestyle=line_styles[i % len(line_styles)], linewidth=1.5)
    ax1.set_title('(a) Effect of Delayed Feedback', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Cumulative Square Loss')
    ax1.legend(title='Delay', loc='upper left', framealpha=0.9, edgecolor='gray', ncol=5)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Subplot 2: Pack Size
    ax2 = axes[1]
    for i, (k, result_data) in enumerate(pack_results.items()):
        ax2.plot(range(start_t, N_SAMPLES), result_data['cumulative_loss'][start_t:], 
                 label=f'K={k}', 
                 color=pack_colors[i], linestyle=line_styles[i % len(line_styles)], linewidth=1.5)
    ax2.set_title('(b) Effect of Pack Size', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Cumulative Square Loss')
    ax2.legend(title='Pack', loc='upper left', framealpha=0.9, edgecolor='gray', ncol=5)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Subplot 3: Corollary Test
    ax3 = axes[2]
    start_t3 = NUM_CONCENTRATED + DELAY_PER_SAMPLE
    ax3.plot(range(start_t3, N_SAMPLES), case1_result['cumulative_loss'][start_t3:], 
             label=f'Concentrated ({NUM_CONCENTRATED}×{DELAY_PER_SAMPLE})', 
             color='darkgreen', linestyle='-', linewidth=2)
    ax3.plot(range(start_t3, N_SAMPLES), case2_result['cumulative_loss'][start_t3:], 
             label=f'Perpetual (d={PERPETUAL_DELAY})', 
             color='crimson', linestyle='--', linewidth=2)
    ax3.set_title(f'(c) Corollary Test: Same Total Delay $D_T$ = {D_T:,}', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Cumulative Square Loss')
    ax3.legend(title='Delay Structure', loc='upper left', framealpha=0.9, edgecolor='gray')
    ax3.grid(True, linestyle='--', alpha=0.4)
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    
    plt.savefig('plot_combined.png', dpi=200, bbox_inches='tight', facecolor='white')
    print("Combined plot saved to plot_combined.png")
    plt.close(fig)

    # --- Also save individual plots ---
    # Plot 1: Effect of Delayed Feedback on Cumulative Square Loss (start from t=2)
    start_t = 2
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for i, (delay, result_data) in enumerate(delay_results.items()):
        ax1.plot(range(start_t, N_SAMPLES), result_data['cumulative_loss'][start_t:], 
                 label=f'Delay = {delay} steps', 
                 color=delay_colors[i], linestyle=line_styles[i % len(line_styles)], linewidth=2)
    ax1.set_title('Effect of Delayed Feedback on Cumulative Square Loss', fontsize=16)
    ax1.set_xlabel('Time Steps', fontsize=12)
    ax1.set_ylabel('Cumulative Square Loss', fontsize=12) 
    ax1.legend(title='Feedback Delay', fontsize=11, loc='upper left', framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig('plot1_delayed_feedback.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Plot 1 saved to plot1_delayed_feedback.png")
    plt.close(fig1)

    # Plot 2: Effect of Pack Size on Cumulative Square Loss (start from t=2)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for i, (k, result_data) in enumerate(pack_results.items()):
        ax2.plot(range(start_t, N_SAMPLES), result_data['cumulative_loss'][start_t:], 
                 label=f'Pack Size = {k}', 
                 color=pack_colors[i], linestyle=line_styles[i % len(line_styles)], linewidth=2)
    ax2.set_title('Effect of Pack Size on Cumulative Square Loss', fontsize=16)
    ax2.set_xlabel('Time Steps', fontsize=12)
    ax2.set_ylabel('Cumulative Square Loss', fontsize=12) 
    ax2.legend(title='Pack Size (K)', fontsize=11, loc='upper left', framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig('plot2_pack_size.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Plot 2 saved to plot2_pack_size.png")
    plt.close(fig2)

    # Plot 3: Corollary Test - Concentrated vs Perpetual Delays
    start_t3 = NUM_CONCENTRATED + DELAY_PER_SAMPLE  # After concentrated case recovers
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(range(start_t3, N_SAMPLES), case1_result['cumulative_loss'][start_t3:], 
             label=f'Concentrated ({NUM_CONCENTRATED}×{DELAY_PER_SAMPLE})', 
             color='darkgreen', linestyle='-', linewidth=2)
    ax3.plot(range(start_t3, N_SAMPLES), case2_result['cumulative_loss'][start_t3:], 
             label=f'Perpetual (d={PERPETUAL_DELAY})', 
             color='crimson', linestyle='--', linewidth=2)
    ax3.set_title(f'Corollary Test: Same Total Delay $D_T = {D_T:,}$', fontsize=16)
    ax3.set_xlabel('Time Steps', fontsize=12)
    ax3.set_ylabel('Cumulative Square Loss', fontsize=12)
    ax3.legend(title='Delay Structure', fontsize=11, loc='upper left', framealpha=0.9)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig('plot3_corollary_test.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Plot 3 saved to plot3_corollary_test.png")
    plt.close(fig3)

    # --- Print summary tables ---
    print("\n--- Delayed Feedback Simulation Summary ---")
    delay_summary_data = []
    for delay, result_data in delay_results.items():
        delay_summary_data.append({
            'Delay': delay,
            'Final Total Loss': result_data['cumulative_loss'][-1],
            'RMSE': result_data['performance']['RMSE'],
            'R2': result_data['performance']['R2']
        })
    delay_summary_df = pd.DataFrame(delay_summary_data).set_index('Delay')
    print(delay_summary_df.to_string(float_format="%.4e")) # Use scientific notation

    pack_summary_data = []
    for k, result_data in pack_results.items():
        pack_summary_data.append({
            'Pack Size': k,
            'Final Total Loss': result_data['cumulative_loss'][-1],
            'RMSE': result_data['performance']['RMSE'],
            'R2': result_data['performance']['R2']
        })
    pack_summary_df = pd.DataFrame(pack_summary_data).set_index('Pack Size')
    print(pack_summary_df.to_string(float_format="%.4e")) # Use scientific notation
    
    print(f"\n--- Corollary Test Summary (Same D_T = {D_T:,}) ---")
    corollary_summary_data = [
        {'Scenario': f'Case 1: Concentrated ({NUM_CONCENTRATED} x {DELAY_PER_SAMPLE})',
         'Total D_T': case1_result['total_delay'],
         'Final Loss': case1_result['cumulative_loss'][-1],
         'RMSE': case1_result['performance']['RMSE'],
         'R2': case1_result['performance']['R2']},
        {'Scenario': f'Case 2: Perpetual (d_t={PERPETUAL_DELAY} for all t)',
         'Total D_T': PERPETUAL_DELAY * N_SAMPLES,
         'Final Loss': case2_result['cumulative_loss'][-1],
         'RMSE': case2_result['performance']['RMSE'],
         'R2': case2_result['performance']['R2']}
    ]
    corollary_summary_df = pd.DataFrame(corollary_summary_data).set_index('Scenario')
    print(corollary_summary_df.to_string(float_format="%.4e"))
    
    print("\n=== Corollary Interpretation ===")
    print(f"Both cases have IDENTICAL total delay D_T = {D_T:,}")
    print(f"Case 1: {NUM_CONCENTRATED} samples with delay={DELAY_PER_SAMPLE}, recovers at t≈{NUM_CONCENTRATED + DELAY_PER_SAMPLE}")
    print(f"Case 2: Perpetual delay={PERPETUAL_DELAY} for all {N_SAMPLES} samples, NEVER recovers")
    if case1_result['cumulative_loss'][-1] < case2_result['cumulative_loss'][-1]:
        print("→ Case 1 (concentrated, recovers) outperforms Case 2 (perpetual)")
        print("  Recovery after cold-start is beneficial for this problem")
    else:
        print("→ Case 2 (perpetual) outperforms Case 1 (concentrated)")
        print("  Incremental learning outweighs cost of permanent staleness")
    print("\nNote: Corollary proves both cases have same D_T but different dynamics.")
    print("Which performs better depends on the algorithm and problem structure.")