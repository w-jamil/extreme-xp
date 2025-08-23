import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import os

def evaluate_on_chunk(algo, X_eval, y_eval):
    """Evaluates a model's accuracy on a chunk of data."""
    if len(X_eval) == 0: return 0.0
    y_preds = np.array([algo.predict(x) for x in X_eval])
    return accuracy_score(y_eval, y_preds)

def calculate_final_metrics(algo, X_eval, y_eval):
    """Calculates final Precision, Recall, and FNR for the malicious class (1)."""
    if len(X_eval) == 0: return 0.0, 0.0, 0.0
    
    y_preds = np.array([algo.predict(x) for x in X_eval])
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_eval, y_preds, labels=[-1, 1]).ravel()
    except ValueError:
        return np.nan, np.nan, np.nan

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return precision, recall, fnr, fpr

def plot_forgetting_over_time(forgetting_history_dict, task_names_list, output_path):
    """
    Plots the average forgetting for each algorithm over the training rounds
    and saves the figure to the specified path.

    Args:
        forgetting_history (dict): A dictionary containing forgetting DataFrames for each algo.
        task_names (list): A list of the task names for x-axis labeling.
        output_path (str): The file path to save the plot image (e.g., 'results/forgetting_plot.png').
    """
    print(f"\n--- Generating and saving forgetting plot to '{output_path}' ---")
    
    # Use a non-interactive backend, essential for running in Docker
    plt.switch_backend('Agg')
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))

    for algo_name, forgetting_df in forgetting_history_dict.items():
        forgetting_timeline = forgetting_df.max(axis=0).dropna()
        if not forgetting_timeline.empty:
            ax.plot(forgetting_timeline.index, forgetting_timeline.values, marker='o', linestyle='-', label=algo_name)

    # Add vertical lines to delineate major task type changes
    last_base_task = ""
    for i, name in enumerate(task_names_list):
        base_task = name.split('_')[0]
        if base_task != last_base_task:
            ax.axvline(x=i, color='gray', linestyle='--', alpha=0.7)
            last_base_task = base_task
            
    ax.set_title('Average Forgetting Caused by Each Training Round', fontsize=18, pad=20)
    ax.set_xlabel('Training Round (Task Trained)', fontsize=14)
    ax.set_ylabel('Average Forgetting Score (Higher is Worse)', fontsize=14)
    ax.set_xticks(range(len(task_names_list)))
    ax.set_xticklabels(task_names_list, rotation=45, ha='right')
    ax.legend(title='Algorithm', fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    fig.tight_layout()
    
    try:
        # Ensure the directory for the plot exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save the figure to the specified path
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig) # Close the figure to free up memory
        print("--> Plot saved successfully.")
    except Exception as e:
        print(f"ERROR: Could not save the plot. Reason: {e}")
# --- END OF THE FIX ---
