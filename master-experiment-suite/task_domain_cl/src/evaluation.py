import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

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
    
    return precision, recall, fnr

def plot_forgetting_over_time(forgetting_history_dict, task_names_list):
    """Plots the average forgetting caused by training on each task."""
    print("\n--- Generating Forgetting Over Time Plot ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))

    for algo_name, forgetting_df in forgetting_history_dict.items():
        # Get the max forgetting value for each round (column)
        forgetting_timeline = forgetting_df.max(axis=0).dropna()
        ax.plot(forgetting_timeline.index, forgetting_timeline.values, marker='o', linestyle='-', label=algo_name)

    # Add vertical lines to delineate major task type changes
    last_base_task = ""
    for i, name in enumerate(task_names_list):
        base_task = name.split('_')[0]
        if base_task != last_base_task:
            ax.axvline(x=i, color='gray', linestyle='--', alpha=0.7)
            last_base_task = base_task
            
    ax.set_title('Average Forgetting Caused by Each Training Round', fontsize=16)
    ax.set_xlabel('Training Round (Task Trained)', fontsize=12)
    ax.set_ylabel('Average Forgetting Score (Higher is Worse)', fontsize=12)
    ax.set_xticks(range(len(task_names_list)))
    ax.set_xticklabels(task_names_list, rotation=45, ha='right')
    ax.legend(title='Algorithm', fontsize=11)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    plt.show()