from sklearn.metrics import confusion_matrix
import numpy as np

def calculate_forgetting_metrics(performance_snapshots, num_chunks):
    """
    Calculates the average and standard deviation of forgetting.
    Forgetting for a past chunk 'j' at current time 'k' is defined as the
    drop in accuracy from its peak on chunk 'j'.

    Args:
        performance_snapshots (dict): A dictionary storing accuracy on past chunks
                                      at each evaluation point.
        num_chunks (int): The total number of evaluation chunks.

    Returns:
        dict: A dictionary mapping model names to their avg and std of forgetting.
    """
    forgetting_results = {}

    for model_name, snapshots in performance_snapshots.items():
        forgetting_scores = []
        # We can only measure forgetting after the first chunk
        if num_chunks < 2:
            continue

        # Iterate over each time step k (after the first one)
        for k in range(1, num_chunks):
            # Iterate over each past chunk j
            for j in range(k):
                # Accuracy on chunk j right after it was learned
                accuracy_at_peak = snapshots[j][j]
                
                # Accuracy on chunk j at the current time k
                accuracy_now = snapshots[k][j]
                
                # Forgetting is the drop from peak performance
                forgetting = accuracy_at_peak - accuracy_now
                if forgetting > 0: # Only count actual drops in performance
                    forgetting_scores.append(forgetting)
        
        if forgetting_scores:
            avg_forgetting = np.mean(forgetting_scores)
            std_forgetting = np.std(forgetting_scores)
        else:
            # If no forgetting occurred
            avg_forgetting = 0.0
            std_forgetting = 0.0
            
        forgetting_results[model_name] = {
            "Avg. Forgetting": avg_forgetting,
            "Std. Forgetting": std_forgetting
        }
        
    return forgetting_results

def calculate_class1_metrics(y_true, y_pred):
    """
    Calculates Precision, TPR, and FPR for the positive class (1).
    Handles cases where only one class is present in predictions.
    
    Args:
        y_true (list or np.ndarray): Ground truth labels.
        y_pred (list or np.ndarray): Predicted labels.

    Returns:
        tuple: A tuple containing (precision, tpr, fpr).
    """
    try:
        # labels=[-1, 1] ensures the confusion matrix order is consistent
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[-1, 1]).ravel()
    except ValueError: 
        # This happens if only one class is present in y_true or y_pred
        return np.nan, np.nan, np.nan

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return precision, recall, fnr, fpr

def calculate_final_metrics(algo, X_eval, y_eval):
    """Calculates final Precision, Recall, and FNR for class 1."""
    if len(X_eval) == 0:
        return 0.0, 0.0, 0.0
    
    y_preds = np.array([algo.predict(x) for x in X_eval])
    
    try:
        # labels=[-1, 1] ensures the confusion matrix order is consistent
        tn, fp, fn, tp = confusion_matrix(y_eval, y_preds, labels=[-1, 1]).ravel()
    except ValueError: # Happens if only one class is present
        return np.nan, np.nan, np.nan

    # Precision for class 1 = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # Recall for class 1 (TPR) = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # False Negative Rate for class 1 = FN / (FN + TP)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return precision, recall, fnr
