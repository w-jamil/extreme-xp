import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import glob
import copy

# Import our custom modules
from data_preprocessor import DataPreprocessor
from algorithms import PassiveAggressive, Perceptron, GradientLearning,AROW,RDA,SCW,AdaRDA
from metrics import calculate_class1_metrics # We will use our own forgetting logic now
from data_handler import prepare_data_from_zenodo

class OnlineStreamSimulator:
    """
    Encapsulates a continual learning simulation based on the prequential
    evaluation methodology (test-then-train on a per-sample basis).
    Forgetting is calculated exactly as per the provided formula.
    """
    def __init__(self, file_path, window_size=250):
        """
        Initializes the simulator.

        Args:
            file_path (str): The path to the single .parquet data file.
            window_size (int): The size of the sliding window (W) for training.
        """
        self.file_path = file_path
        self.window_size = window_size
        self.X, self.y, self.models = None, None, {}
        # Stores the stream of predictions made at each step j
        self.prediction_history = {}
        # Stores the calculated forgetting score F_k(j) at each step j
        self.forgetting_history = {}
        
    def _setup_environment(self):
        """Private method to load data and initialize models."""
        preprocessor = DataPreprocessor(file_path=self.file_path)
        self.X, self.y = preprocessor.process()
        
        if self.X is None or len(self.X) <= self.window_size:
            print(f"  - Warning: Not enough data in '{os.path.basename(self.file_path)}' to run simulation. Skipping.")
            return False
        
        n_features = self.X.shape[1]
        
        # --- FIX: PROVIDE THE CORRECT PARAMETERS FOR EACH ALGORITHM ---
        self.models = {
            "PA": PassiveAggressive(n_features=n_features),
            "Perceptron": Perceptron(n_features=n_features),
            "GL": GradientLearning(n_features=n_features),
            
            # AROW expects 'r'
            "AROW": AROW(n_features=n_features, r=1.0),
            
            # RDA expects 'lambda_param' and 'gamma_param'
            "RDA": RDA(n_features=n_features, lambda_param=0.01, gamma_param=1.0),
            
            # SCW expects 'C' and 'eta'
            "SCW": SCW(n_features=n_features, C=0.1, eta=0.95),
            
            # AdaRDA expects 'lambda_param', 'eta_param', and 'delta_param'
            "AdaRDA": AdaRDA(n_features=n_features, lambda_param=0.01, eta_param=0.1, delta_param=1.0)
        }
        # --- END OF FIX ---
        
        self.prediction_history = {name: [] for name in self.models.keys()}
        self.forgetting_history = {name: [] for name in self.models.keys()}
        return True
    
    def _run_learning_loop(self):
        """
        Private method to run the main continual learning loop, calculating
        forgetting at each time step as per the provided formula.
        
        NOTE: This approach has a time complexity of approximately O(N^2) due to
        re-evaluating on all past samples at each step. It can be slow on very large datasets.
        """
        total_samples = len(self.X)
        
        # The main loop iterates from j = W to N-1
        for j in range(self.window_size, total_samples):
            if (j + 1) % 250 == 0:
                print(f"  - Processing step {j + 1}/{total_samples}...")

            # --- 1. Store Pre-Update Weights (θ_{j-1}) ---
            weights_before_update = {name: copy.deepcopy(model.weights) for name, model in self.models.items()}
            
            # --- 2. Train on the window D_j^train (from j-W to j-1) ---
            start_idx = j - self.window_size
            X_train_window, y_train_window = self.X[start_idx:j], self.y[start_idx:j]
            for model in self.models.values():
                for i in range(len(X_train_window)):
                    model.partial_fit(X_train_window[i], y_train_window[i])
            
            # Post-update weights are now θ_j

            # --- 3. Test on D_j^test (the single sample at j) ---
            X_test, y_test = self.X[j], self.y[j]
            for name, model in self.models.items():
                self.prediction_history[name].append(model.predict(X_test))

            # --- 4. Calculate Forgetting F_k(j) ---
            past_test_indices = range(self.window_size, j)
            
            # If there are no past tasks yet, forgetting is 0
            if not past_test_indices:
                for name in self.models.keys():
                    self.forgetting_history[name].append(0.0)
                continue

            for name, model in self.models.items():
                forgetting_events_for_model = []
                # Temporarily set weights to the "before" state to calculate R_k[i, j-1]
                original_weights = copy.deepcopy(model.weights)
                model.weights = weights_before_update[name]
                
                # Get predictions on all past data with old weights
                preds_before = np.array([model.predict(self.X[i]) for i in past_test_indices])
                
                # Restore the new weights to calculate R_k[i, j]
                model.weights = original_weights
                preds_after = np.array([model.predict(self.X[i]) for i in past_test_indices])
                
                # Get true labels for all past data
                y_past_true = self.y[past_test_indices]
                
                # Accuracy = 1 if correct, 0 if incorrect
                acc_before = (preds_before == y_past_true).astype(int)
                acc_after = (preds_after == y_past_true).astype(int)
                
                # Forgetting is the sum of max(0, R[i, j-1] - R[i, j])
                forgetting_sum = np.sum(np.maximum(0, acc_before - acc_after))
                
                # Average over the number of past tasks (j - W)
                avg_forgetting_at_j = forgetting_sum / len(past_test_indices)
                self.forgetting_history[name].append(avg_forgetting_at_j)


    def _calculate_final_results(self):
        """Calculates final metrics based on the simulation history."""
        final_results = []
        # The true labels for the prediction stream start from the W-th sample
        y_true_stream = self.y[self.window_size:]

        for name in self.models.keys():
            # Standard metrics on the entire prediction stream
            y_pred_stream = self.prediction_history[name]
            precision, tpr, fpr = calculate_class1_metrics(y_true_stream, y_pred_stream)
            
            # Overall forgetting is the average of the F_k(j) scores calculated at each step
            avg_forgetting = np.mean(self.forgetting_history[name]) if self.forgetting_history[name] else 0.0
            std_forgetting = np.std(self.forgetting_history[name]) if self.forgetting_history[name] else 0.0

            final_results.append({
                "Algorithm": name,
                "Precision": precision,
                "TPR": tpr,
                "FPR": fpr,
                "Forgetting": avg_forgetting,
                "Deviation": std_forgetting
            })
            
        return pd.DataFrame(final_results)

    def run(self):
        """
        Public method to run the entire simulation for one file and return results.
        """
        if self._setup_environment():
            self._run_learning_loop()
            return self._calculate_final_results()
        return None


if __name__ == "__main__":
    # --- BATCH EXPERIMENT CONFIGURATION ---
    ZENODO_ARCHIVE_URL = 'https://zenodo.org/api/records/13787591/files-archive'
    DATA_DIRECTORY = 'cyber/'  # This is the target folder for our data

    data_ready = prepare_data_from_zenodo(ZENODO_ARCHIVE_URL, DATA_DIRECTORY)



    if data_ready:
        
        OUTPUT_CSV_FILE = 'results/tacl_results.csv'
        WINDOW_SIZE = 250 # Corresponds to W in the formula

        # --- BATCH EXECUTION SCRIPT ---
        search_path = os.path.join(DATA_DIRECTORY, '*.parquet')
        all_data_files = sorted(glob.glob(search_path))

        if not all_data_files:
            print(f"FATAL: No .parquet files found in '{DATA_DIRECTORY}'. Please check the path.")
        else:
            print(f"Found {len(all_data_files)} datasets. Starting batch simulation...")
            
            all_results_list = []
            
            for i, file_path in enumerate(all_data_files):
                dataset_name = os.path.basename(file_path).replace('.parquet', '')
                print("\n" + "="*80)
                print(f"Running Experiment {i+1}/{len(all_data_files)} on Dataset: {dataset_name}")
                print("="*80)

                simulator = OnlineStreamSimulator(
                    file_path=file_path,
                    window_size=WINDOW_SIZE
                )
                
                results_df = simulator.run()
                
                if results_df is not None:
                    results_df.insert(0, 'Dataset', dataset_name)
                    all_results_list.append(results_df)

            if all_results_list:
                final_combined_df = pd.concat(all_results_list, ignore_index=True)
                
                cols_order = ['Dataset', 'Algorithm', 'Precision', 'TPR', 'FPR', 'Forgetting', 'Deviation']
                final_combined_df = final_combined_df[cols_order]

                print("\n" + "="*80)
                print("BATCH SIMULATION COMPLETE. FINAL COMBINED RESULTS:")
                print("="*80)
                print(final_combined_df.round(4).to_string())
                
                try:
                    final_combined_df.to_csv(OUTPUT_CSV_FILE, index=False, float_format='%.4f')
                    print(f"\nSUCCESS: All combined results saved to '{OUTPUT_CSV_FILE}'")
                except Exception as e:
                    print(f"\nERROR: Could not save combined results to CSV. Reason: {e}")
            else:
                print("\nNo simulations were completed successfully. No output file generated.")