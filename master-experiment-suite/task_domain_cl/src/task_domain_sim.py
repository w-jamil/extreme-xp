import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Import our custom modules
from data_processor import TaskGenerator
from algorithms import PassiveAggressive, Perceptron, GradientLearning
from evaluation import evaluate_on_chunk, calculate_final_metrics, plot_forgetting_over_time
from data_handler import prepare_data_from_zenodo


class ContinualLearningSimulator:
    """
    A class to encapsulate the entire continual learning simulation, making it
    configurable and easy to run.
    """
    def __init__(self, data_directory_path, output_csv_path, output_plot_path, training_chunk_size=500, eval_chunk_size=200):
        """
        Initializes the simulator with configuration parameters.
        (Corrected to accept output_plot_path)
        """
        self.data_directory_path = data_directory_path
        self.output_csv_path = output_csv_path
        self.output_plot_path = output_plot_path 
        self.training_chunk_size = training_chunk_size
        self.eval_chunk_size = eval_chunk_size
        
        self.tasks_processed = {}
        self.task_names = []
        self.algorithms = {}
        self.performance_history = {}
        self.forgetting_history = {}

    def _setup_environment(self):
        """Private method to load data, preprocess it, and initialize models."""
        processor = TaskGenerator(directory_path=self.data_directory_path)
        tasks_raw = processor.generate_tasks()

        if not tasks_raw:
            print("ERROR: No tasks were generated. Halting simulation.")
            return False

        print("\nScaling and preparing data streams...")
        for name, (X, y) in tasks_raw.items():
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.tasks_processed[name] = {"X": X_scaled, "y": y}
        
        self.task_names = sorted(list(self.tasks_processed.keys()))

        print("\nInitializing persistent algorithms...")
        n_features = list(self.tasks_processed.values())[0]["X"].shape[1]
        self.algorithms = {
            "PassiveAggressive": PassiveAggressive(n_features=n_features),
            "Perceptron": Perceptron(n_features=n_features),
            "GradientLearning": GradientLearning(n_features=n_features)
        }
        
        num_rounds = len(self.task_names)
        self.performance_history = {name: pd.DataFrame(np.nan, index=self.task_names, columns=range(num_rounds)) for name in self.algorithms}
        self.forgetting_history = {name: pd.DataFrame(np.nan, index=self.task_names, columns=range(num_rounds)) for name in self.algorithms}
        
        return True

    def _run_learning_loop(self):
        """Private method to run the main continual learning simulation loop."""
        num_rounds = len(self.task_names)
        for round_num in range(num_rounds):
            task_to_train_on = self.task_names[round_num]
            print(f"\nROUND {round_num + 1}/{num_rounds}: TRAINING ON TASK -> {task_to_train_on}")

            acc_before_training = {name: {} for name in self.algorithms}
            for algo_name, algo in self.algorithms.items():
                for eval_task_name, eval_task_data in self.tasks_processed.items():
                    if len(eval_task_data["X"]) < self.eval_chunk_size:
                        acc_before_training[algo_name][eval_task_name] = np.nan
                        continue
                    X_eval, y_eval = eval_task_data["X"][:self.eval_chunk_size], eval_task_data["y"].iloc[:self.eval_chunk_size]
                    acc_before_training[algo_name][eval_task_name] = evaluate_on_chunk(algo, X_eval, y_eval)

            train_task_data = self.tasks_processed[task_to_train_on]
            X_train_full, y_train_full = train_task_data["X"], train_task_data["y"]
            start_idx = (round_num * self.training_chunk_size) % len(X_train_full)
            end_idx = start_idx + self.training_chunk_size

            if end_idx > len(X_train_full):
                X_chunk1, y_chunk1 = X_train_full[start_idx:], y_train_full.iloc[start_idx:]
                remaining = end_idx - len(X_train_full)
                X_chunk2, y_chunk2 = X_train_full[:remaining], y_train_full.iloc[:remaining]
                X_train, y_train = np.vstack([X_chunk1, X_chunk2]), pd.concat([y_chunk1, y_chunk2])
            else:
                X_train, y_train = X_train_full[start_idx:end_idx], y_train_full.iloc[start_idx:end_idx]
            
            for j in range(len(X_train)):
                for algo in self.algorithms.values():
                    algo.partial_fit(X_train[j], y_train.iloc[j])
            
            for algo_name, algo in self.algorithms.items():
                other_task_forgetting = []
                for eval_task_name, eval_task_data in self.tasks_processed.items():
                    if len(eval_task_data["X"]) < self.eval_chunk_size: continue
                    X_eval, y_eval = eval_task_data["X"][:self.eval_chunk_size], eval_task_data["y"].iloc[:self.eval_chunk_size]
                    acc_after = evaluate_on_chunk(algo, X_eval, y_eval)
                    self.performance_history[algo_name].loc[eval_task_name, round_num] = acc_after
                    if eval_task_name != task_to_train_on:
                        acc_before = acc_before_training[algo_name][eval_task_name]
                        if not np.isnan(acc_before):
                            forgetting = max(0.0, acc_before - acc_after)
                            other_task_forgetting.append(forgetting)
                avg_forgetting = np.mean(other_task_forgetting) if other_task_forgetting else 0.0
                self.forgetting_history[algo_name].loc[task_to_train_on, round_num] = avg_forgetting

    def _report_results(self):
        """
        Private method to calculate, print, and save the final results and plots.
        """
        print("\n" + "="*70)
        print("CONTINUAL LEARNING EXPERIMENT SUMMARY")
        print("="*70)

        # --- 1. Calculate Final Performance Metrics (Precision, Recall, FNR) ---
        final_metrics_summary = {'Algorithm': [], 'Task': [], 'Precision': [], 'Recall': [], 'FNR': []}
        print("Calculating final performance metrics on all tasks...")
        for task_name, task_data in self.tasks_processed.items():
            X_full, y_full = task_data["X"], task_data["y"]
            for algo_name, algo in self.algorithms.items():
                precision, recall, fnr = calculate_final_metrics(algo, X_full, y_full)
                final_metrics_summary['Algorithm'].append(algo_name)
                final_metrics_summary['Task'].append(task_name)
                final_metrics_summary['Precision'].append(precision)
                final_metrics_summary['Recall'].append(recall)
                final_metrics_summary['FNR'].append(fnr)

        final_summary_df = pd.DataFrame(final_metrics_summary)
        
        # --- 2. Calculate Overall Forgetting and Add to DataFrame ---
        overall_forgetting_scores = {}
        for algo_name in self.algorithms:
            overall_avg_forget = self.forgetting_history[algo_name].mean().mean()
            overall_forgetting_scores[algo_name] = overall_avg_forget

        # Add the forgetting score as a new column to the summary DataFrame
        final_summary_df['Overall Avg Forgetting'] = final_summary_df['Algorithm'].map(overall_forgetting_scores)

        # --- 3. Print Console Reports ---
        for algo_name in self.algorithms:
            print(f"\n--- Final Metrics for Algorithm: {algo_name} ---")
            # Filter the main DataFrame for this algorithm's results
            algo_df_for_print = final_summary_df[
                final_summary_df['Algorithm'] == algo_name
            ].drop(columns=['Algorithm', 'Overall Avg Forgetting'])
            print(algo_df_for_print.round(4).to_string(index=False))

        print("\n" + "-"*70)
        print("FORGETTING ANALYSIS")
        print("-"*70)
        for algo_name, score in overall_forgetting_scores.items():
            print(f"  - {algo_name:<12}: {score:.4f}")

        # --- 4. Save Combined Results to CSV ---
        try:
            final_summary_df.to_csv(self.output_csv_path, index=False, float_format='%.4f')
            print(f"\nSUCCESS: All results have been saved to '{self.output_csv_path}'")
        except Exception as e:
            print(f"\nERROR: Could not save results to CSV. Reason: {e}")

        # --- 5. Generate Plot ---
        plot_forgetting_over_time(self.forgetting_history, self.task_names, self.output_plot_path)

    def run(self):
        """Public method to run the entire simulation from setup to reporting."""
        if self._setup_environment():
            self._run_learning_loop()
            self._report_results()


if __name__ == "__main__":
    # --- CONFIGURE AND RUN THE SIMULATION ---
    
    ZENODO_ARCHIVE_URL = 'https://zenodo.org/api/records/13787591/files-archive'
    DATA_DIRECTORY = 'cyber/'  # This is the target folder for our data

    data_ready = prepare_data_from_zenodo(ZENODO_ARCHIVE_URL, DATA_DIRECTORY)



    if data_ready:
    
    # Define the name for the output CSV file
        OUTPUT_CSV_FILE = 'results/task_domain_results.csv'
        OUTPUT_PLOT_FILE = 'results/task_domain_forgetting_plot.png'

        # Instantiate the simulator with all necessary paths
        simulator = ContinualLearningSimulator(
            data_directory_path=DATA_DIRECTORY,
            output_csv_path=OUTPUT_CSV_FILE,
            output_plot_path=OUTPUT_PLOT_FILE, 
            training_chunk_size=500,
            eval_chunk_size=200
        )

        # Run the full experiment
        simulator.run()