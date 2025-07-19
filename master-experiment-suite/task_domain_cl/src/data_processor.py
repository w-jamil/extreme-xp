import pandas as pd
import os
import glob

class TaskGenerator:
    """
    Loads multiple .parquet files from a directory, processing each file as a
    distinct continual learning task.
    """
    def __init__(self, directory_path):
        """
        Initializes the processor with the path to the directory containing task files.
        
        Args:
            directory_path (str): The path to the directory (e.g., 'cyber/').
        """
        self.directory_path = directory_path

    def generate_tasks(self):
        """
        Scans the directory for .parquet files, loads each one, processes it,
        and returns a dictionary of tasks.

        Returns:
            dict: A dictionary where keys are task names (from filenames) and 
                  values are tuples of (X, y) data.
        """
        print(f"Scanning for task files in directory: '{self.directory_path}'")
        
        # Find all files ending with .parquet in the specified directory
        search_path = os.path.join(self.directory_path, '*.parquet')
        parquet_files = glob.glob(search_path)

        if not parquet_files:
            print(f"ERROR: No .parquet files found in '{self.directory_path}'.")
            print("Please ensure your data files are in the correct location.")
            return {}

        tasks_raw = {}
        print(f"Found {len(parquet_files)} task files. Processing each one...")
        
        for file_path in sorted(parquet_files): # Sorting for consistent order
            # Create the task name from the filename (e.g., "Crypto_Desktop")
            task_name = os.path.basename(file_path).replace('.parquet', '')
            
            try:
                task_df = pd.read_parquet(file_path)
            except Exception as e:
                print(f"  - Warning: Could not read file '{file_path}'. Error: {e}. Skipping.")
                continue

            # --- Process this specific task's data ---
            task_df = task_df.sort_values(['timestamp']).reset_index(drop=True)
            
            # The 'groupby' and 'sum' logic from your original script
            if 'user_id' in task_df.columns:
                 task_df = task_df.drop(columns=['user_id'], axis=1)
            x_task = task_df.groupby(['timestamp']).sum()
            
            # Map labels: 0 is normal (-1), all others are malicious (1)
            y_task = x_task["label"].map({0: -1, 1: 1, 2: 1, 3: 1, 4: 1}).fillna(-1)
            
            # Create feature matrix X by dropping the label
            x_task = x_task.drop(columns=['label'], axis=1)
            
            tasks_raw[task_name] = (x_task, y_task)
            print(f"  - Generated task: '{task_name}' with {len(x_task)} samples.")

        return tasks_raw