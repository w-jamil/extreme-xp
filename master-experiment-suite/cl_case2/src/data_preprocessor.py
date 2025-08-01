import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

class DataPreprocessor:
    """
    A class to load, preprocess, and scale the crypto dataset.
    """
    def __init__(self, file_path):
        """
        Initializes the preprocessor with the path to the data file.
        
        Args:
            file_path (str): The path to the .parquet data file.
        """
        self.file_path = file_path

    def process(self):
        """
        Loads the data, performs cleaning and transformation, and returns
        the final feature matrix (X) and target vector (y).

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The scaled feature matrix (X).
                - np.ndarray: The target vector (y).
        """
        print("Loading and preprocessing data...")
        # Load and sort data
        crypto = pd.read_parquet(self.file_path)
        crypto = crypto.sort_values(['timestamp']).reset_index(drop=True)
        
        # Drop user_id and group by timestamp
        crypto = crypto.drop(columns=['user_id'], axis=1)
        x_crypto = crypto.groupby(['timestamp']).sum()

        # Create target variable y (-1 for normal, 1 for attack)
        y_crypto = x_crypto["label"].map({0: -1, 1: 1, 2: 1, 3: 1, 4: 1,5: 1,6: 1,7: 1,8:1,9:1,10:1,11:1,12:1,13:1,14:1,15:1,16:1,17:1,18:1,19:1,20:1}).fillna(-1)
        
        # Create feature matrix X
        x_crypto = x_crypto.drop(columns=['label'], axis=1)
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(x_crypto)
        
        print(f"Data processed. Shape of X: {X_scaled.shape}, Shape of y: {y_crypto.shape}")
        
        return X_scaled, np.array(y_crypto)