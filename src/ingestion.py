import pandas as pd

def load_csv(file_path):
    """Load data from a CSV file into a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise
