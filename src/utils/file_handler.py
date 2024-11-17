import os
import pandas as pd

def read_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def save_csv(dataframe, output_path):
    dataframe.to_csv(output_path, index=False)
    print(f"File saved to {output_path}")
