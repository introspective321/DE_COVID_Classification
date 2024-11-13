import pandas as pd

def clean_data(df):
    """Apply data cleaning steps to DataFrame."""
    # Fill missing values for critical columns with median or specific values
    critical_columns = ['T_CRmax', 'T_CLmax']
    for col in critical_columns:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    # Format date columns to ensure consistency
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    print("Data cleaning applied.")
    return df
