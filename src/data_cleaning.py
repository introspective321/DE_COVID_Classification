import pandas as pd

def fill_missing_values(df, strategy='mean', columns=None):
    """Fill missing values in specified columns using the given strategy."""
    for column in columns:
        if strategy == 'mean':
            df[column] = df[column].fillna(df[column].mean())
        elif strategy == 'median':
            df[column] = df[column].fillna(df[column].median())
        elif strategy == 'drop':
            df = df.dropna(subset=[column])
    return df

def format_date(df, date_column, date_format='%d-%m-%y'):
    """Format date column to a consistent format."""
    df[date_column] = pd.to_datetime(df[date_column], format=date_format)
    return df

def clean_data(df, date_column, columns_to_fill):
    """Perform data cleaning including filling missing values and formatting date."""
    df = fill_missing_values(df, columns=columns_to_fill)
    df = format_date(df, date_column)
    return df
