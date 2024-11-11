def check_missing_values(df, critical_columns):
    """Check for and handle missing values in critical columns."""
    for col in critical_columns:
        if df[col].isnull().any():
            print(f"Warning: Missing values found in {col}. Filling with mean.")
            df[col].fillna(df[col].mean(), inplace=True)
    return df

def check_out_of_range(df, column_ranges):
    """Filter out values outside specified ranges for critical columns."""
    for column, (min_val, max_val) in column_ranges.items():
        df = df[(df[column] >= min_val) & (df[column] <= max_val)]
    return df

def apply_reliability_checks(df, critical_columns, column_ranges):
    """Apply missing values and range checks on DataFrame."""
    df = check_missing_values(df, critical_columns)
    df = check_out_of_range(df, column_ranges)
    return df
