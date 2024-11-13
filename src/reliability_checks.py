def apply_reliability_checks(df, column_ranges):
    """Filter data based on specified column ranges."""
    for column, (min_val, max_val) in column_ranges.items():
        if column in df.columns:
            df = df[(df[column] >= min_val) & (df[column] <= max_val)]
    print("Reliability checks applied.")
    return df
