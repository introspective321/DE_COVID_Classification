def normalize_subject_data(df):
    """Extract and normalize subject data."""
    return df[['Subject_ID', 'Gender', 'Age', 'Ethnicity']].drop_duplicates()

def normalize_measurement_data(df):
    """Extract measurement data."""
    return df[['Subject_ID', 'Date', 'T_CRmax', 'T_CLmax']]
