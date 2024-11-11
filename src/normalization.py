def normalize_subject_data(df):
    """Extract and structure subject data."""
    subject_df = df[['SubjectID', 'Gender', 'Age', 'Ethnicity', 'Cosmetics']].drop_duplicates()
    return subject_df.rename(columns={'SubjectID': 'subject_id', 'Gender': 'gender', 
                                      'Age': 'age_range', 'Ethnicity': 'ethnicity', 
                                      'Cosmetics': 'cosmetics'})

def normalize_temperature_data(df):
    """Extract and structure temperature data."""
    temperature_columns = [col for col in df.columns if 'T_' in col]
    temp_df = df[['SubjectID'] + temperature_columns]
    return temp_df.rename(columns={'SubjectID': 'subject_id'})

def normalize_environment_data(df):
    """Extract and structure environment data."""
    environment_df = df[['SubjectID', 'Date', 'Time', 'T_atm', 'Humidity', 'Distance']].drop_duplicates()
    return environment_df.rename(columns={'SubjectID': 'subject_id', 'Date': 'date', 
                                          'Time': 'time', 'T_atm': 'ambient_temp', 
                                          'Humidity': 'humidity', 'Distance': 'distance'})
