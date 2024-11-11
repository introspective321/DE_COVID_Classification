from ingestion import ingest_data
from reliability_checks import apply_reliability_checks
from data_cleaning import clean_data
from normalization import normalize_subject_data, normalize_temperature_data, normalize_environment_data
from database_setup import setup_database
from elasticsearch_setup import setup_elasticsearch

# File paths
file_paths = {
    "ici_group1": "data/ICI_group1.csv",
    "ici_group2": "data/ICI_group2.csv",
    # Add other file paths...
}

# Critical columns and range thresholds for reliability checks
critical_columns = ['T_CRmax', 'T_CLmax']  # Add more as needed
column_ranges = {
    'T_CRmax': (30.0, 40.0),
    'T_CLmax': (30.0, 40.0),
    # Define other ranges...
}

def run_pipeline():
    """Run the complete pipeline end-to-end."""
    # Step 1: Database and Elasticsearch setup
    setup_database()
    setup_elasticsearch()
    
    # Step 2: Ingest data
    ingest_data(file_paths)
    
    # Step 3: Data Cleaning and Reliability Checks
    for table_name in file_paths.keys():
        df = ingest_data(file_paths[table_name])
        df = clean_data(df, date_column='Date', columns_to_fill=critical_columns)
        df = apply_reliability_checks(df, critical_columns, column_ranges)
    
        # Step 4: Normalize Data
        subject_df = normalize_subject_data(df)
        temperature_df = normalize_temperature_data(df)
        environment_df = normalize_environment_data(df)
    
        # Step 5: Save Normalized Data to Database
        save_to_database(subject_df, "subjects")
        save_to_database(temperature_df, "temperatures")
        save_to_database(environment_df, "environment")

if __name__ == "__main__":
    run_pipeline()
