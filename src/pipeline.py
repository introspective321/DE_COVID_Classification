from ingestion import load_csv
from data_cleaning import clean_data
from reliability_checks import apply_reliability_checks
from normalization import normalize_subject_data, normalize_measurement_data
from database_setup import setup_database
from save_to_database import save_to_database
from elasticsearch_setup import setup_elasticsearch
from utils.config import DATABASE_URL, ELASTICSEARCH_URL

def run_pipeline():
    # Step 1: Set up database and Elasticsearch
    setup_database()
    setup_elasticsearch(ELASTICSEARCH_URL)
    
    # Step 2: Load data
    file_path = "data/ICI_group1.csv"
    df = load_csv(file_path)
    
    # Step 3: Clean data
    df = clean_data(df)
    
    # Step 4: Apply reliability checks
    column_ranges = {'T_CRmax': (30, 40), 'T_CLmax': (30, 40)}
    df = apply_reliability_checks(df, column_ranges)
    
    # Step 5: Normalize data and save to SQL Server
    subject_df = normalize_subject_data(df)
    measurement_df = normalize_measurement_data(df)
    
    save_to_database(subject_df, "subjects")
    save_to_database(measurement_df, "measurements")
    
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
