import pandas as pd
from sqlalchemy import create_engine
from utils.config import DATABASE_URL

def load_csv(file_path):
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(file_path)

def save_to_database(df, table_name, if_exists='append'):
    """Save a DataFrame to the specified table in the database."""
    engine = create_engine(DATABASE_URL)
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)

def ingest_data(file_paths):
    """Ingest multiple CSV files and store them as raw tables in the database."""
    for table_name, file_path in file_paths.items():
        df = load_csv(file_path)
        save_to_database(df, table_name)
