import time
from sqlalchemy import create_engine

DATABASE_URL = "mssql+pyodbc://sa:StrongP@ssw0rd123@sqlserver:1433/Covid?driver=ODBC+Driver+17+for+SQL+Server"

def save_to_db(data, table_name):
    """
    Saves a DataFrame to the database. Retries if connection fails.
    """
    for attempt in range(5):  # Retry 5 times
        try:
            print(f"Attempting to connect to database (Attempt {attempt + 1})...")
            engine = create_engine(DATABASE_URL)
            data.to_sql(table_name, con=engine, if_exists="append", index=False)
            print(f"Data saved successfully to {table_name}.")
            break
        except Exception as e:
            print(f"Error saving to database: {e}")
            if attempt < 4:  # Wait before retrying
                time.sleep(10)
            else:
                raise Exception("Failed to connect to the database after 5 attempts.")
