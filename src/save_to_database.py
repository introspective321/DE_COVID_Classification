from sqlalchemy import create_engine
from utils.config import DATABASE_URL

def save_to_database(df, table_name):
    """Save DataFrame to a specified table in the database."""
    engine = create_engine(DATABASE_URL)
    df.to_sql(table_name, con=engine, if_exists='append', index=False)
    print(f"Data saved to {table_name}.")
