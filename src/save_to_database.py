from sqlalchemy import create_engine, text
from utils.config import DATABASE_URL

engine = create_engine(DATABASE_URL)

def save_to_database(df, table_name):
    """Save DataFrame to a specified table in the database."""
    engine = create_engine(DATABASE_URL)
    df.to_sql(table_name, con=engine, if_exists='append', index=False)
    print(f"Data saved to {table_name}.")


def save_with_versioning(df, table_name, primary_keys):
    """
    Save DataFrame to a specified table with versioning.
    :param df: Pandas DataFrame containing data.
    :param table_name: Name of the target database table.
    :param primary_keys: List of primary key columns for conflict detection.
    """
    with engine.connect() as conn:
        for _, row in df.iterrows():
            # Build primary key condition
            pk_conditions = " AND ".join([f"{pk}='{row[pk]}'" for pk in primary_keys])
            
            # Check if record exists
            check_query = f"SELECT version FROM {table_name} WHERE {pk_conditions}"
            existing = conn.execute(text(check_query)).fetchone()
            
            if existing:
                # Increment version and update record
                new_version = existing[0] + 1
                update_query = f"""
                UPDATE {table_name}
                SET {", ".join([f"{col}='{row[col]}'" for col in df.columns if col not in primary_keys])},
                    version={new_version}
                WHERE {pk_conditions}
                """
                conn.execute(text(update_query))
            else:
                # Insert new record with version 1
                insert_query = f"""
                INSERT INTO {table_name} ({", ".join(df.columns)}, version)
                VALUES ({", ".join([f"'{row[col]}'" if col in row else "NULL" for col in df.columns])}, 1)
                """
                conn.execute(text(insert_query))
    print(f"Data saved to {table_name} with versioning.")
