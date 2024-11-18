from sqlalchemy import create_engine, text
import time

# Environment variables for database URL
SERVER_URL = "mssql+pyodbc://sa:StrongP@ssw0rd123@sqlserver:1433/?driver=ODBC+Driver+17+for+SQL+Server"
DATABASE_URL = "mssql+pyodbc://sa:StrongP@ssw0rd123@sqlserver:1433/Covid?driver=ODBC+Driver+17+for+SQL+Server"

def create_database():
    """
    Connect to the SQL Server and create the 'Covid' database if it does not exist.
    """
    print("Connecting to SQL Server...")
    engine = create_engine(SERVER_URL)
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT name FROM sys.databases WHERE name = 'Covid'")
        ).fetchone()
        if not result:
            print("Database 'Covid' does not exist. Creating it...")
            connection.execute(text("CREATE DATABASE Covid"))
            print("Database 'Covid' created successfully.")
        else:
            print("Database 'Covid' already exists.")

def initialize_schema():
    """
    Connect to the 'Covid' database and initialize its schema.
    """
    print("Connecting to the 'Covid' database...")
    for attempt in range(5):  # Retry up to 5 times
        try:
            engine = create_engine(DATABASE_URL)
            with engine.connect() as connection:
                print("Connected to the 'Covid' database.")
                print("Initializing schema...")
                with open("database/schema.sql", "r") as schema_file:
                    schema_script = schema_file.read()
                connection.execute(text(schema_script))
                print("Schema initialized successfully.")
                break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(10)  # Wait for 10 seconds before retrying
    else:
        raise Exception("Failed to connect to the 'Covid' database after 5 attempts.")

def setup_database():
    """
    Orchestrates the database setup process by creating the database (if needed) and initializing the schema.
    """
    try:
        create_database()
        initialize_schema()
    except Exception as e:
        print(f"Error during database setup: {e}")

if __name__ == "__main__":
    setup_database()
