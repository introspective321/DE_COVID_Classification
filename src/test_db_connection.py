from sqlalchemy import create_engine
from utils.config import DATABASE_URL

def test_connection():
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            print("Connection to SQL Server successful.")
    except Exception as e:
        print(f"Failed to connect to SQL Server: {e}")

if __name__ == "__main__":
    test_connection()
