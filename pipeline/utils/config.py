import os

DATABASE_URL = os.getenv("DATABASE_URL", "mssql+pyodbc://sa:StrongP@ssw0rd123@localhost:1433/Covid?driver=ODBC+Driver+17+for+SQL+Server")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200")
