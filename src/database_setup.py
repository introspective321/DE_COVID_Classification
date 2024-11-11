from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float
from utils.config import DATABASE_URL

def setup_database():
    """Create database schema."""
    engine = create_engine(DATABASE_URL)
    metadata = MetaData()

    subjects = Table('subjects', metadata,
                     Column('subject_id', String, primary_key=True),
                     Column('gender', String),
                     Column('age_range', String),
                     Column('ethnicity', String),
                     Column('cosmetics', Integer))

    temperatures = Table('temperatures', metadata,
                         Column('subject_id', String),
                         Column('round_number', Integer),
                         Column('t_cr_max', Float),
                         Column('t_cl_max', Float),
                         # Add other temperature columns here...
                         )

    environment = Table('environment', metadata,
                        Column('subject_id', String),
                        Column('date', String),
                        Column('time', String),
                        Column('ambient_temp', Float),
                        Column('humidity', Float),
                        Column('distance', Float))

    metadata.create_all(engine)
    print("Database setup complete.")
