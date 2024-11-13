from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Date
from utils.config import DATABASE_URL

def setup_database():
    """Set up the database schema."""
    engine = create_engine(DATABASE_URL)
    metadata = MetaData()

    subjects = Table('subjects', metadata,
                     Column('subject_id', String, primary_key=True),
                     Column('gender', String),
                     Column('age', Integer),
                     Column('ethnicity', String))

    measurements = Table('measurements', metadata,
                         Column('subject_id', String),
                         Column('date', Date),
                         Column('t_cr_max', Float),
                         Column('t_cl_max', Float))

    metadata.create_all(engine)
    print("Database setup complete.")
