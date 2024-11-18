from database.setup import setup_database  # Direct import for setup_database
from .ingestion import load_csv, extract_video_metadata
from .data_cleaning import clean_data
from .reliability_checks import apply_reliability_checks
from .save_to_database import save_to_db

def run_pipeline():
    print("Starting pipeline...")

    # Step 1: Initialize the database schema
    setup_database()

    # Additional steps...
    survey_data = load_csv("data/raw/subject_description.csv")
    video_metadata = extract_video_metadata("data/raw/thermal_mpg_data")

    cleaned_survey = clean_data(survey_data)
    cleaned_video_metadata = clean_data(video_metadata)

    reliable_survey = apply_reliability_checks(cleaned_survey)

    save_to_db(reliable_survey, "subjects")
    save_to_db(cleaned_video_metadata, "videos")

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
