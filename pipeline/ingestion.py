import pandas as pd
import os

def load_csv(file_path):
    """Load survey data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded CSV data from {file_path}.")
        return data
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise

def extract_video_metadata(base_dir):
    """Extract metadata from video files."""
    metadata = []
    for subject_id in os.listdir(base_dir):
        subject_path = os.path.join(base_dir, subject_id)
        if os.path.isdir(subject_path):
            for view in os.listdir(subject_path):
                view_path = os.path.join(subject_path, view)
                if os.path.isdir(view_path):
                    for video_file in os.listdir(view_path):
                        if video_file.endswith(('.mp4', '.avi')):
                            file_path = os.path.join(view_path, video_file)
                            metadata.append({
                                "subject_id": subject_id,
                                "view": view,
                                "file_name": video_file,
                                "file_path": file_path
                            })
    print(f"Extracted metadata for {len(metadata)} video files.")
    return pd.DataFrame(metadata)
