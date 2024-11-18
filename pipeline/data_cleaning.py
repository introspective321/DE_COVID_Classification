def clean_data(data):
    """Clean survey or video metadata."""
    try:
        # Example cleaning: fill missing values with defaults
        data = data.fillna({
            "age": 0,
            "gender": "Unknown",
            "temperature": 36.5,
            "oxygen_level": 98.0
        })
        print("Cleaned data successfully.")
        return data
    except Exception as e:
        print(f"Error cleaning data: {e}")
        raise
