def apply_reliability_checks(data):
    # Standardize column names
    data.columns = data.columns.str.strip()  # Remove leading/trailing spaces
    print("Columns in data:", data.columns)  # Debugging
    
    # Apply reliability check
    if "Body Temperature (°C)" in data.columns:
        data = data[(data["Body Temperature (°C)"] >= 35.0) & (data["Body Temperature (°C)"] <= 42.0)]
    else:
        raise KeyError("'Body Temperature (°C)' column is missing in the dataset.")
    return data
