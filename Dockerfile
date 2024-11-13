# Start with a base Python image
FROM python:3.8-slim

# Set up the working directory
WORKDIR /app

# Copy the project files into the Docker container
COPY . /app

# Install system dependencies and pyodbc dependencies
RUN apt-get update && \
    apt-get install -y curl gnupg2 unixodbc unixodbc-dev && \
    # Add Microsoft's ODBC Driver for SQL Server repository
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql17 && \
    # Install Python dependencies
    apt-get install -y python3-pyodbc && \
    # Install Python packages listed in requirements.txt
    pip install -r requirements.txt && \
    # Clean up to reduce image size
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the command to run the pipeline
CMD ["python", "src/pipeline.py"]
