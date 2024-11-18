FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies, including ODBC drivers
RUN apt-get update && apt-get install -y \
    curl \
    unixodbc \
    unixodbc-dev \
    apt-transport-https \
    gnupg \
    && curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update && ACCEPT_EULA=Y apt-get install -y \
    msodbcsql17 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


    RUN apt-get update && apt-get install -y \
    unixodbc \
    unixodbc-dev \
    curl \
    gnupg \
    apt-transport-https \
    && curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY pipeline/ ./pipeline/
COPY database/ ./database/
COPY data/ ./data/

# Add /app to PYTHONPATH
ENV PYTHONPATH=/app

# Set entrypoint
CMD ["python", "-m", "pipeline.pipeline"]
