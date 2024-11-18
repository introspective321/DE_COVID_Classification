# DE_COVID_Classification
Course Project
https://physionet.org/content/face-oral-temp-data/1.0.0/#files-panel

This repository contains the implementation of a data engineering pipeline for classifying individuals as healthy or unhealthy using thermal imaging data and survey responses. The project includes a full data pipeline, database management, Elasticsearch integration, and a Dockerized setup to ensure smooth deployment and scalability.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Features](#features)
3. [Folder Structure](#folder-structure)
4. [Technologies Used](#technologies-used)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Data Flow](#data-flow)
8. [Database Schema](#database-schema)
9. [Error Handling and Logging](#error-handling-and-logging)
10. [Future Enhancements](#future-enhancements)
11. [License](#license)

---

## **Overview**
This project processes and analyzes thermal imaging data and associated clinical survey responses to classify subjects as healthy or unhealthy. It supports:
- Data ingestion and cleaning
- Reliability checks
- Metadata extraction for videos
- Storing cleaned data in a relational database (SQL Server)
- Integration with Elasticsearch for indexing and search capabilities
- Machine learning model integration for classification
- Deployment via Docker containers

---

## **Features**
- **Modularized Pipeline**: Components for ingestion, cleaning, and database handling.
- **Database Management**: Relational data storage with SQL Server.
- **Elasticsearch Integration**: Efficient indexing and retrieval of data.
- **Containerization**: All services run in Docker containers for ease of deployment.
- **Machine Learning**: Supports model inference for classification.

---

## **Folder Structure**
```plaintext
├── database/
│   ├── setup.py            # Initializes database and schema
│   ├── schema.sql          # SQL script to define database schema
├── pipeline/
│   ├── pipeline.py         # Main entry point for the pipeline
│   ├── ingestion.py        # Handles data ingestion from raw sources
│   ├── data_cleaning.py    # Performs data cleaning and transformations
│   ├── reliability_checks.py  # Applies reliability checks to survey data
│   ├── save_to_database.py # Handles saving cleaned data to SQL Server
│   ├── elasticsearch_setup.py  # Integrates with Elasticsearch
│   ├── utils/
│       ├── config.py       # Configuration file for environment variables
│       ├── logger.py       # Custom logger for error handling
├── data/
│   ├── raw/                # Raw thermal videos and CSV files
│   ├── processed/          # Processed data files
├── Dockerfile              # Dockerfile for building the pipeline image
├── docker-compose.yml      # Docker Compose file for multi-container setup
├── requirements.txt        # Python dependencies
├── README.md               # Comprehensive documentation
```
---

## Technologies Used

- **Programming Language**: Python 3.9
- **Relational Database**: SQL Server 2019
- **Search Engine**: Elasticsearch 7.10.2
- **Containerization**: Docker and Docker Compose
- **Machine Learning**: Pretrained ML models for inference
- **Libraries**:
  - `pandas`, `sqlalchemy`, `pyodbc` for data handling
  - `elasticsearch-py` for Elasticsearch integration
  - `pytest` for unit testing

---