-- Create subjects table
CREATE TABLE subjects (
    subject_id INT PRIMARY KEY IDENTITY(1,1),
    age INT NOT NULL,
    gender VARCHAR(10) NOT NULL,
    weight FLOAT,
    height FLOAT,
    exposure VARCHAR(50),
    created_at DATETIME DEFAULT GETDATE(),
    updated_at DATETIME DEFAULT GETDATE()
);

-- Create survey_data table
CREATE TABLE survey_data (
    survey_id INT PRIMARY KEY IDENTITY(1,1),
    subject_id INT NOT NULL,
    fever BOOLEAN,
    cough BOOLEAN,
    sore_throat BOOLEAN,
    diarrhea BOOLEAN,
    smell_loss BOOLEAN,
    taste_loss BOOLEAN,
    oxygen_level FLOAT,
    temperature FLOAT,
    heart_rate INT,
    blood_pressure VARCHAR(20),
    created_at DATETIME DEFAULT GETDATE(),
    updated_at DATETIME DEFAULT GETDATE(),
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);

-- Create videos table
CREATE TABLE videos (
    video_id INT PRIMARY KEY IDENTITY(1,1),
    subject_id INT NOT NULL,
    view ENUM('Front', 'Back', 'Left', 'Right'),
    file_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(255) NOT NULL,
    duration FLOAT,
    thermal_quality VARCHAR(255),
    created_at DATETIME DEFAULT GETDATE(),
    updated_at DATETIME DEFAULT GETDATE(),
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);

-- Create pcr_results table
CREATE TABLE pcr_results (
    pcr_id INT PRIMARY KEY IDENTITY(1,1),
    subject_id INT NOT NULL,
    result BOOLEAN,
    viral_load FLOAT,
    test_date DATETIME,
    created_at DATETIME DEFAULT GETDATE(),
    updated_at DATETIME DEFAULT GETDATE(),
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);
