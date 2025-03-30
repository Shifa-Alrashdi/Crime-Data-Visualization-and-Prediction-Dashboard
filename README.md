# Crime Analytics Dashboard

An interactive dashboard for law enforcement that combines geospatial crime visualization with AI-powered report analysis.
You can access this app using the link: https://crime-data-visualization-and-prediction-dashboard.streamlit.app/

## Features

- **Crime Mapping**: Interactive Folium maps with heatmaps and clustered markers
- **Report Analysis**: PDF processing and machine learning classification
- **Risk Assessment**: Severity prediction for tactical response planning
- **Real-time Filtering**: Filter crime data by category and location

### File Details

- **Competition_Dataset.csv**: - Crime dataset used to Pre-trained ML model.

- **Data Analysis & Modeling.ipynb**: including all exploratory data analysis, preprocessing steps, model training, evaluation, and any visualizations (Level 1 and 2).

- **crime_classifier_model.joblib**: - Pre-trained ML model generated from Data Analysis & Modeling.ipynb code.

- **static/ (Supporting Files)** : For uploaded PDF storage (created automatically if missing)

- **requirements.txt**: Include all labraries requierd.

- **Dockerfile**: packages the app, model, data, and dependencies into a deployable container image.

- **app.py**:	Main Streamlit application.

- **Deployment_scripts.txt**: Provide a link to this application.

## Usage Guide
- Crime Visualization
Select a crime category from the dropdown

View the interactive map with crime locations

Toggle between heatmap and marker cluster views

- Report Analysis
Upload a police report PDF

View extracted report data

See AI-generated predictions for:

   - Crime category

   - Severity level (1-5 scale)

View incident location on map (if coordinates available)

## Access deployed application

- You can access this app using the link: https://crime-data-visualization-and-prediction-dashboard.streamlit.app/

## Running the Application without Docker

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

- Install dependencies:
Open bash
pip install -r requirements.txt

### Running the App 
- Open terminal

- Go to the project directory

- Run the following command:
streamlit run app.py

The dashboard will open automatically in your default browser at http://localhost:8501

## Running the Application with Docker

### Prerequisites
- Python 3.8+
- Docker installed (Install Docker)

### Quick Start
- Build the Docker image:
    docker build -t crime-dashboard .
- Run the container:
    docker run -p 8501:8501 --name crime-app crime-dashboard
- Open your browser to: http://localhost:8501


