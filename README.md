# Smart Manufacturing Machines Efficiency Prediction

## Project Overview

This project implements a machine learning system to predict the efficiency of manufacturing machines based on various operational parameters. The system uses a Logistic Regression model to classify machine efficiency as either "Efficient" or "Inefficient" based on real-time operational data.

## Features

- Real-time machine efficiency prediction
- Web interface for input and predictions
- Automated data processing pipeline
- Model training and evaluation
- Docker containerization
- Kubernetes deployment support
- Jenkins CI/CD pipeline integration
- Comprehensive logging system

## Project Structure

```
├── application.py           # Flask web application
├── Dockerfile              # Docker configuration
├── Jenkinsfile            # Jenkins pipeline configuration
├── requirements.txt       # Python dependencies
├── setup.py              # Project setup configuration
├── artifacts/            # Stored models and data
│   ├── models/          # Trained model files
│   ├── processed/       # Processed datasets
│   └── raw/            # Raw input data
├── logs/                # Application logs
├── manifests/          # Kubernetes configuration
│   ├── deployment.yaml
│   └── service.yaml
├── notebook/           # Jupyter notebooks
├── pipeline/          # Training pipeline
├── src/              # Source code
│   ├── data_processing.py
│   ├── model_training.py
│   ├── logger.py
│   └── custome_exception.py
├── static/           # Web assets
└── templates/        # HTML templates
```

## Features Used in Prediction

- Operation Mode
- Temperature (°C)
- Vibration (Hz)
- Power Consumption (kW)
- Network Latency (ms)
- Packet Loss (%)
- Quality Control Defect Rate (%)
- Production Speed (units/hr)
- Predictive Maintenance Score
- Error Rate (%)
- Temporal Features (Year, Month, Day, Hour)

## Setup and Installation

### Prerequisites

- Python 3.11+
- Docker
- Kubernetes (for deployment)
- Jenkins (for CI/CD)

### Local Development Setup

1. Create a virtual environment:

```bash
python -m venv Machines_efficiency_prediction
source Machines_efficiency_prediction/Scripts/activate  # On Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the data processing pipeline:

```bash
python src/data_processing.py
```

4. Train the model:

```bash
python src/model_training.py
```

5. Start the Flask application:

```bash
python application.py
```

### Docker Deployment

1. Build the Docker image:

```bash
docker build -t machine-efficiency-predictor .
```

2. Run the container:

```bash
docker run -p 5000:5000 machine-efficiency-predictor
```

### Kubernetes Deployment

1. Apply the Kubernetes manifests:

```bash
kubectl apply -f manifests/
```

## API Endpoints

### Web Interface

- `GET /`: Home page with prediction form
- `POST /predict`: Endpoint for making predictions

## Model Performance

The system uses a Logistic Regression model with the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score

*Note: Actual metrics will be logged during model training*

## Logging

The application uses a custom logging system that:

- Logs all predictions
- Tracks errors and exceptions
- Maintains dated log files
- Stores logs in the `logs/` directory

## CI/CD Pipeline

The project includes a Jenkins pipeline that:

- Runs tests
- Builds Docker image
- Deploys to Kubernetes
- Monitors application health

