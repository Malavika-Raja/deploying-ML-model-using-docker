# California Housing Price Prediction API

A machine learning API built using the California Housing dataset.  
The project trains a Random Forest Regressor model and exposes prediction functionality through a FastAPI application. The service is containerized using Docker and deployed on Render.

---

## 🚀 Live Deployment

The API is publicly accessible via Render

---

## 📌 Project Overview

This project:

- Uses the California Housing dataset from `sklearn`
- Trains a `RandomForestRegressor`
- Saves the trained model using `joblib`
- Builds a REST API using FastAPI
- Containerizes the application with Docker
- Deploys the service to the cloud using Render

---

## 🛠 Tech Stack

- Python  
- Scikit-learn  
- FastAPI  
- Uvicorn  
- Docker 
- Render 

---

## ⚙️ Run locally 

### 1. Clone the repository

```bash
git clone https://github.com/Malavika-Raja/deploying-ML-model-using-docker.git
cd <repository-name>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model (if needed)

```bash
python train.py
```

### 4. Run the FastAPI app

```bash
uvicorn main:app --reload
```

Visit:

http://127.0.0.1:8000/docs


---
## 🐳 Run with Docker

Build the image:

```bash
docker build -t dock .
```
Run the container:

```bash
docker run -d -p 8000:8000 dock
```

---

## ☁️ Deployment

This application is deployed using:

- Docker containerization
- Render cloud platform
- GitHub integration with auto-deploy

Any push to the main branch triggers a new deployment.

