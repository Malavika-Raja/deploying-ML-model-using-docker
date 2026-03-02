# California Housing Price Prediction API

This project trains a **Random Forest Regressor** on the California Housing dataset and exposes predictions through a **FastAPI** service. The application is fully containerized using Docker for consistent and portable deployment.

---

## 🚀 Features

- Trains a Random Forest regression model  
- REST API built with FastAPI  
- Interactive API documentation  
- Dockerized for easy deployment  

---

## 🛠 Tech Stack

- Python  
- scikit-learn  
- FastAPI  
- Uvicorn  
- Docker  

---

## ▶ Run Locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

API available at:
```bash
http://127.0.0.1:8000/docs
```
---
## 🐳 Run with Docker

```bash
docker build -t housing-api .
docker run -d -p 8000:8000 housing-api
```