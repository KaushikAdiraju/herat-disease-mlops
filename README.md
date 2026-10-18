# End-to-End MLOps Pipeline: Heart Disease Prediction

This repository contains a complete Machine Learning Operations (MLOps) pipeline developed for the BITS Pilani M.Tech WILP assignment. It demonstrates the full lifecycle of a machine learning model, from automated testing and experiment tracking to containerized Kubernetes deployment and active production monitoring.

## 🏗️ Architecture & Tech Stack
*   **API Framework:** FastAPI
*   **Experiment Tracking:** MLflow
*   **Containerization:** Docker
*   **Orchestration:** Kubernetes (Local via Docker Desktop)
*   **Monitoring & Observability:** Prometheus & Grafana
*   **CI/CD:** GitHub Actions

## 📂 Project Structure
```text
├── .github/workflows/   # CI/CD pipelines (Flake8 & Pytest)
├── api/                 # FastAPI application code
├── deploy/              # Dockerfile and Kubernetes manifests (.yaml)
├── monitoring/          # Prometheus configuration files
├── src/                 # Model training scripts and data preprocessing
├── tests/               # Unit tests for the API and model
└── requirements.txt     # Python dependencies

How to Run the Project Locally
1. Prerequisites
Python 3.10+

Docker Desktop (ensure Kubernetes is enabled in settings)

2. Model Training & Experiment Tracking
Run the training script to train the models (Logistic Regression, Random Forest, and Gradient Boosting) and launch the MLflow UI to compare the runs.

python src/train.py
mlflow ui

Access MLflow at: http://localhost:5000

3. Containerization & Kubernetes Deployment
Build the Docker image and deploy the FastAPI application to your local Kubernetes cluster.

docker build -t heart-disease-mlops-api:v1 -f deploy/Dockerfile .
kubectl apply -f deploy/k8s_deployment.yaml
kubectl apply -f deploy/k8s_service.yaml

# Route traffic to the cluster
kubectl port-forward svc/heart-disease-api-service 8081:80

Access the API and Swagger UI at: http://localhost:8081/docs

4. Monitoring Stack (Prometheus & Grafana)
In a separate terminal, launch the observability containers to scrape and visualize API traffic.

# Start Prometheus
docker run -d --name prometheus -p 9090:9090 -v "$(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml" prom/prometheus

# Start Grafana
docker run -d --name grafana -p 3001:3000 grafana/grafana

Accessing the Dashboards:

Prometheus: Navigate to http://localhost:9090/targets to verify the FastAPI target is UP.

Grafana: Navigate to http://localhost:3001 (Username: admin, Password: admin).

Connect Data: Add Prometheus as a data source using URL http://host.docker.internal:9090.

Visualize: Create a dashboard querying http_requests_total to view live API traffic.

🧹 Cleanup
To spin down the monitoring containers and free up local ports:

docker stop prometheus grafana
docker rm prometheus grafana