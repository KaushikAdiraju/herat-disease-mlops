@echo off
echo ==========================================
echo 🚀 Starting End-to-End MLOps Pipeline...
echo ==========================================

echo.
echo 📦 1. Deploying FastAPI to Kubernetes...
kubectl apply -f deploy/k8s_deployment.yaml
kubectl apply -f deploy/k8s_service.yaml

echo.
echo 📊 2. Starting Prometheus Data Collector...
docker rm -f prometheus 2>NUL
docker run -d --name prometheus -p 9090:9090 -v "%cd%/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml" prom/prometheus

echo.
echo 📈 3. Starting Grafana Dashboard...
docker rm -f grafana 2>NUL
docker run -d --name grafana -p 3001:3000 grafana/grafana

echo.
echo ✅ All systems launched!
echo ➡️  API Swagger UI: http://localhost:8081/docs
echo ➡️  Prometheus:     http://localhost:9090
echo ➡️  Grafana:        http://localhost:3001
echo.
echo 🔌 Forwarding Kubernetes port (Press Ctrl+C to stop)...
kubectl port-forward svc/heart-disease-api-service 8081:80