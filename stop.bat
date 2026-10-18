@echo off
echo ==========================================
echo 🛑 Shutting down MLOps Pipeline...
echo ==========================================

echo.
echo 🗑️  1. Removing Kubernetes resources...
kubectl delete -f deploy/k8s_deployment.yaml
kubectl delete -f deploy/k8s_service.yaml

echo.
echo 🗑️  2. Stopping and removing monitoring containers...
docker stop prometheus grafana
docker rm prometheus grafana

echo.
echo ✅ Pipeline successfully shut down.