from api.app import app, load_model  # noqa: E402
import os
import sys
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

client = TestClient(app)


def test_health_endpoint():
    """Verifies the /health endpoint responds with HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_artifact_loads_and_predicts():
    """Verifies that the packaged model pipeline exists and outputs valid probabilities."""
    pipeline = load_model()
    assert pipeline is not None


def test_predict_endpoint_valid_input():
    """Tests the FastAPI /predict endpoint with sample clinical JSON payload."""
    sample_payload = {
        "age": 63.0, "sex": 1, "cp": 1, "trestbps": 145.0, "chol": 233.0,
        "fbs": 1, "restecg": 2, "thalach": 150.0, "exang": 0, "oldpeak": 2.3,
        "slope": 3, "ca": 0.0, "thal": 6.0
    }
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["disease_probability"] <= 1.0


def test_predict_endpoint_invalid_input_rejected():
    """Verifies Pydantic validation rejects out-of-range clinical values."""
    bad_payload = {
        "age": -10.0, "sex": 1, "cp": 1, "trestbps": 145.0, "chol": 233.0,
        "fbs": 1, "restecg": 2, "thalach": 150.0, "exang": 0, "oldpeak": 2.3,
        "slope": 3, "ca": 0.0, "thal": 6.0
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
