import os
import sys
import time
import logging
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

# Ensure root directory is in path so custom transformers in src.preprocess can be unpickled
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.preprocess  # noqa: E402, F401

# Configure structured logging for API monitoring (Task 8)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
)
logger = logging.getLogger("heart_disease_api")

MODEL_PATH = os.path.join("models", "best_model_pipeline.joblib")
model_pipeline = None


def load_model():
    """Loads the serialized end-to-end preprocessing + model pipeline."""
    global model_pipeline
    if model_pipeline is None:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model artifact not found at {MODEL_PATH}. Run src/train.py first.")
            raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
        model_pipeline = joblib.load(MODEL_PATH)
        logger.info(f"Successfully loaded model pipeline from {MODEL_PATH}")
    return model_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup if available
    if os.path.exists(MODEL_PATH):
        load_model()
    yield


app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Production MLOps API for predicting cardiovascular disease risk from clinical markers.",
    version="1.0.0",
    lifespan=lifespan
)

# Expose /metrics endpoint for Prometheus scraping (Task 8)
Instrumentator().instrument(app).expose(app)


class PatientClinicalData(BaseModel):
    age: float = Field(..., ge=1, le=120, example=63.0, description="Age in years")
    sex: int = Field(..., ge=0, le=1, example=1, description="Sex (1 = male; 0 = female)")
    cp: int = Field(..., ge=1, le=4, example=1, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=300, example=145.0, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=80, le=700, example=233.0, description="Serum cholesterol in mg/dl")
    fbs: int = Field(..., ge=0, le=1, example=1, description="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)")
    restecg: int = Field(..., ge=0, le=2, example=2, description="Resting electrocardiographic results (0-2)")
    thalach: float = Field(..., ge=50, le=250, example=150.0, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, example=0, description="Exercise induced angina (1 = yes; 0 = no)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, example=2.3, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, example=3, description="Slope of the peak exercise ST segment (1-3)")
    ca: float = Field(..., ge=0.0, le=4.0, example=0.0,
                      description="Number of major vessels (0-3) colored by fluoroscopy")
    thal: float = Field(..., ge=3.0, le=7.0, example=6.0,
                        description="Thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)")


class PredictionResponse(BaseModel):
    prediction: int
    risk_label: str
    confidence: float
    disease_probability: float


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log every incoming request and latency for Task 8 monitoring."""
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"Method={request.method} Path={request.url.path} "
        f"Status={response.status_code} Latency={duration_ms:.2f}ms"
    )
    return response


@app.get("/health")
def health_check():
    """Health check endpoint for Kubernetes liveness/readiness probes."""
    return {"status": "healthy", "model_loaded": model_pipeline is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_heart_disease(patient: PatientClinicalData):
    """
    Accepts patient clinical markers in JSON format and returns heart disease
    risk prediction (0 or 1) along with model confidence.
    """
    try:
        pipeline = load_model()
        input_dict = patient.model_dump()
        logger.info(f"Received /predict payload: {input_dict}")

        input_df = pd.DataFrame([input_dict])
        pred_class = int(pipeline.predict(input_df)[0])
        prob_disease = float(pipeline.predict_proba(input_df)[0][1])

        confidence = prob_disease if pred_class == 1 else (1.0 - prob_disease)
        risk_label = "High Risk (Heart Disease Present)" if pred_class == 1 else "Low Risk (Heart Disease Absent)"

        result = {
            "prediction": pred_class,
            "risk_label": risk_label,
            "confidence": round(confidence, 4),
            "disease_probability": round(prob_disease, 4)
        }
        logger.info(f"Prediction output: {result}")
        return result

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
