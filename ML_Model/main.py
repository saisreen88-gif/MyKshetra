import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="KSHETRA Crop Stress Prediction")

# Load single pickle file
bundle = joblib.load("crop_stress_model.pkl")

model = bundle["model"]
FEATURES = bundle["features"]

# ------------------ INPUT DATA ------------------

class CropInput(BaseModel):

    crop_type: str
    soil_type: str
    growth_stage: str
    irrigation_source: str

    days_since_last_irrigation: int
    past_stress_events: int
    soil_moisture_pct: float

    temperature_c: float
    humidity_pct: float

    rainfall_last_7d_mm: float
    rainfall_forecast_3d_mm: float


# ------------------ HOME ------------------

@app.get("/")
def home():
    return {
        "project": "KSHETRA",
        "message": "Crop Stress Prediction API is running"
    }


# ------------------ MODEL INFO ------------------

@app.get("/model-info")
def model_info():
    return {
        "algorithm": "Random Forest Classifier",
        "input_features": FEATURES,
        "accuracy": round(bundle["accuracy"] * 100, 2),
        "classes": bundle["classes"]
    }


# ------------------ PREDICTION ------------------

@app.post("/predict")
def predict(data: CropInput):

    input_data = pd.DataFrame([data.model_dump()])
    input_data = input_data[FEATURES]

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    probability_output = {
        str(label): round(float(prob) * 100, 2)
        for label, prob in zip(model.classes_, probabilities)
    }

    # Irrigation priority (simple decision rules)
    if prediction == "High":
        priority = "High"
    elif prediction == "Medium":
        priority = "Moderate"
    else:
        priority = "Low"

    return {
        "predicted_stress_level": str(prediction),
        "probabilities": probability_output,
        "irrigation_priority": priority
    }