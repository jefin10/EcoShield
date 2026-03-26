from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Vehicle Emission Predictor")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
bundle = joblib.load(os.path.join(BASE_DIR, "xgboost_emission_model.pkl"))
model          = bundle["model"]
le_dict        = bundle["label_encoders"]
feature_columns = bundle["feature_columns"]   # exact column order from training

class EmissionInput(BaseModel):
    speed_kmph:     float
    idle_pct:       float
    elevation_grad: float
    vehicle_type:   str   # "2W", "Bus", "Car", "Truck"
    bs_norm:        str   # "BS4", "BS6"
    fuel_type:      str   # "Petrol", "Diesel", "Hybrid", "Electric"

class EmissionOutput(BaseModel):
    co2_g_per_km:  float
    co2_kg_per_km: float
    fuel_type:     str
    vehicle_type:  str
    bs_norm:       str

@app.get("/")
def home():
    return {"message": "Emission Prediction API is live 🚀"}

@app.post("/predict", response_model=EmissionOutput)
def predict(data: EmissionInput):
    # Electric vehicles have zero tailpipe emissions — return immediately
    if data.fuel_type.strip().lower() == "electric":
        return {
            "co2_g_per_km":  0.0,
            "co2_kg_per_km": 0.0,
            "fuel_type":     data.fuel_type,
            "vehicle_type":  data.vehicle_type,
            "bs_norm":       data.bs_norm,
        }

    df = pd.DataFrame([data.dict()])

    # Apply the same label encoding used during training
    for col, le in le_dict.items():
        if df[col][0] not in le.classes_:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown value '{df[col][0]}' for '{col}'. "
                       f"Allowed values: {le.classes_.tolist()}"
            )
        df[col] = le.transform(df[col].astype(str))

    # Ensure column order matches training
    df = df[feature_columns]

    co2_g_per_km = float(model.predict(df)[0])
    co2_kg_per_km = round(co2_g_per_km / 1000.0, 6)

    return {
        "co2_g_per_km":  round(co2_g_per_km, 2),
        "co2_kg_per_km": co2_kg_per_km,
        "fuel_type":     data.fuel_type,
        "vehicle_type":  data.vehicle_type,
        "bs_norm":       data.bs_norm,
    }

@app.get("/model-info")
def model_info():
    """Returns the allowed values for categorical inputs."""
    return {
        col: le.classes_.tolist()
        for col, le in le_dict.items()
    }