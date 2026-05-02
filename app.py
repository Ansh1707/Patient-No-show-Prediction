import joblib
import json
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import os

# Load model and threshold saved by your training script
MODEL_PATH = "model/best_no_show_model.pkl"
THRESHOLD_PATH = "model/threshold.json"  # optional, if you saved threshold separately

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

if os.path.exists(THRESHOLD_PATH):
    with open(THRESHOLD_PATH) as f:
        THRESHOLD = json.load(f)["threshold"]
else:
    THRESHOLD = 0.304  # fallback threshold, adjust if needed

app = FastAPI(title="Patient No-Show Prediction API")

# Enable CORS so frontend (e.g., on different port) can call API
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Appointment(BaseModel):
    Gender:       str = Field(..., example="F")
    Age:          int = Field(..., ge=0, le=115, example=30)
    Scholarship:  int = Field(..., ge=0, le=1, example=0)
    Hipertension: int = Field(..., ge=0, le=1, example=0)
    Diabetes:     int = Field(..., ge=0, le=1, example=0)
    Alcoholism:   int = Field(..., ge=0, le=1, example=0)
    Handcap:      int = Field(..., ge=0, le=4, example=0)
    SMS_received: int = Field(..., ge=0, le=1, example=1)
    wait_days:    int = Field(..., ge=-7, le=365, example=2)


@app.get("/")
def root():
    return {"message": "Patient No-Show Prediction API is running."}

@app.post("/predict")
def predict(appointment: Appointment):
    data = pd.DataFrame([appointment.dict()])
    prob_no_show = float(model.predict_proba(data)[0][1])
    prediction = int(prob_no_show >= THRESHOLD)
    return {"probability_no_show": round(prob_no_show, 3),
            "prediction": prediction}  # 1 = no-show, 0 = show
