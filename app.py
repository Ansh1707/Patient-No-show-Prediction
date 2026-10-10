from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from prediction_service import get_prediction_service, PredictionUnavailable
from fastapi import HTTPException

app = FastAPI(title="Patient No-Show Prediction API")

# Enable CORS so frontend (e.g., on different port) can call API
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Appointment(BaseModel):
    Gender:       Literal["F", "M"] = Field(..., example="F")
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
    try:
        return get_prediction_service().predict(appointment.dict()).response()
    except PredictionUnavailable as exc:
        raise HTTPException(status_code=503, detail="Prediction service is unavailable.") from exc
