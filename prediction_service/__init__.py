"""Shared inference for Django and the original FastAPI application."""
import hashlib
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES = ('Gender', 'Age', 'Scholarship', 'Hipertension', 'Diabetes',
            'Alcoholism', 'Handcap', 'SMS_received', 'wait_days')


class PredictionUnavailable(Exception):
    """Model artifacts cannot be used for inference."""


@dataclass(frozen=True)
class PredictionResult:
    probability_no_show: float
    prediction: int
    threshold: float
    model_version: str

    def response(self):
        return {'probability_no_show': round(self.probability_no_show, 3),
                'prediction': self.prediction}


class PredictionService:
    def __init__(self, model_path, threshold_path):
        try:
            self.model = joblib.load(model_path)
            self.model_version = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
            path = Path(threshold_path)
            self.threshold = float(json.loads(path.read_text())['threshold']) if path.exists() else 0.304
            if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
                raise ValueError('Invalid threshold')
        except Exception as exc:
            raise PredictionUnavailable('Prediction model is unavailable. Configure valid model artifacts.') from exc

    def predict(self, features):
        try:
            data = pd.DataFrame([{name: features[name] for name in FEATURES}], columns=FEATURES)
            probability = float(self.model.predict_proba(data)[0][1])
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError('Invalid model probability')
        except Exception as exc:
            raise PredictionUnavailable('Prediction model could not process this appointment.') from exc
        return PredictionResult(probability, int(probability >= self.threshold),
                                self.threshold, self.model_version)


@lru_cache(maxsize=4)
def _load_service(model_path, threshold_path):
    return PredictionService(model_path, threshold_path)


def get_prediction_service():
    # Only load trusted, operator-provided artifacts: joblib is executable serialization.
    return _load_service(os.environ.get('MODEL_PATH', str(BASE_DIR / 'model/best_no_show_model.pkl')),
                         os.environ.get('THRESHOLD_PATH', str(BASE_DIR / 'model/threshold.json')))
