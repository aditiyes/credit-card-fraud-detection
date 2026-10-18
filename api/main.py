from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import Transaction
from src.decisions import decision_for_probability
from src.features import MODEL_FEATURES, engineer_features, scale_features
from src.store import recent_predictions, save_prediction

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/fraud_model.joblib"))
THRESHOLDS = {"review": 0.30, "block": 0.70}


def _load_runtime() -> dict[str, Any]:
    if MODEL_PATH.is_file():
        artifact = joblib.load(MODEL_PATH)
        if not isinstance(artifact, dict) or not {"model", "scaler", "features"} <= set(
            artifact
        ):
            raise RuntimeError(f"Model artifact at {MODEL_PATH} has an invalid format.")
        if artifact["features"] != MODEL_FEATURES:
            raise RuntimeError(f"Model artifact at {MODEL_PATH} has an unexpected feature order.")
        import shap

        artifact["explainer"] = shap.TreeExplainer(artifact["model"])
        artifact["mode"] = "kaggle_trained"
        logger.info("Loaded Kaggle-trained model from %s", MODEL_PATH)
        return artifact

    logger.warning(
        "No trained model found at %s. API is running in clearly labeled synthetic demo mode.",
        MODEL_PATH,
    )
    return {"mode": "synthetic_demo", "metrics": None}


runtime = _load_runtime()
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Transaction scoring with an explicit synthetic-demo mode until a model is trained.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _demo_score(features: pd.DataFrame) -> tuple[float, list[dict[str, Any]]]:
    row = features.iloc[0]
    raw_signals = [
        ("V14", max(0.0, -float(row["V14"]) - 1.5), "unusually low V14"),
        ("V10", max(0.0, -float(row["V10"]) - 1.5), "unusually low V10"),
        ("V12", max(0.0, -float(row["V12"]) - 1.5), "unusually low V12"),
        ("Amount", max(0.0, float(row["amount_log"]) - 7.0), "unusually large amount"),
    ]
    weighted_signal = sum(signal for _, signal, _ in raw_signals)
    probability = float(1 / (1 + np.exp(-(weighted_signal - 1.4))))
    explanations = [
        {"feature": name, "impact": round(signal, 4), "description": description}
        for name, signal, description in sorted(raw_signals, key=lambda item: item[1], reverse=True)
        if signal > 0
    ][:3]
    return probability, explanations


def _score(transaction: Transaction) -> tuple[float, list[dict[str, Any]]]:
    raw_features = pd.DataFrame([transaction.model_dump()])
    features = engineer_features(raw_features)
    if runtime["mode"] == "synthetic_demo":
        return _demo_score(features)

    scaled_features, _ = scale_features(features, runtime["scaler"])
    probability = float(runtime["model"].predict_proba(scaled_features)[0, 1])
    values = np.asarray(runtime["explainer"].shap_values(scaled_features)).reshape(-1)
    explanations = [
        {
            "feature": MODEL_FEATURES[index],
            "impact": round(float(values[index]), 6),
            "description": (
                "increases fraud risk" if values[index] > 0 else "decreases fraud risk"
            ),
        }
        for index in np.argsort(np.abs(values))[::-1][:5]
    ]
    return probability, explanations


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": runtime["mode"]}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    if runtime["mode"] == "synthetic_demo":
        return {
            "mode": runtime["mode"],
            "message": "Evaluation metrics are unavailable until the Kaggle model is trained.",
        }
    return {"mode": runtime["mode"], **runtime.get("metrics", {})}


@app.get("/transactions")
def transactions() -> dict[str, list[dict[str, Any]]]:
    return {"items": recent_predictions()}


@app.post("/predict")
def predict(transaction: Transaction) -> dict[str, Any]:
    try:
        probability, explanations = _score(transaction)
    except (ValueError, FloatingPointError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    decision = decision_for_probability(probability)
    prediction_id = save_prediction(
        amount=transaction.Amount,
        probability=probability,
        decision=decision,
        mode=runtime["mode"],
        explanations=explanations,
    )
    return {
        "prediction_id": prediction_id,
        "fraud_probability": probability,
        "decision": decision,
        "thresholds": THRESHOLDS,
        "mode": runtime["mode"],
        "explanation_type": (
            "synthetic_demo_signals"
            if runtime["mode"] == "synthetic_demo"
            else "shap"
        ),
        "top_factors": explanations,
    }
