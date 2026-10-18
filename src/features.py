from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
RAW_FEATURES = ["Time", *PCA_FEATURES, "Amount"]
MODEL_FEATURES = [*RAW_FEATURES, "amount_log", "time_hour"]
SCALED_FEATURES = ["Time", "Amount"]


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(RAW_FEATURES) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required transaction features: {', '.join(missing)}")

    features = data.loc[:, RAW_FEATURES].astype(float).copy()
    if (features["Amount"] < 0).any():
        raise ValueError("Amount must be zero or greater.")
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Transaction features must contain only finite numbers.")

    features["amount_log"] = np.log1p(features["Amount"])
    features["time_hour"] = (features["Time"] // 3600) % 24
    return features.loc[:, MODEL_FEATURES]


def scale_features(
    features: pd.DataFrame,
    scaler: RobustScaler | None = None,
    *,
    fit: bool = False,
) -> tuple[pd.DataFrame, RobustScaler]:
    if fit:
        fitted_scaler = RobustScaler()
        scaled = features.copy()
        scaled.loc[:, SCALED_FEATURES] = fitted_scaler.fit_transform(
            scaled.loc[:, SCALED_FEATURES]
        )
        return scaled, fitted_scaler

    if scaler is None:
        raise ValueError("A fitted scaler is required when fit is false.")
    scaled = features.copy()
    scaled.loc[:, SCALED_FEATURES] = scaler.transform(
        scaled.loc[:, SCALED_FEATURES]
    )
    return scaled, scaler


def validate_feature_columns(columns: Sequence[str]) -> None:
    if list(columns) != MODEL_FEATURES:
        raise ValueError("Model feature columns do not match the expected feature order.")
