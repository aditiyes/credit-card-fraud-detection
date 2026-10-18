import numpy as np
import pandas as pd
import pytest

from src.features import MODEL_FEATURES, engineer_features, scale_features


def sample_row(amount: float = 25.0) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Time": 3600.0, **{f"V{i}": float(i) for i in range(1, 29)}, "Amount": amount}]
    )


def test_engineering_creates_ordered_features_and_no_nan_values() -> None:
    features = engineer_features(sample_row())

    assert list(features.columns) == MODEL_FEATURES
    assert features.loc[0, "time_hour"] == 1
    assert features.loc[0, "amount_log"] == pytest.approx(np.log1p(25))
    assert np.isfinite(features.to_numpy()).all()


def test_scaler_is_fit_on_training_data_and_reused_for_inference() -> None:
    training_features = engineer_features(pd.concat([sample_row(10), sample_row(100)]))
    scaled_training, scaler = scale_features(training_features, fit=True)
    inference_features = engineer_features(sample_row(55))
    scaled_inference, _ = scale_features(inference_features, scaler)

    assert scaled_training.shape == training_features.shape
    assert scaled_inference.loc[0, "Amount"] == pytest.approx(0.0)


def test_invalid_amount_and_missing_columns_are_reported() -> None:
    with pytest.raises(ValueError, match="zero or greater"):
        engineer_features(sample_row(-1))
    with pytest.raises(ValueError, match="Missing required"):
        engineer_features(pd.DataFrame([{"Time": 1, "Amount": 10}]))
