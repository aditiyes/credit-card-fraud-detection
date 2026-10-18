from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.features import MODEL_FEATURES, engineer_features, scale_features


def train(dataset_path: Path, model_path: Path, reports_dir: Path) -> dict[str, object]:
    data = pd.read_csv(dataset_path)
    if "Class" not in data.columns:
        raise ValueError("Dataset must contain a 'Class' target column.")
    if not data["Class"].isin([0, 1]).all():
        raise ValueError("The 'Class' target must contain only 0 and 1.")
    if data["Class"].nunique() != 2:
        raise ValueError("Training requires both legitimate and fraudulent transactions.")

    raw_features = engineer_features(data)
    labels = data["Class"].astype(int)
    train_features, test_features, y_train, y_test = train_test_split(
        raw_features,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )
    train_features, scaler = scale_features(train_features, fit=True)
    test_features, _ = scale_features(test_features, scaler)

    fraud_count = int(y_train.sum())
    legitimate_count = int(len(y_train) - fraud_count)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=legitimate_count / fraud_count,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_features, y_train)
    probabilities = model.predict_proba(test_features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)

    metrics: dict[str, object] = {
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "precision_at_0_5": float(precision_score(y_test, predictions, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix_at_0_5": confusion_matrix(y_test, predictions).tolist(),
        "test_transactions": int(len(y_test)),
        "fraud_transactions_in_test": int(y_test.sum()),
        "threshold": 0.5,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": MODEL_FEATURES,
            "metrics": metrics,
        },
        model_path,
    )
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"XGBoost (AP={metrics['average_precision']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "precision_recall_curve.png", dpi=150)
    plt.close()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the fraud detection model.")
    parser.add_argument("--data", type=Path, default=Path("data/raw/creditcard.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/fraud_model.joblib"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.model, args.reports), indent=2))


if __name__ == "__main__":
    main()
