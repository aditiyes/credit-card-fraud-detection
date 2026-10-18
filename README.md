# Credit Card Fraud Detection

An end-to-end fraud risk monitor with a FastAPI scoring API, Streamlit analyst dashboard, SQLite prediction log, XGBoost training pipeline, precision-recall evaluation, and SHAP explanations.

> **Demo safety:** The app starts in clearly labeled synthetic-demo mode when there is no trained model artifact. Demo scores and signals are illustrative only: they are not learned from Kaggle data and must not be used for real decisions. Train the model on the Kaggle dataset before using model scores as predictions.

## Features

- XGBoost with `scale_pos_weight` for highly imbalanced transaction data.
- Stratified train/test split before fitting the robust scaler or model.
- AUPRC (average precision), precision, recall, F1, ROC-AUC, confusion matrix, and a precision-recall curve.
- FastAPI `POST /predict` with risk score, action (`APPROVE`, `REVIEW`, `BLOCK`), thresholds, and top factors.
- SHAP TreeExplainer factors for a trained XGBoost model.
- Streamlit scoring form, risk/activity view, metrics, and analyst transaction feed.
- SQLite audit records; only amount, risk, action, model mode, and explanations are stored. Raw PCA transaction features are not logged.
- GitHub Actions, Docker Compose, and a Render Blueprint for a separate API and dashboard.

## Run locally

Use Python 3.11 or newer:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn api.main:app --reload
```

In another terminal, from the project directory:

```powershell
.\.venv\Scripts\Activate.ps1
$env:API_URL = "http://127.0.0.1:8000"
streamlit run dashboard/app.py
```

Open the Streamlit URL shown in the terminal. The API is at `http://127.0.0.1:8000`; interactive API documentation is at `/docs`.

### Train with the Kaggle dataset

Download the Kaggle **Credit Card Fraud Detection** dataset (`mlg-ulb/creditcardfraud`) and place `creditcard.csv` at `data/raw/creditcard.csv`. The dataset is intentionally excluded from Git.

```powershell
python -m src.train --data data/raw/creditcard.csv
```

This creates `models/fraud_model.joblib`, `reports/metrics.json`, and `reports/precision_recall_curve.png`. Restart the API to load the trained model. Keep the model artifact out of public Git; deploy it through a protected artifact store or a private deployment configuration.

The training split is stratified and performed before fitting `RobustScaler`. The initial benchmark uses XGBoost class weighting; SMOTE is intentionally not applied before splitting. For a final report, compare class weighting with training-only SMOTE and tune the decision threshold against validation data and business costs.

### Score a transaction

Send `Time`, `V1`–`V28`, and `Amount` to `POST /predict`. Each field is required; `Time` and `Amount` must be non-negative. The response includes a probability, action thresholds, explanation mode, and top factors. `GET /health`, `/metrics`, and `/transactions` provide service health, evaluation metrics, and recent scores.

Default actions are:

| Fraud probability | Action |
| --- | --- |
| Below 0.30 | Approve |
| 0.30 to below 0.70 | Review |
| 0.70 or above | Block |

The 0.5 classification threshold in evaluation metrics is reported separately from the operational action thresholds.

## Run with Docker

```powershell
docker compose up --build
```

Open the dashboard at `http://localhost:8501` and the API docs at `http://localhost:8000/docs`. Local prediction history is stored in `data/predictions.db`.

## Deploy to Render

1. Push this repository to GitHub.
2. Sign in to Render and create a new **Blueprint** from the repository.
3. Approve the two services in `render.yaml`: the FastAPI service and the Streamlit dashboard.
4. Wait for both health/build checks, then open the dashboard service URL.

The free Render plan can sleep when idle. Its temporary filesystem is not durable, so demo prediction history can be cleared when the API restarts. Use a persistent database and a protected model-artifact delivery mechanism for production.

## Tests

```powershell
python -m pytest -q
```
