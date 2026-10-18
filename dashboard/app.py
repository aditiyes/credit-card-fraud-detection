from __future__ import annotations

import os
import random

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from src.config import normalize_service_url

API_URL = normalize_service_url(os.getenv("API_URL", "http://127.0.0.1:8000"))
REQUEST_TIMEOUT = 15

st.set_page_config(page_title="Fraud Risk Monitor", page_icon="🛡️", layout="wide")
st.title("🛡️ Fraud Risk Monitor")
st.caption("Credit-card transaction scoring · XGBoost · precision-recall focused")


def get_api(path: str) -> dict:
    response = requests.get(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def sample_transaction() -> dict[str, float]:
    return {
        "Time": float(random.randint(0, 172800)),
        **{f"V{i}": round(random.gauss(0, 1), 3) for i in range(1, 29)},
        "Amount": round(random.uniform(2, 5000), 2),
    }


try:
    health = get_api("/health")
    metrics = get_api("/metrics")
    history = get_api("/transactions").get("items", [])
except requests.RequestException as error:
    st.error(f"Scoring API is unavailable at {API_URL}: {error}")
    st.stop()

demo_mode = health["mode"] == "synthetic_demo"
if demo_mode:
    st.warning(
        "Synthetic demo mode: scores and explanations are illustrative only, not trained on the "
        "Kaggle dataset. Run the training command and deploy the generated model to enable "
        "validated XGBoost scoring."
    )
else:
    st.success("Kaggle-trained XGBoost model is loaded. Per-transaction factors use SHAP.")

left, middle, right = st.columns(3)
left.metric("Scored transactions", len(history))
middle.metric("Review / block alerts", sum(item["decision"] != "APPROVE" for item in history))
right.metric("Model mode", "Synthetic demo" if demo_mode else "Kaggle-trained XGBoost")

st.subheader("Score a transaction")
with st.form("score_transaction"):
    amount = st.number_input("Transaction amount", min_value=0.0, value=125.0, step=10.0)
    time_value = st.number_input("Elapsed seconds", min_value=0.0, value=10000.0, step=60.0)
    use_sample = st.checkbox("Generate randomized PCA features", value=True)
    with st.expander("PCA features (V1–V28)"):
        cols = st.columns(4)
        pca_values = {
            f"V{i}": cols[(i - 1) % 4].number_input(
                f"V{i}", value=0.0, step=0.1, key=f"feature_{i}"
            )
            for i in range(1, 29)
        }
    submitted = st.form_submit_button("Score transaction", type="primary")

if submitted:
    payload = sample_transaction() if use_sample else {"Time": time_value, **pca_values}
    payload["Time"] = time_value
    payload["Amount"] = amount
    try:
        response = requests.post(
            f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        result = response.json()
        st.session_state["last_result"] = result
    except requests.RequestException as error:
        st.error(f"Scoring failed: {error}")

result = st.session_state.get("last_result")
if result:
    score_col, decision_col = st.columns(2)
    score_col.metric("Fraud probability", f"{result['fraud_probability']:.1%}")
    decision_col.metric("Recommended action", result["decision"])
    st.progress(result["fraud_probability"])
    if result["top_factors"]:
        st.caption(f"Explanation: {result['explanation_type'].replace('_', ' ').title()}")
        st.dataframe(pd.DataFrame(result["top_factors"]), hide_index=True, use_container_width=True)

st.subheader("Recent scoring activity")
if history:
    activity = pd.DataFrame(history)
    activity["created_at"] = pd.to_datetime(activity["created_at"], utc=True)
    chart = px.scatter(
        activity.sort_values("created_at"),
        x="created_at",
        y="fraud_probability",
        color="decision",
        hover_data=["amount", "mode"],
        title="Risk score by transaction time",
        labels={"created_at": "Scored at", "fraud_probability": "Fraud probability"},
        color_discrete_map={"APPROVE": "#20a779", "REVIEW": "#eea83b", "BLOCK": "#d9534f"},
    )
    chart.update_yaxes(tickformat=".0%")
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(
        activity[["created_at", "amount", "fraud_probability", "decision", "mode"]]
        .rename(columns={"created_at": "Scored at", "amount": "Amount", "fraud_probability": "Fraud probability", "decision": "Decision", "mode": "Mode"}),
        hide_index=True,
        use_container_width=True,
    )
else:
    st.info("No transactions scored yet. Use the form above to populate the activity feed.")

st.subheader("Model evaluation")
if demo_mode:
    st.info(metrics["message"])
else:
    metric_cols = st.columns(4)
    metric_cols[0].metric("AUPRC", f"{metrics['average_precision']:.3f}")
    metric_cols[1].metric("Precision @ 0.5", f"{metrics['precision_at_0_5']:.3f}")
    metric_cols[2].metric("Recall @ 0.5", f"{metrics['recall_at_0_5']:.3f}")
    metric_cols[3].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    st.caption(
        "Threshold-dependent classification results use 0.5 for evaluation; decisions use "
        "0.30 (review) and 0.70 (block). Accuracy is intentionally not emphasized."
    )
    st.json({"Confusion matrix [TN, FP; FN, TP]": metrics["confusion_matrix_at_0_5"]})
