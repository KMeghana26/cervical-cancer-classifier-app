"""
Streamlit app
Loads pre-trained classifiers and evaluates them on an uploaded test CSV.
No retraining happens here - models are loaded from model/*.pkl.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

MODEL_DIR = Path("model")
TARGET = "Diagnosis"
CLASS_NAMES = ["No Cancer", "Cancer"]

MODEL_FILES = {
    "Logistic Regression": ("logistic_regression.pkl", True),
    "Decision Tree": ("decision_tree.pkl", False),
    "KNN": ("knn.pkl", True),
    "Naive Bayes": ("naive_bayes.pkl", False),
    "Support Vector Machine": ("support_vector_machine.pkl", True),
}

st.set_page_config(page_title="Breast Cancer Risk Classifier", page_icon="🩺", layout="wide")

feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")


def run_model(name, X, y_true):
    filename, needs_scaling = MODEL_FILES[name]
    model = joblib.load(MODEL_DIR / filename)
    X_input = scaler.transform(X) if needs_scaling else X
    y_pred = model.predict(X_input)
    y_proba = model.predict_proba(X_input)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return y_pred, y_proba, metrics


st.title("🩺 Breast Cancer Risk Classification")
st.caption("Evaluate 5 pre-trained classifiers on your test data — no retraining happens here.")

with st.sidebar:
    st.header("Input")
    uploaded_file = st.file_uploader(
        "Upload test CSV", type="csv",
        help=f"Must contain all {len(feature_columns)} feature columns plus the '{TARGET}' target column."
    )
    st.markdown("---")
    st.markdown(
        "**Dataset:** Breast Cancer Wisconsin (Diagnostic)\n\n"
        "**Models:** Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM"
    )

if uploaded_file is None:
    st.info("👈 Upload a test CSV from the sidebar to get started (e.g. the provided `test_data.csv`).")
    st.stop()

data = pd.read_csv(uploaded_file)
missing_cols = [c for c in feature_columns + [TARGET] if c not in data.columns]
if missing_cols:
    st.error(f"Uploaded CSV is missing required columns: {missing_cols}")
    st.stop()

X = data[feature_columns].apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())
y_true = data[TARGET]

tab_compare, tab_detail = st.tabs(["📊 Compare All Models", "🔍 Model Deep-Dive"])

with tab_compare:
    st.subheader("All 5 Models on Uploaded Test Data")
    rows = []
    for name in MODEL_FILES:
        _, _, metrics = run_model(name, X, y_true)
        rows.append({"Model": name, **metrics})
    compare_df = pd.DataFrame(rows).set_index("Model").round(4)

    best_model = compare_df["F1 Score"].idxmax()
    st.dataframe(
        compare_df.style.highlight_max(axis=0, color="#b7e4c7"),
        use_container_width=True,
    )
    st.success(f"🏆 Best F1 Score on this test set: **{best_model}**")

    st.bar_chart(compare_df[["Accuracy", "AUC", "F1 Score", "MCC"]])

with tab_detail:
    model_name = st.selectbox("Choose a model to inspect", list(MODEL_FILES.keys()))
    y_pred, y_proba, metrics = run_model(model_name, X, y_true)

    st.subheader(f"{model_name} — Evaluation Metrics")
    metric_cols = st.columns(len(metrics))
    for col, (label, value) in zip(metric_cols, metrics.items()):
        col.metric(label, f"{value:.3f}")

    col_cm, col_report = st.columns(2)

    with col_cm:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig, use_container_width=True)

    with col_report:
        st.markdown("**Classification Report**")
        report_dict = classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
        )
        report_df = pd.DataFrame(report_dict).transpose().round(3)
        report_df = report_df.rename(columns={"support": "Support"})
        st.dataframe(
            report_df.style.background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"]),
            use_container_width=True,
        )
