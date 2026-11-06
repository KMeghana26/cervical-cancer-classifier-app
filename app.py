"""
ML Assignment 2 - Streamlit app
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
TARGET = "Biopsy"

MODEL_FILES = {
    "Logistic Regression": ("logistic_regression.pkl", True),
    "Decision Tree": ("decision_tree.pkl", False),
    "KNN": ("knn.pkl", True),
    "Naive Bayes": ("naive_bayes.pkl", False),
    "Support Vector Machine": ("support_vector_machine.pkl", True),
}

st.set_page_config(page_title="Cervical Cancer Risk Classifier", layout="centered")
st.title("Cervical Cancer Risk Classification")
st.caption("Upload test data, pick a model, and view its evaluation results.")

feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

model_name = st.selectbox("Choose a model", list(MODEL_FILES.keys()))

uploaded_file = st.file_uploader("Upload test CSV (must include all feature columns + 'Biopsy' target)", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    missing_cols = [c for c in feature_columns + [TARGET] if c not in data.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required columns: {missing_cols}")
    else:
        X = data[feature_columns].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median())
        y_true = data[TARGET]

        filename, needs_scaling = MODEL_FILES[model_name]
        model = joblib.load(MODEL_DIR / filename)

        X_input = scaler.transform(X) if needs_scaling else X
        y_pred = model.predict(X_input)
        y_proba = model.predict_proba(X_input)[:, 1]

        st.subheader("Evaluation Metrics")
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_proba),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred),
        }
        st.table(pd.DataFrame(metrics, index=[model_name]).round(4))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Cancer", "Cancer"], yticklabels=["No Cancer", "Cancer"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.subheader("Classification Report")
        report = classification_report(y_true, y_pred, target_names=["No Cancer", "Cancer"])
        st.text(report)
else:
    st.info("Upload a test CSV to see results. You can use the provided test_data.csv.")
