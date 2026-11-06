"""
ML Assignment 2 - Model training script
Dataset: Cervical Cancer Risk Factors (UCI)
Trains 5 classifiers, saves models + scaler, writes test_data.csv and metrics.csv
"""
import pandas as pd
import numpy as np
import joblib
import urllib.request
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

DATA_PATH = Path("data/cervical_cancer_raw.csv")
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv"
MODEL_DIR = Path("model")
TARGET = "Biopsy"
LEAKAGE_COLS = ["Hinselmann", "Schiller", "Citology"]  # other screening results - drop to avoid leakage

MODEL_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    print(f"{DATA_PATH} not found locally - downloading from UCI...")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print(f"Downloaded to {DATA_PATH}")

df = pd.read_csv(DATA_PATH, na_values="?")

X = df.drop(columns=[TARGET] + LEAKAGE_COLS)
y = df[TARGET]

X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())

feature_columns = list(X.columns)
joblib.dump(feature_columns, MODEL_DIR / "feature_columns.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

test_df = X_test.copy()
test_df[TARGET] = y_test.values
test_df.to_csv("test_data.csv", index=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), True),
    "Decision Tree": (DecisionTreeClassifier(random_state=42), False),
    "KNN": (KNeighborsClassifier(n_neighbors=5), True),
    "Naive Bayes": (GaussianNB(), False),
    "Support Vector Machine": (SVC(probability=True, random_state=42), True),
}

results = []

for name, (model, needs_scaling) in models.items():
    if needs_scaling:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    results.append(metrics)

    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, MODEL_DIR / filename)
    print(f"Trained & saved: {name} -> model/{filename}")

results_df = pd.DataFrame(results)
results_df.to_csv("metrics.csv", index=False)
print("\n=== Evaluation Metrics ===")
print(results_df.round(4).to_string(index=False))
print("\nSaved: test_data.csv, metrics.csv, model/*.pkl, model/scaler.pkl, model/feature_columns.pkl")
