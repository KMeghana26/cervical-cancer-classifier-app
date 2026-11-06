# Breast Cancer Risk Classification

## a. Problem Statement
This project predicts whether a breast tumor is malignant (cancerous) or benign based on
cell-nuclei measurements taken from a digitized image of a fine needle aspirate (FNA),
using five classical machine learning classifiers.

## b. Dataset Description
- **Source:** Breast Cancer Wisconsin (Diagnostic) Data Set, loaded via
  `sklearn.datasets.load_breast_cancer()` (originally from UCI)
- **Instances:** 569
- **Features:** 30 (all numeric - mean/standard-error/worst-case of 10 cell-nuclei measurements
  such as radius, texture, perimeter, area, smoothness)
- **Target:** `Diagnosis` (0 = No Cancer / Benign, 1 = Cancer / Malignant)
- **Class balance:** 357 Benign / 212 Malignant (~63% / 37%) - a realistic, moderate imbalance,
  not the severe imbalance (~6% positive) seen in the Cervical Cancer dataset originally
  considered for this assignment.
- **Preprocessing:** No missing values in this dataset. Features scaled with `StandardScaler`
  for Logistic Regression, KNN, and SVM (not needed for Decision Tree / Naive Bayes).

## c. GitHub Repository Link
https://github.com/KMeghana26/cervical-cancer-classifier-app

## d. Models Used

Comparison Table (fill in after running `python train_models.py` - values come from `metrics.csv`):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Support Vector Machine | |
| **Overall, Winner for your dataset?** | |

## How to Run

```bash
pip install -r requirements.txt
python train_models.py      # trains all 5 models, saves to model/, writes test_data.csv + metrics.csv
streamlit run app.py         # launches the interactive app
```

## Project Structure
```
project-folder/
  app.py
  train_models.py
  requirements.txt
  README.md
  test_data.csv
  data/
    breast_cancer_raw.csv
  model/
    logistic_regression.pkl
    decision_tree.pkl
    knn.pkl
    naive_bayes.pkl
    support_vector_machine.pkl
    scaler.pkl
    feature_columns.pkl
```
