# Health Risk Scoring Model

A machine learning project that predicts a patient's health risk level from clinical and lifestyle data.

The goal is to build a transparent, explainable model that could be used by healthcare providers, insurers, or wellness platforms to identify high-risk individuals and support early interventions.

---

## 1. Problem Statement

Many healthcare systems struggle to proactively identify patients at high risk of developing
serious conditions. This project explores how tabular clinical data (age, blood pressure,
cholesterol, lifestyle factors, etc.) can be used to build a **health risk scoring model**
that classifies patients into risk segments (e.g. Low, Medium, High).

---

## 2. Objectives

- Clean and preprocess a real-world-style health dataset.
- Engineer features suitable for supervised learning.
- Train baseline and advanced models (e.g. Logistic Regression, Tree-based models).
- Evaluate performance with metrics such as ROC-AUC, precision/recall, and calibration.
- Derive interpretable risk scores and visual explanations (e.g. feature importance, SHAP).
- Optionally expose a simple **Streamlit app** to interactively score new patients.

---

## 3. Dataset

- **Type:** Tabular clinical / health records
- **Target:** Binary or multi-class health risk label (e.g. high_risk = 1/0 or Low/Med/High)
- **Features (examples):** Age, sex, blood pressure, cholesterol, smoking, BMI, etc.

> ⚠️ Note: The dataset used here is for educational and demonstration purposes only and should
> not be used for real medical decision-making.

(You can update this section with the exact dataset name and citation once chosen.)

---

## 4. Methodology

1. **Exploratory Data Analysis (EDA)**
   - Inspect distributions, missing values, outliers.
   - Analyze correlations between features and the target.

2. **Preprocessing & Feature Engineering**
   - Handle missing values, encode categorical variables, scale numerical features.
   - Create derived features if useful (e.g., BMI categories, risk factor counts).

3. **Modeling**
   - Baseline: DummyClassifier / simple Logistic Regression.
   - Main models: Regularized Logistic Regression, Random Forest, Gradient Boosting, etc.
   - Hyperparameter tuning using cross-validation.

4. **Evaluation**
   - Metrics: ROC-AUC, accuracy, precision, recall, F1, confusion matrix.
   - Calibration: reliability curve, predicted vs actual risk.
   - Error analysis: Where does the model fail?

5. **Explainability**
   - Global feature importance.
   - Local explanations (e.g. SHAP values) for individual patients.

6. **(Optional) Deployment Demo**
   - Simple **Streamlit app** to enter patient features and see risk prediction + explanation.

---

## 5. Tech Stack

- Python 3.x
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- shap (for explainability)
- streamlit (for the interactive app)

---

## 6. How to Run

```bash
# 1. Create environment & install dependencies
pip install -r requirements.txt

# 2. Run notebooks for EDA & modeling
jupyter notebook notebooks/analysis.ipynb
jupyter notebook notebooks/modeling.ipynb

# 3. (Optional) Launch Streamlit app
streamlit run app/streamlit_app.py
