import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    classification_report,
    roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ==========================
# Create Required Folders
# ==========================

os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================
# EDA
# ==========================

plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title("Customer Churn Distribution")
plt.savefig("images/churn_distribution.png")
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Contract Type vs Churn")
plt.xticks(rotation=20)
plt.savefig("images/contract_vs_churn.png")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.savefig("images/monthlycharges_vs_churn.png")
plt.show()

# ==========================
# Data Cleaning
# ==========================

df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df.dropna(inplace=True)

# ==========================
# Encode Categorical Columns
# ==========================

df = pd.get_dummies(df, drop_first=True)

print("\nColumns after encoding:")
print(df.columns)

# ==========================
# Features and Target
# ==========================

X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================
# Feature Scaling
# ==========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================
# Logistic Regression
# ==========================

print("\n========== Logistic Regression ==========")

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train_scaled, y_train)

pred_lr = lr.predict(X_test_scaled)
prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

print("Accuracy :", accuracy_score(y_test, pred_lr))
print("Recall   :", recall_score(y_test, pred_lr))
print("ROC AUC  :", roc_auc_score(y_test, prob_lr))

print("\nClassification Report:")
print(classification_report(y_test, pred_lr))

# ==========================
# Random Forest
# ==========================

print("\n========== Random Forest ==========")

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)
prob_rf = rf.predict_proba(X_test)[:, 1]

print("Accuracy :", accuracy_score(y_test, pred_rf))
print("Recall   :", recall_score(y_test, pred_rf))
print("ROC AUC  :", roc_auc_score(y_test, prob_rf))

print("\nClassification Report:")
print(classification_report(y_test, pred_rf))

# ==========================
# XGBoost
# ==========================

print("\n========== XGBoost ==========")

xgb = XGBClassifier(
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X_train, y_train)

pred_xgb = xgb.predict(X_test)
prob_xgb = xgb.predict_proba(X_test)[:, 1]

print("Accuracy :", accuracy_score(y_test, pred_xgb))
print("Recall   :", recall_score(y_test, pred_xgb))
print("ROC AUC  :", roc_auc_score(y_test, prob_xgb))

print("\nClassification Report:")
print(classification_report(y_test, pred_xgb))

# ==========================
# Model Comparison
# ==========================

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    "Accuracy": [
        accuracy_score(y_test, pred_lr),
        accuracy_score(y_test, pred_rf),
        accuracy_score(y_test, pred_xgb)
    ],
    "Recall": [
        recall_score(y_test, pred_lr),
        recall_score(y_test, pred_rf),
        recall_score(y_test, pred_xgb)
    ],
    "ROC_AUC": [
        roc_auc_score(y_test, prob_lr),
        roc_auc_score(y_test, prob_rf),
        roc_auc_score(y_test, prob_xgb)
    ]
})

print("\n========== Model Comparison ==========")
print(results)

# ==========================
# ROC Curve
# ==========================

plt.figure(figsize=(8,6))

fpr_lr, tpr_lr, _ = roc_curve(y_test, prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, prob_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, prob_xgb)

plt.plot(fpr_lr, tpr_lr, label="Logistic Regression")
plt.plot(fpr_rf, tpr_rf, label="Random Forest")
plt.plot(fpr_xgb, tpr_xgb, label="XGBoost")

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()

plt.savefig("images/roc_curve.png")
plt.show()

# ==========================
# Feature Importance
# ==========================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(importance.head(10))

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")
plt.savefig("images/feature_importance.png")
plt.show()

# ==========================
# Save Model
# ==========================

joblib.dump(
    rf,
    "models/random_forest.pkl"
)

print("\nRandom Forest model saved successfully!")