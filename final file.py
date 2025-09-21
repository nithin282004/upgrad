# Component 4: Running ML Models & Plotting ROC Curves
# Mentor-Led Internship: Employee Performance Prediction
# Name: Nithin Aparadapu
# Date: 21-Sep-2025

# -----------------------------
# Step 0: Import Libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
data = pd.read_csv('employee_data_cleaned.csv')

# Binary target for performance
data['PerformanceBinary'] = (data['PerformanceRating'] >= 4).astype(int)

# Separate features and target
X = data.drop(['PerformanceRating', 'PerformanceBinary'], axis=1)
y = data['PerformanceBinary']

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Step 2: Scaling Function
# -----------------------------
def scale_features(X_train, X_test, scaler_name="standard"):
    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "minmax":
        scaler = MinMaxScaler()
    elif scaler_name == "robust":
        scaler = RobustScaler()
    elif scaler_name == "maxabs":
        scaler = MaxAbsScaler()
    else:
        raise ValueError("Invalid scaler_name")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# -----------------------------
# Step 3: Initialize Models
# -----------------------------
models = {
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "LightGBM": lgb.LGBMClassifier(random_state=42)
}

scalers = ["standard", "minmax", "robust", "maxabs"]

# -----------------------------
# Step 4: Train, Evaluate & Plot ROC
# -----------------------------
results = []

plt.figure(figsize=(10,8))  # ROC plot

for scaler_name in scalers:
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test, scaler_name)
    for model_name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_prob = model.predict_proba(X_test_scaled)[:,1]
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)
        cm = confusion_matrix(y_test, y_pred)
        results.append({
            "Scaler": scaler_name,
            "Model": model_name,
            "Accuracy": acc,
            "ROC_AUC": auc,
            "ConfusionMatrix": cm
        })
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        plt.plot(fpr, tpr, label=f'{model_name}-{scaler_name} (AUC={auc:.2f})')

# ROC plot formatting
plt.plot([0,1], [0,1], 'k--')
plt.title("ROC Curves for Employee Performance Prediction")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# -----------------------------
# Step 5: Display Results
# -----------------------------
for res in results:
    print(f"Scaler: {res['Scaler']}, Model: {res['Model']}")
    print(f"Accuracy: {res['Accuracy']:.4f}, ROC-AUC: {res['ROC_AUC']:.4f}")
    print("Confusion Matrix:")
    print(res['ConfusionMatrix'])
    print("-"*50)

# -----------------------------
# Step 6: Best Model Selection
# -----------------------------
best_result = max(results, key=lambda x: x['ROC_AUC'])
print("\nBest Model & Scaler Combination:")
print(best_result)
