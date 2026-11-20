# ==============================
# KSHETRA - CROP STRESS MODEL
# Random Forest + GridSearchCV
# ==============================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================
# 1. INPUT FEATURES
# ==============================

TEXT_INPUTS = [
    "crop_type",
    "soil_type",
    "growth_stage",
    "irrigation_source"
]

NUMBER_INPUTS = [
    "days_since_last_irrigation",
    "past_stress_events",
    "soil_moisture_pct",
    "temperature_c",
    "humidity_pct",
    "rainfall_last_7d_mm",
    "rainfall_forecast_3d_mm"
]

FEATURES = TEXT_INPUTS + NUMBER_INPUTS
TARGET = "stress_risk_level"

# ==============================
# 2. LOAD DATA
# ==============================

df = pd.read_csv(os.path.join(BASE_DIR, "crop_stress_dataset.csv"))

X = df[FEATURES]
y = df[TARGET]

print("Dataset loaded")
print("Rows:", len(df))
print("Features:", len(FEATURES))
print("\nStress distribution:")
print(y.value_counts())

# ==============================
# 3. TRAIN-TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==============================
# 4. PREPROCESSING
# ==============================

text_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

number_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer([
    ("text", text_pipeline, TEXT_INPUTS),
    ("number", number_pipeline, NUMBER_INPUTS)
])

# ==============================
# 5. RANDOM FOREST PIPELINE
# ==============================

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# ==============================
# 6. GRID SEARCH TUNING
# ==============================

param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [10, 20, None],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2],
    "classifier__max_features": ["sqrt"]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1,
    verbose=1
)

print("\nStarting hyperparameter tuning...")

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest CV Macro F1:")
print(round(grid_search.best_score_, 4))

# ==============================
# 7. TEST PREDICTIONS
# ==============================

y_pred = best_model.predict(X_test)

# ==============================
# 8. EVALUATION METRICS
# ==============================

accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

precision_macro = precision_score(
    y_test, y_pred, average="macro", zero_division=0
)

recall_macro = recall_score(
    y_test, y_pred, average="macro", zero_division=0
)

f1_macro = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)

precision_weighted = precision_score(
    y_test, y_pred, average="weighted", zero_division=0
)

recall_weighted = recall_score(
    y_test, y_pred, average="weighted", zero_division=0
)

f1_weighted = f1_score(
    y_test, y_pred, average="weighted", zero_division=0
)

# ==============================
# 9. PRINT RESULTS
# ==============================

print("\n========== KSHETRA RESULTS ==========")

print(f"Accuracy            : {accuracy:.4f}")
print(f"Balanced Accuracy   : {balanced_accuracy:.4f}")
print(f"Macro Precision    : {precision_macro:.4f}")
print(f"Macro Recall       : {recall_macro:.4f}")
print(f"Macro F1-score    : {f1_macro:.4f}")
print(f"Weighted Precision : {precision_weighted:.4f}")
print(f"Weighted Recall    : {recall_weighted:.4f}")
print(f"Weighted F1-score  : {f1_weighted:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ==============================
# 10. CONFUSION MATRIX
# ==============================

cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)

plt.figure(figsize=(7, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Crop Stress Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(range(len(best_model.classes_)), best_model.classes_)
plt.yticks(range(len(best_model.classes_)), best_model.classes_)

for i in range(len(cm)):
    for j in range(len(cm[i])):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"))
plt.show()

# ==============================
# 11. FEATURE IMPORTANCE
# ==============================

classifier = best_model.named_steps["classifier"]
preprocessor_fitted = best_model.named_steps["preprocessing"]

feature_names = preprocessor_fitted.get_feature_names_out()
importances = classifier.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nTop 10 Important Features:")
print(importance_df.head(10))

plt.figure(figsize=(10, 6))

top_features = importance_df.head(10).sort_values("importance")

plt.barh(top_features["feature"], top_features["importance"])
plt.title("Top 10 Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ==============================
# 12. SAVE ONE PICKLE FILE
# ==============================

# Train final model using entire dataset
best_model.fit(X, y)

joblib.dump({
    "model": best_model,
    "features": FEATURES,
    "classes": list(best_model.classes_),
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "macro_precision": precision_macro,
    "macro_recall": recall_macro,
    "macro_f1": f1_macro,
    "weighted_f1": f1_weighted,
    "best_params": grid_search.best_params_
},  os.path.join(BASE_DIR, "crop_stress_model.pkl"))

print("\n====================================")
print("Training completed successfully!")
print("Saved file: crop_stress_model.pkl")
print("====================================")