
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Load the existing dataset splits
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")

# Basic data validation
required_columns = {"text", "label"}
for name, df in [("train", train_df), ("validation", val_df)]:
    if not required_columns.issubset(df.columns):
        raise ValueError(f"{name}.csv must contain: {required_columns}")
    if df["text"].isna().any() or df["label"].isna().any():
        raise ValueError(f"{name}.csv contains missing text or labels")

X_train = train_df["text"].astype(str)
y_train = train_df["label"].astype(int)
X_val = val_df["text"].astype(str)
y_val = val_df["label"].astype(int)

# TF-IDF + Random Forest pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            stop_words="english",
        ),
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    ),
])

print("Training TF-IDF + Random Forest...")
model.fit(X_train, y_train)

# Evaluate on validation data only
predictions = model.predict(X_val)
probabilities = model.predict_proba(X_val)

# Find the probability column corresponding to label 1
positive_index = list(model.named_steps["classifier"].classes_).index(1)
positive_probabilities = probabilities[:, positive_index]

metrics = {
    "model": "TF-IDF + Random Forest",
    "train_samples": len(train_df),
    "validation_samples": len(val_df),
    "accuracy": float(accuracy_score(y_val, predictions)),
    "precision": float(
        precision_score(y_val, predictions, zero_division=0)
    ),
    "recall": float(recall_score(y_val, predictions, zero_division=0)),
    "f1_score": float(f1_score(y_val, predictions, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_val, positive_probabilities)),
    "confusion_matrix": confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).tolist(),
}

print("\n=== Validation Results ===")
for metric, value in metrics.items():
    print(f"{metric}: {value}")

print("\n=== Classification Report ===")
print(classification_report(y_val, predictions, zero_division=0))

# Save the fitted pipeline and validation results
joblib.dump(model, MODEL_DIR / "tfidf_random_forest.joblib")

with open(
    MODEL_DIR / "random_forest_validation_report.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metrics, file, indent=4)

print("\nSaved model: models/tfidf_random_forest.joblib")
print("Saved metrics: models/random_forest_validation_report.json")
