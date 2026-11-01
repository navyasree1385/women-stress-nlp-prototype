
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

# Validate data
for name, df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df)
]:
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError(f"{name}.csv must contain text and label columns")

    df.dropna(subset=["text", "label"], inplace=True)
    df["text"] = df["text"].astype(str)

# TF-IDF + Logistic Regression baseline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

print("Training baseline model...")
model.fit(train_df["text"], train_df["label"])

# Use validation data for an initial check
val_predictions = model.predict(val_df["text"])

print("\nVALIDATION RESULTS")
print("Accuracy:", round(accuracy_score(
    val_df["label"], val_predictions
), 4))
print("Precision:", round(precision_score(
    val_df["label"], val_predictions, zero_division=0
), 4))
print("Recall:", round(recall_score(
    val_df["label"], val_predictions, zero_division=0
), 4))
print("F1-score:", round(f1_score(
    val_df["label"], val_predictions, zero_division=0
), 4))

# Save the trained model
joblib.dump(model, MODEL_DIR / "tfidf_logistic_regression.joblib")

# Save validation report
report = classification_report(
    val_df["label"],
    val_predictions,
    output_dict=True,
    zero_division=0
)

with open(MODEL_DIR / "baseline_validation_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\nModel saved to models/tfidf_logistic_regression.joblib")
print("Validation report saved.")
print("\nConfusion matrix (validation):")
print(confusion_matrix(val_df["label"], val_predictions))
print("\nBaseline training complete.")
