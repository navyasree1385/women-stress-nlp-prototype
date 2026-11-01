
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Load the trained Random Forest pipeline
rf_model = joblib.load(MODEL_DIR / "tfidf_random_forest.joblib")

# Load the fine-tuned DistilBERT model saved earlier
transformer_dir = MODEL_DIR / "distilbert_stress"
tokenizer = AutoTokenizer.from_pretrained(
    str(transformer_dir), local_files_only=True
)
bert_model = AutoModelForSequenceClassification.from_pretrained(
    str(transformer_dir), local_files_only=True
)
bert_model.eval()

# Load validation data only. Do not use the test set here.
val_df = pd.read_csv(DATA_DIR / "validation.csv")
val_df = val_df.dropna(subset=["text", "label"]).copy()
val_df["text"] = val_df["text"].astype(str).str.strip()
val_df = val_df[val_df["text"] != ""].reset_index(drop=True)

texts = val_df["text"].tolist()
y_true = val_df["label"].astype(int).to_numpy()

# Random Forest probability for class 1
rf_class_index = list(rf_model.classes_).index(1)
rf_probs = rf_model.predict_proba(texts)[:, rf_class_index]

# DistilBERT probability for class 1
bert_probs = []

batch_size = 8

with torch.no_grad():
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=96,
        )

        logits = bert_model(**inputs).logits
        batch_probabilities = torch.softmax(logits, dim=1)[:, 1]
        bert_probs.extend(batch_probabilities.cpu().numpy())

bert_probs = np.asarray(bert_probs)

# Equal-weight probability fusion
hybrid_probs = 0.5 * rf_probs + 0.5 * bert_probs
hybrid_predictions = (hybrid_probs >= 0.5).astype(int)

print("\n===== HYBRID FUSION: VALIDATION RESULTS =====")
print(f"Validation samples: {len(y_true)}")
print(f"Accuracy : {accuracy_score(y_true, hybrid_predictions):.4f}")
print(
    f"Precision: "
    f"{precision_score(y_true, hybrid_predictions, zero_division=0):.4f}"
)
print(
    f"Recall   : "
    f"{recall_score(y_true, hybrid_predictions, zero_division=0):.4f}"
)
print(
    f"F1-score : "
    f"{f1_score(y_true, hybrid_predictions, zero_division=0):.4f}"
)
print(f"ROC-AUC  : {roc_auc_score(y_true, hybrid_probs):.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_true, hybrid_predictions))

print("\nClassification report:")
print(classification_report(y_true, hybrid_predictions, zero_division=0))
