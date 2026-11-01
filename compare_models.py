
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
)

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
DATA = BASE / "data"

# Load the same validation split for every model
df = pd.read_csv(DATA / "validation.csv")
df = df.dropna(subset=["text", "label"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].reset_index(drop=True)

texts = df["text"].tolist()
y = df["label"].astype(int).to_numpy()

# Load both traditional ML models
lr = joblib.load(MODELS / "tfidf_logistic_regression.joblib")
rf = joblib.load(MODELS / "tfidf_random_forest.joblib")

def positive_probability(model, texts):
    class_index = list(model.classes_).index(1)
    return model.predict_proba(texts)[:, class_index]

lr_probs = positive_probability(lr, texts)
rf_probs = positive_probability(rf, texts)

# Load DistilBERT
model_dir = MODELS / "distilbert_stress"
tokenizer = AutoTokenizer.from_pretrained(
    str(model_dir), local_files_only=True
)
bert = AutoModelForSequenceClassification.from_pretrained(
    str(model_dir), local_files_only=True
)
bert.eval()

bert_probs = []

with torch.no_grad():
    for start in range(0, len(texts), 8):
        batch = tokenizer(
            texts[start:start + 8],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=96,
        )
        logits = bert(**batch).logits
        probs = torch.softmax(logits, dim=1)[:, 1]
        bert_probs.extend(probs.cpu().numpy())

bert_probs = np.asarray(bert_probs)

# Equal-weight hybrid fusion
hybrid_probs = 0.5 * rf_probs + 0.5 * bert_probs

models = {
    "TF-IDF + Logistic Regression": lr_probs,
    "TF-IDF + Random Forest": rf_probs,
    "DistilBERT": bert_probs,
    "Hybrid RF + DistilBERT": hybrid_probs,
}

rows = []

for name, probs in models.items():
    predictions = (probs >= 0.5).astype(int)

    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y, probs),
    })

results = pd.DataFrame(rows)
print("\n===== VALIDATION MODEL COMPARISON =====")
print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

results.to_csv(MODELS / "validation_model_comparison.csv", index=False)
print("\nSaved: models/validation_model_comparison.csv")
