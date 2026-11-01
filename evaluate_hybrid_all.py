
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
)

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
DATA = BASE / "data"

# Load validation data only
df = pd.read_csv(DATA / "validation.csv")
df = df.dropna(subset=["text", "label"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].reset_index(drop=True)

texts = df["text"].tolist()
y_true = df["label"].astype(int).to_numpy()

# Load Logistic Regression pipeline
lr_model = joblib.load(
    MODELS / "tfidf_logistic_regression.joblib"
)
lr_class_index = list(lr_model.classes_).index(1)
lr_probs = lr_model.predict_proba(texts)[:, lr_class_index]


def get_transformer_probabilities(model_folder, model_name):
    folder = MODELS / model_folder

    tokenizer = AutoTokenizer.from_pretrained(
        str(folder), local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(folder), local_files_only=True
    )
    model.eval()

    probabilities = []

    with torch.no_grad():
        for start in range(0, len(texts), 4):
            batch_texts = texts[start:start + 4]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=96 if model_name == "DistilBERT" else 64,
            )

            logits = model(**inputs).logits
            batch_probs = torch.softmax(logits, dim=1)[:, 1]
            probabilities.extend(batch_probs.cpu().numpy())

    return np.asarray(probabilities)


print("Loading DistilBERT...")
distil_probs = get_transformer_probabilities(
    "distilbert_stress", "DistilBERT"
)

print("Loading BERT...")
bert_probs = get_transformer_probabilities(
    "bert_stress", "BERT"
)

# Candidate combinations
experiments = {
    "LR + DistilBERT (50:50)": (
        0.50 * lr_probs + 0.50 * distil_probs
    ),
    "LR + BERT (50:50)": (
        0.50 * lr_probs + 0.50 * bert_probs
    ),
    "DistilBERT + BERT (50:50)": (
        0.50 * distil_probs + 0.50 * bert_probs
    ),
    "LR + DistilBERT + BERT (equal)": (
        (lr_probs + distil_probs + bert_probs) / 3
    ),
}

results = []

for name, probabilities in experiments.items():
    predictions = (probabilities >= 0.5).astype(int)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_true, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "Confusion matrix": confusion_matrix(
            y_true, predictions
        ).tolist(),
    })

results_df = pd.DataFrame(results)

print("\n===== HYBRID VALIDATION COMPARISON =====")
print(results_df.to_string(index=False))

output = MODELS / "hybrid_all_validation.csv"
results_df.to_csv(output, index=False)
print(f"\nSaved results to: {output}")
