
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

# Load the same validation set used in earlier experiments
df = pd.read_csv(DATA / "validation.csv")
df = df.dropna(subset=["text", "label"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].reset_index(drop=True)

texts = df["text"].tolist()
y_true = df["label"].astype(int).to_numpy()

# Load trained Logistic Regression pipeline
lr_model = joblib.load(
    MODELS / "tfidf_logistic_regression.joblib"
)

# Get the probability for class 1 (stress-positive)
lr_class_index = list(lr_model.classes_).index(1)
lr_probs = lr_model.predict_proba(texts)[:, lr_class_index]

# Load the fine-tuned DistilBERT model
bert_dir = MODELS / "distilbert_stress"

tokenizer = AutoTokenizer.from_pretrained(
    str(bert_dir), local_files_only=True
)
bert_model = AutoModelForSequenceClassification.from_pretrained(
    str(bert_dir), local_files_only=True
)
bert_model.eval()

bert_probs = []

with torch.no_grad():
    for start in range(0, len(texts), 8):
        batch_texts = texts[start:start + 8]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=96,
        )

        logits = bert_model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        bert_probs.extend(probabilities.cpu().numpy())

bert_probs = np.asarray(bert_probs)

# Compare several predefined fusion weights
weights = [0.25, 0.50, 0.75]
results = []

for lr_weight in weights:
    bert_weight = 1 - lr_weight
    hybrid_probs = (
        lr_weight * lr_probs + bert_weight * bert_probs
    )
    predictions = (hybrid_probs >= 0.5).astype(int)

    results.append({
        "LR_weight": lr_weight,
        "DistilBERT_weight": bert_weight,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1_score": f1_score(
            y_true, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_true, hybrid_probs),
        "Confusion_matrix": confusion_matrix(
            y_true, predictions
        ).tolist(),
    })

results_df = pd.DataFrame(results)

print("\n===== LOGISTIC REGRESSION + DISTILBERT FUSION =====")
print(results_df.to_string(index=False))

output_path = MODELS / "lr_distilbert_fusion_validation.csv"
results_df.to_csv(output_path, index=False)
print(f"\nSaved results to: {output_path}")
