
from pathlib import Path
import json

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

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELS = BASE / "models"
OUTPUT = MODELS / "final_test_results.json"

# Load test data ONLY for final evaluation
df = pd.read_csv(DATA / "test.csv")
df = df.dropna(subset=["text", "label"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].reset_index(drop=True)

texts = df["text"].tolist()
y_true = df["label"].astype(int).to_numpy()

print(f"Test samples: {len(y_true)}")
print("Loading trained models...")


def get_positive_probs(model, texts):
    class_index = list(model.classes_).index(1)
    return model.predict_proba(texts)[:, class_index]


# 1. Logistic Regression
lr_model = joblib.load(
    MODELS / "tfidf_logistic_regression.joblib"
)
lr_probs = get_positive_probs(lr_model, texts)

# 2. Random Forest
rf_model = joblib.load(
    MODELS / "tfidf_random_forest.joblib"
)
rf_probs = get_positive_probs(rf_model, texts)


def transformer_probs(folder_name, max_length, batch_size=4):
    folder = MODELS / folder_name

    tokenizer = AutoTokenizer.from_pretrained(
        str(folder), local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(folder), local_files_only=True
    )
    model.eval()

    probabilities = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[:, 1]
            probabilities.extend(probs.cpu().numpy())

    return np.asarray(probabilities)


# 3. DistilBERT
print("Evaluating DistilBERT...")
distil_probs = transformer_probs("distilbert_stress", 96)

# 4. BERT
print("Evaluating BERT...")
bert_probs = transformer_probs("bert_stress", 64)

# 5. Best validation hybrid: LR + DistilBERT, equal weights
hybrid_probs = 0.5 * lr_probs + 0.5 * distil_probs

models = {
    "TF-IDF + Logistic Regression": lr_probs,
    "TF-IDF + Random Forest": rf_probs,
    "DistilBERT": distil_probs,
    "BERT": bert_probs,
    "Hybrid LR + DistilBERT (50:50)": hybrid_probs,
}

results = {}

for name, probs in models.items():
    predictions = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1_score": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "confusion_matrix": confusion_matrix(
            y_true, predictions
        ).tolist(),
        "classification_report": classification_report(
            y_true, predictions, zero_division=0, output_dict=True
        ),
    }

    results[name] = metrics

    print(f"\n===== {name} =====")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("Confusion matrix:")
    print(np.array(metrics["confusion_matrix"]))
    print(classification_report(
        y_true, predictions, zero_division=0
    ))

OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

summary = pd.DataFrame([
    {
        "Model": name,
        "Accuracy": m["accuracy"],
        "Precision": m["precision"],
        "Recall": m["recall"],
        "F1-score": m["f1_score"],
        "ROC-AUC": m["roc_auc"],
    }
    for name, m in results.items()
])

summary.to_csv(MODELS / "final_test_comparison.csv", index=False)

print("\n===== FINAL TEST COMPARISON =====")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nSaved detailed results: {OUTPUT}")
print(f"Saved comparison table: {MODELS / 'final_test_comparison.csv'}")
