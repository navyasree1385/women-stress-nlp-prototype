
from pathlib import Path
import json
import time

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# Configuration: CPU-friendly BERT experiment
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
SAVE_DIR = MODEL_DIR / "bert_stress"

MODEL_NAME = "bert-base-uncased"
TRAIN_LIMIT = 300
MAX_LENGTH = 64
BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 2e-5
SEED = 42

torch.manual_seed(SEED)
device = torch.device("cpu")
torch.set_num_threads(2)

print(f"Device: {device}")
print(f"Model: {MODEL_NAME}")
print("Loading dataset...")

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")

for name, df in [("train", train_df), ("validation", val_df)]:
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError(f"{name}.csv must contain text and label columns")

    df.dropna(subset=["text", "label"], inplace=True)
    df["text"] = df["text"].astype(str).str.strip()
    df.drop(df[df["text"] == ""].index, inplace=True)
    df["label"] = df["label"].astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{name}.csv must use labels 0 and 1")

# Use a reproducible, stratified subset for the first CPU run.
if len(train_df) > TRAIN_LIMIT:
    train_df, _ = train_test_split(
        train_df,
        train_size=TRAIN_LIMIT,
        random_state=SEED,
        stratify=train_df["label"],
    )

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print(f"Training samples used: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print("\nLoading tokenizer and BERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)
model.to(device)


class StressDataset(Dataset):
    def __init__(self, frame):
        self.texts = frame["text"].tolist()
        self.labels = frame["label"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoded.items()
        }
        item["labels"] = torch.tensor(
            self.labels[index], dtype=torch.long
        )
        return item


train_loader = DataLoader(
    StressDataset(train_df),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    StressDataset(val_df),
    batch_size=4,
    shuffle=False,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)

# --------------------------------------------------
# Fine-tuning
# --------------------------------------------------
print("\nStarting BERT fine-tuning...")
start_time = time.time()
model.train()

for epoch in range(EPOCHS):
    total_loss = 0.0

    for step, batch in enumerate(train_loader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item()

        if step % 25 == 0 or step == len(train_loader):
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {step}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    print(
        f"Average training loss: "
        f"{total_loss / len(train_loader):.4f}"
    )

# --------------------------------------------------
# Validation evaluation
# --------------------------------------------------
print("\nEvaluating on validation data...")
model.eval()

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():
    for batch in val_loader:
        labels = batch["labels"].numpy()
        inputs = {
            key: value.to(device)
            for key, value in batch.items()
            if key != "labels"
        }

        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        predictions = torch.argmax(logits, dim=1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_probabilities.extend(probabilities.cpu().numpy().tolist())

accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(
    all_labels, all_predictions, zero_division=0
)
recall = recall_score(
    all_labels, all_predictions, zero_division=0
)
f1 = f1_score(all_labels, all_predictions, zero_division=0)
roc_auc = roc_auc_score(all_labels, all_probabilities)

results = {
    "model": MODEL_NAME,
    "training_samples": len(train_df),
    "validation_samples": len(val_df),
    "epochs": EPOCHS,
    "max_length": MAX_LENGTH,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "confusion_matrix": confusion_matrix(
        all_labels, all_predictions
    ).tolist(),
    "classification_report": classification_report(
        all_labels,
        all_predictions,
        zero_division=0,
        output_dict=True,
    ),
    "training_seconds": round(time.time() - start_time, 2),
}

# Save the fine-tuned model and tokenizer for later use.
SAVE_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

report_path = MODEL_DIR / "bert_validation_report.json"
report_path.write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)

print("\n===== BERT VALIDATION RESULTS =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print("\nConfusion matrix:")
print(confusion_matrix(all_labels, all_predictions))
print("\nClassification report:")
print(classification_report(
    all_labels, all_predictions, zero_division=0
))
print(f"\nSaved model to: {SAVE_DIR}")
print(f"Saved metrics to: {report_path}")
print(f"Training and evaluation time: {results['training_seconds']} seconds")
