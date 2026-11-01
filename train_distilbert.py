
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Conservative settings for a CPU-only laptop
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 96
BATCH_SIZE = 2
EPOCHS = 1
MAX_TRAIN_SAMPLES = 300

DATA_DIR = Path("data")
MODEL_DIR = Path("models/distilbert_stress")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)

train_df = pd.read_csv(DATA_DIR / "train.csv").dropna(subset=["text", "label"])
val_df = pd.read_csv(DATA_DIR / "validation.csv").dropna(subset=["text", "label"])

# Use a small stratified subset for the first feasibility run
train_df = (
    train_df.groupby("label", group_keys=False)
    .sample(n=min(MAX_TRAIN_SAMPLES // 2, train_df["label"].value_counts().min()),
            random_state=42)
    .reset_index(drop=True)
)

class StressDataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.texts = frame["text"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)
model.config.id2label = {0: "STRESS_NEGATIVE", 1: "STRESS_POSITIVE"}
model.config.label2id = {"STRESS_NEGATIVE": 0, "STRESS_POSITIVE": 1}

train_loader = DataLoader(
    StressDataset(train_df, tokenizer),
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_loader = DataLoader(
    StressDataset(val_df, tokenizer),
    batch_size=4,
    shuffle=False
)

optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

print(f"Training on {len(train_df)} examples for {EPOCHS} epoch...")
model.train()

for epoch in range(EPOCHS):
    running_loss = 0.0
    for step, batch in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        batch = {key: value for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()

        if step % 10 == 0:
            print(f"Step {step}/{len(train_loader)} | loss={loss.item():.4f}")

    print(f"Epoch {epoch + 1} average loss: {running_loss / len(train_loader):.4f}")

print("Evaluating on validation set...")
model.eval()
all_labels, all_preds = [], []

with torch.no_grad():
    for batch in val_loader:
        labels = batch.pop("labels")
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.numpy().tolist())

accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="binary", zero_division=0
)

print(f"Validation accuracy:  {accuracy:.4f}")
print(f"Validation precision: {precision:.4f}")
print(f"Validation recall:    {recall:.4f}")
print(f"Validation F1-score:   {f1:.4f}")

model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"Model saved to {MODEL_DIR}")
