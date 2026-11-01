
import transformers
from transformers import AutoModelForSequenceClassification

print("Transformers version:", transformers.__version__)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

print("Classification model loaded.")
print("Labels:", model.config.num_labels)
