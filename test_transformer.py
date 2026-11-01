
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "distilbert-base-uncased"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading pretrained DistilBERT...")
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

text = "I have been feeling overwhelmed by work and responsibilities."

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=128
)

with torch.no_grad():
    outputs = model(**inputs)

print("\nTransformer loaded successfully!")
print("Input tokens:", inputs["input_ids"].shape[1])
print("Output shape:", outputs.last_hidden_state.shape)
print("This is a text representation, not a stress prediction.")
