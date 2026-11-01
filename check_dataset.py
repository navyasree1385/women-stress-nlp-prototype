
from pathlib import Path
import pandas as pd

for name in ["train.csv", "validation.csv", "test.csv"]:
    path = Path("data") / name
    df = pd.read_csv(path)

    print(f"\n--- {name} ---")
    print("Rows and columns:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df[["text", "label"]].head(2))
    print("Labels:\n", df["label"].value_counts().sort_index())
