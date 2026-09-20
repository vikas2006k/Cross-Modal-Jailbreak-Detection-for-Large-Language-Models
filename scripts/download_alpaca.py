import os
from datasets import load_dataset

# Destination directory inside dataset/
output_dir = "dataset/external/alpaca"
os.makedirs(output_dir, exist_ok=True)

print("Downloading the Alpaca dataset from Hugging Face...")
ds = load_dataset("tatsu-lab/alpaca")

# Save as Parquet (recommended)
parquet_path = os.path.join(output_dir, "train.parquet")
ds["train"].to_parquet(parquet_path)

print("Downloaded successfully!")
print(f"Saved to: {parquet_path}")
print(ds["train"])
