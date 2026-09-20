import os
import re
import pandas as pd

def prepare_alpaca():
    input_path = "dataset/external/alpaca/train.parquet"
    out_clean_path = "dataset/processed/safe_prompts_clean.csv"
    out_safe_path = "dataset/text/safe/safe_prompts.csv"

    # Verify input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    # Read dataset
    df = pd.read_parquet(input_path)
    orig_count = len(df)

    # Keep only instruction and input
    df = df[["instruction", "input"]].copy()

    # Create prompt field
    def build_prompt(row):
        inst = str(row["instruction"]).strip() if pd.notna(row["instruction"]) else ""
        inp = str(row["input"]).strip() if pd.notna(row["input"]) else ""
        if not inp:
            return inst
        return f"{inst}\n\n{inp}"

    df["prompt"] = df.apply(build_prompt, axis=1)

    # Cleaning:
    # 1. Remove null/empty prompts
    df = df[df["prompt"].notna() & (df["prompt"].str.strip() != "")].copy()

    # 2. Remove duplicate prompts
    df = df.drop_duplicates(subset=["prompt"]).copy()

    # 3. Filter length: 15 <= len <= 500 characters
    prompt_len = df["prompt"].str.len()
    df = df[(prompt_len >= 15) & (prompt_len <= 500)].copy()

    # 4. Remove prompts containing only symbols or numbers (must contain at least one alphabetic character)
    has_letters = df["prompt"].apply(lambda p: bool(re.search(r"[a-zA-Z]", p)))
    df = df[has_letters].copy()

    cleaned_count = len(df)

    # Randomly sample exactly 5000 prompts using random seed = 42
    target_sample_size = 5000
    if cleaned_count < target_sample_size:
        raise ValueError(f"Cleaned dataset has only {cleaned_count} samples, less than {target_sample_size}")

    sampled_df = df.sample(n=target_sample_size, random_state=42).reset_index(drop=True)

    # Format columns: id, prompt, label, source, modality
    final_df = pd.DataFrame({
        "id": range(1, len(sampled_df) + 1),
        "prompt": sampled_df["prompt"],
        "label": "safe",
        "source": "alpaca",
        "modality": "text"
    })

    final_count = len(final_df)

    # Ensure target directories exist
    os.makedirs(os.path.dirname(out_clean_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_safe_path), exist_ok=True)

    # Save CSVs
    final_df.to_csv(out_clean_path, index=False, encoding="utf-8")
    final_df.to_csv(out_safe_path, index=False, encoding="utf-8")

    # Print summary
    print(f"Original sample count:   {orig_count}")
    print(f"Cleaned sample count:    {cleaned_count}")
    print(f"Final saved sample count:{final_count}")
    print("\nSaved files:")
    print(f"1. {out_clean_path}")
    print(f"2. {out_safe_path}")

if __name__ == "__main__":
    prepare_alpaca()
