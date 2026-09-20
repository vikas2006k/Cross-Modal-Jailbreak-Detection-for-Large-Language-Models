"""
================================================================================
Cross-Modal Jailbreak Detection for Large Language Models (CMJD)
Phase 3.1.2: Dataset Loader & Tokenization for DistilBERT
================================================================================
File: scripts/prepare_text_dataset.py

Description:
    Loads the verified CMJD-10K text dataset and partition annotations,
    converts categorical labels to integer targets (0: safe, 1: jailbreak),
    tokenizes prompts using distilbert-base-uncased (max_length=128),
    computes unpadded sequence length statistics, and serializes the
    resulting Hugging Face DatasetDict to disk.

Pipeline Steps:
    1. Verify dataset files and annotation counts.
    2. Load CMJD text dataset and validate schema/integrity.
    3. Convert labels into binary numeric targets (label_id).
    4. Construct disjoint train, validation, and test splits.
    5. Load and configure DistilBERT tokenizer (max_length=128).
    6. Tokenize prompts and compute sequence metrics.
    7. Assemble Hugging Face DatasetDict with PyTorch tensor formatting.
    8. Serialize tokenized dataset using save_to_disk().
    9. Export comprehensive tokenization report JSON.
    10. Output Phase 3.1.2 verification report.

Usage:
    python scripts/prepare_text_dataset.py
================================================================================
"""

import os
import json
import shutil
import logging
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def verify_dataset_files(
    csv_path="dataset/processed/cmjd_text_dataset.csv",
    train_json="dataset/annotations/train.json",
    val_json="dataset/annotations/val.json",
    test_json="dataset/annotations/test.json"
):
    """
    Step 1: Verify existence and sample counts of all required dataset files.
    """
    logger.info("Step 1: Verifying dataset files and split annotations...")
    paths = [Path(p) for p in [csv_path, train_json, val_json, test_json]]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Required dataset file missing: {p}")

    df = pd.read_csv(csv_path)
    train_data = json.load(open(train_json, encoding="utf-8"))
    val_data = json.load(open(val_json, encoding="utf-8"))
    test_data = json.load(open(test_json, encoding="utf-8"))

    total_ok = len(df) == 10000
    train_ok = len(train_data) == 7000
    val_ok = len(val_data) == 1500
    test_ok = len(test_data) == 1500

    print(f"Dataset File Check -> {'PASS' if total_ok and train_ok and val_ok and test_ok else 'FAIL'}")
    print(f"  * Total Prompts: {len(df)} (Expected: 10000) -> {'PASS' if total_ok else 'FAIL'}")
    print(f"  * Train Split:   {len(train_data)} (Expected: 7000)  -> {'PASS' if train_ok else 'FAIL'}")
    print(f"  * Val Split:     {len(val_data)} (Expected: 1500)  -> {'PASS' if val_ok else 'FAIL'}")
    print(f"  * Test Split:    {len(test_data)} (Expected: 1500)  -> {'PASS' if test_ok else 'FAIL'}")

    return df, train_data, val_data, test_data


def load_and_validate_cmjd(df):
    """
    Step 2 & 3: Validate schema, remove duplicate IDs, verify non-empty prompts,
    and create numeric label_id (safe -> 0, jailbreak -> 1).
    """
    logger.info("Step 2 & 3: Validating dataset and converting labels to binary targets...")
    expected_cols = {"id", "prompt", "label", "source", "modality", "attack_type"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"Missing expected columns: {expected_cols - set(df.columns)}")

    # Remove duplicate IDs if any
    initial_len = len(df)
    df = df.drop_duplicates(subset=["id"]).copy()
    if len(df) != initial_len:
        logger.warning(f"Removed {initial_len - len(df)} duplicate IDs.")

    # Validate non-empty prompts
    empty_prompts = (df["prompt"].astype(str).str.strip() == "").sum()
    if empty_prompts > 0:
        raise ValueError(f"Encountered {empty_prompts} empty prompts in dataset.")

    # Validate labels
    valid_labels = {"safe", "jailbreak"}
    found_labels = set(df["label"].unique())
    if not found_labels.issubset(valid_labels):
        raise ValueError(f"Unexpected label values found: {found_labels - valid_labels}")

    # Step 3: Convert labels to numeric label_id while keeping original label column
    label_map = {"safe": 0, "jailbreak": 1}
    df["label_id"] = df["label"].map(label_map)

    safe_count = int((df["label_id"] == 0).sum())
    jb_count = int((df["label_id"] == 1).sum())

    if safe_count != 5000 or jb_count != 5000:
        raise ValueError(f"Class imbalance detected: Safe={safe_count}, Jailbreak={jb_count}")

    logger.info(f"Label conversion verified: Safe (0) = {safe_count}, Jailbreak (1) = {jb_count}")
    return df


def create_dataset_splits(df, train_data, val_data, test_data):
    """
    Step 4: Partition DataFrame into train, val, and test splits matching annotation IDs.
    """
    logger.info("Step 4: Matching annotation split IDs and creating split DataFrames...")
    train_ids = [x["id"] for x in train_data]
    val_ids = [x["id"] for x in val_data]
    test_ids = [x["id"] for x in test_data]

    # Verify zero overlap between IDs
    t_set, v_set, te_set = set(train_ids), set(val_ids), set(test_ids)
    if len(t_set & v_set) > 0 or len(t_set & te_set) > 0 or len(v_set & te_set) > 0:
        raise ValueError("Overlap detected across annotation split IDs!")

    df_indexed = df.set_index("id", drop=False)
    train_df = df_indexed.loc[train_ids].reset_index(drop=True)
    val_df = df_indexed.loc[val_ids].reset_index(drop=True)
    test_df = df_indexed.loc[test_ids].reset_index(drop=True)

    if len(train_df) != 7000 or len(val_df) != 1500 or len(test_df) != 1500:
        raise ValueError(f"Split sizes mismatch: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    logger.info("Splits created: Train=7000, Validation=1500, Test=1500 (Zero overlap)")
    return train_df, val_df, test_df


def load_tokenizer_model(model_name="distilbert-base-uncased", tokenizer_dir="models/text_classifier/tokenizer"):
    """
    Step 5: Load DistilBERT tokenizer and serialize configuration to tokenizer directory.
    """
    logger.info(f"Step 5: Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Save tokenizer to models/text_classifier/tokenizer/
    tok_path = Path(tokenizer_dir)
    tok_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(tok_path))
    logger.info(f"Tokenizer saved to {tokenizer_dir}")

    return tokenizer


def tokenize_splits(tokenizer, train_df, val_df, test_df, max_length=128):
    """
    Step 6: Tokenize the prompt column across all splits and compute token length metrics.
    """
    logger.info("Step 6: Tokenizing prompt column and computing unpadded sequence statistics...")
    
    all_prompts = pd.concat([train_df["prompt"], val_df["prompt"], test_df["prompt"]]).tolist()
    
    # Analyze unpadded token lengths across the 10,000 prompts
    unpadded_lengths = [len(tokenizer.encode(p, truncation=False, add_special_tokens=True)) for p in all_prompts]
    max_tokens = int(max(unpadded_lengths))
    avg_tokens = float(sum(unpadded_lengths) / len(unpadded_lengths))
    truncated_samples = int(sum(1 for l in unpadded_lengths if l > max_length))

    logger.info(f"Token Length Statistics: Max={max_tokens}, Avg={avg_tokens:.2f}, Truncated(>{max_length})={truncated_samples}")

    def tokenize_dataframe(df_subset):
        prompts = df_subset["prompt"].tolist()
        encoded = tokenizer(
            prompts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True
        )
        return {
            "id": df_subset["id"].tolist(),
            "prompt": prompts,
            "label": df_subset["label_id"].tolist(),       # Standard HF Trainer target integer
            "label_id": df_subset["label_id"].tolist(),    # Explicit label_id
            "original_label": df_subset["label"].tolist(), # Original string label
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }

    tokenized_train = tokenize_dataframe(train_df)
    tokenized_val = tokenize_dataframe(val_df)
    tokenized_test = tokenize_dataframe(test_df)

    stats = {
        "max_tokens": max_tokens,
        "average_tokens": round(avg_tokens, 2),
        "truncated_samples": truncated_samples,
        "max_length": max_length
    }

    return tokenized_train, tokenized_val, tokenized_test, stats


def build_and_save_dataset_dict(
    tokenized_train,
    tokenized_val,
    tokenized_test,
    output_dir="models/text_classifier/tokenized_dataset"
):
    """
    Step 7 & 8: Build Hugging Face DatasetDict with PyTorch tensor formatting
    and serialize to disk using save_to_disk().
    """
    logger.info("Step 7 & 8: Assembling DatasetDict and saving to disk...")
    train_ds = Dataset.from_dict(tokenized_train)
    val_ds = Dataset.from_dict(tokenized_val)
    test_ds = Dataset.from_dict(tokenized_test)

    dataset_dict = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds
    })

    # Set PyTorch tensor format for model inputs
    dataset_dict.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
        output_all_columns=True
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(out_path))

    # Also place arrow/json references in root of tokenized_dataset for file-level audits
    for split in ["train", "validation", "test"]:
        split_arrow = list((out_path / split).glob("*.arrow"))
        if split_arrow:
            root_arrow = out_path / f"{split}.arrow"
            if not root_arrow.exists():
                shutil.copy(split_arrow[0], root_arrow)

    train_info = out_path / "train" / "dataset_info.json"
    if train_info.exists() and not (out_path / "dataset_info.json").exists():
        shutil.copy(train_info, out_path / "dataset_info.json")

    logger.info(f"DatasetDict successfully saved to {output_dir}")
    return dataset_dict


def generate_tokenization_report(stats, report_path="models/text_classifier/metrics/tokenization_report.json"):
    """
    Step 9: Generate tokenization report JSON.
    """
    logger.info(f"Step 9: Generating tokenization report at {report_path}...")
    report_p = Path(report_path)
    report_p.parent.mkdir(parents=True, exist_ok=True)

    report_content = {
        "total_samples": 10000,
        "train_samples": 7000,
        "validation_samples": 1500,
        "test_samples": 1500,
        "safe_samples": 5000,
        "jailbreak_samples": 5000,
        "max_length": stats["max_length"],
        "average_tokens": stats["average_tokens"],
        "max_tokens": stats["max_tokens"],
        "truncated_samples": stats["truncated_samples"],
        "tokenizer": "distilbert-base-uncased",
        "seed": 42
    }

    with open(report_p, "w", encoding="utf-8") as f:
        json.dump(report_content, f, indent=2)

    logger.info("Tokenization report saved successfully.")
    return report_content


def run_pipeline():
    """
    Main execution pipeline for Phase 3.1.2.
    """
    # Step 1: Verify files
    df, train_data, val_data, test_data = verify_dataset_files()

    # Step 2 & 3: Load, validate, and convert labels
    df = load_and_validate_cmjd(df)

    # Step 4: Create splits
    train_df, val_df, test_df = create_dataset_splits(df, train_data, val_data, test_data)

    # Step 5: Load tokenizer
    tokenizer = load_tokenizer_model()

    # Step 6: Tokenize splits & compute metrics
    tokenized_train, tokenized_val, tokenized_test, stats = tokenize_splits(tokenizer, train_df, val_df, test_df)

    # Step 7 & 8: Create and save DatasetDict
    dataset_dict = build_and_save_dataset_dict(tokenized_train, tokenized_val, tokenized_test)

    # Step 9: Generate report
    report = generate_tokenization_report(stats)

    # Step 11: Print exact verification report
    print("=========================================================")
    print("CMJD PHASE 3.1.2 TOKENIZATION REPORT")
    print("=========================================================")
    print()
    print("Dataset Loaded = PASS")
    print()
    print("Labels Converted = PASS")
    print()
    print(f"Train Samples = {len(train_df)}")
    print()
    print(f"Validation Samples = {len(val_df)}")
    print()
    print(f"Test Samples = {len(test_df)}")
    print()
    print("Tokenizer Loaded = PASS")
    print()
    print("Tokenizer = distilbert-base-uncased")
    print()
    print(f"Maximum Length = {stats['max_length']}")
    print()
    print(f"Average Token Length = {stats['average_tokens']}")
    print()
    print(f"Truncated Samples = {stats['truncated_samples']}")
    print()
    print("DatasetDict Created = PASS")
    print()
    print("Tokenized Dataset Saved = PASS")
    print()
    print("Report Generated = PASS")
    print()
    print("STATUS")
    print()
    print("READY FOR PHASE 3.1.3 — DistilBERT Training Pipeline")
    print()
    print("=========================================================")


if __name__ == "__main__":
    run_pipeline()
