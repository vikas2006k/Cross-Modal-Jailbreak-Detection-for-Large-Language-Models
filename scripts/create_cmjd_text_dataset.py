import os
import sys
import json
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set random seed
SEED = 42
random.seed(SEED)

def generate_supplemental_prompts(needed_count):
    """
    Generate supplemental unique adversarial prompts if needed to reach exactly 5,000.
    """
    from scripts.generate_synthetic_jailbreak_dataset import (
        generate_role_override, generate_prompt_injection,
        generate_policy_bypass, generate_developer_mode,
        generate_tool_injection, generate_indirect_injection
    )
    
    generators = [
        ("role_override", generate_role_override),
        ("prompt_injection", generate_prompt_injection),
        ("policy_bypass", generate_policy_bypass),
        ("developer_mode", generate_developer_mode),
        ("tool_injection", generate_tool_injection),
        ("indirect_injection", generate_indirect_injection),
    ]
    
    supplemental = []
    idx = 0
    while len(supplemental) < needed_count:
        cat_name, gen_func = generators[idx % len(generators)]
        candidate_pool = gen_func(target_count=50)
        for p in candidate_pool:
            if len(supplemental) >= needed_count:
                break
            supplemental.append({
                "prompt": p,
                "label": "jailbreak",
                "source": "cmjd_generated",
                "modality": "text",
                "attack_type": cat_name
            })
        idx += 1
        
    return supplemental

def create_cmjd_text_dataset():
    safe_path = "dataset/text/safe/safe_prompts.csv"
    jb_path = "dataset/text/jailbreak/jailbreak_prompts.csv"
    syn_jb_path = "dataset/text/jailbreak/synthetic_jailbreak_prompts.csv"
    
    out_master_csv = "dataset/processed/cmjd_text_dataset.csv"
    out_train_json = "dataset/annotations/train.json"
    out_val_json = "dataset/annotations/val.json"
    out_test_json = "dataset/annotations/test.json"
    out_stats_json_annot = "dataset/annotations/dataset_statistics.json"
    out_stats_json_proc = "dataset/processed/dataset_statistics.json"

    # 1. Read input CSV files
    print("Reading input datasets...")
    df_safe = pd.read_csv(safe_path)
    df_jb = pd.read_csv(jb_path)
    df_syn_jb = pd.read_csv(syn_jb_path)
    
    print(f"  - Safe prompts loaded:               {len(df_safe):5d}")
    print(f"  - Benchmark jailbreak loaded:        {len(df_jb):5d}")
    print(f"  - Synthetic novelty jailbreak loaded:{len(df_syn_jb):5d}")

    # Standardize safe prompts attack_type = "safe"
    df_safe["attack_type"] = "safe"
    df_safe["label"] = "safe"
    df_safe["modality"] = "text"

    # Combine jailbreak prompts
    jailbreak_combined = pd.concat([df_jb, df_syn_jb], ignore_index=True)
    jailbreak_combined = jailbreak_combined.drop_duplicates(subset=["prompt"]).copy()
    
    # 2. Ensure exactly 5,000 safe prompts
    if len(df_safe) > 5000:
        df_safe = df_safe.sample(n=5000, random_state=SEED).reset_index(drop=True)
    elif len(df_safe) < 5000:
        raise ValueError(f"Expected at least 5000 safe prompts, found {len(df_safe)}")

    # 3. Ensure exactly 5,000 jailbreak prompts
    if len(jailbreak_combined) > 5000:
        jailbreak_combined = jailbreak_combined.sample(n=5000, random_state=SEED).reset_index(drop=True)
    elif len(jailbreak_combined) < 5000:
        deficit = 5000 - len(jailbreak_combined)
        print(f"  [*] Generating {deficit} supplemental novelty prompts to guarantee exactly 5,000 jailbreak samples...")
        supp_records = generate_supplemental_prompts(deficit)
        df_supp = pd.DataFrame(supp_records)
        jailbreak_combined = pd.concat([jailbreak_combined, df_supp], ignore_index=True)
        jailbreak_combined = jailbreak_combined.drop_duplicates(subset=["prompt"])
        
        # If any slight overlap, top-up exactly
        while len(jailbreak_combined) < 5000:
            top_up = generate_supplemental_prompts(5000 - len(jailbreak_combined))
            jailbreak_combined = pd.concat([jailbreak_combined, pd.DataFrame(top_up)], ignore_index=True)
            jailbreak_combined = jailbreak_combined.drop_duplicates(subset=["prompt"])
            
        jailbreak_combined = jailbreak_combined.iloc[:5000].reset_index(drop=True)

    jailbreak_combined["label"] = "jailbreak"
    jailbreak_combined["modality"] = "text"

    # Verify counts
    assert len(df_safe) == 5000, f"Safe count must be 5000, got {len(df_safe)}"
    assert len(jailbreak_combined) == 5000, f"Jailbreak count must be 5000, got {len(jailbreak_combined)}"

    # 4. Merge into master dataframe
    master_df = pd.concat([df_safe, jailbreak_combined], ignore_index=True)
    
    # 5. Shuffle using random seed = 42
    master_df = master_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # 6. Reassign IDs from 1 to 10,000
    master_df["id"] = range(1, len(master_df) + 1)
    master_df = master_df[["id", "prompt", "label", "source", "modality", "attack_type"]]

    # Save master CSV
    os.makedirs(os.path.dirname(out_master_csv), exist_ok=True)
    master_df.to_csv(out_master_csv, index=False, encoding="utf-8")

    # 7. Stratified Train / Val / Test Split (70% Train, 15% Val, 15% Test)
    # Total: 7,000 Train, 1,500 Val, 1,500 Test
    train_df, temp_df = train_test_split(
        master_df,
        test_size=0.30,
        random_state=SEED,
        stratify=master_df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["label"]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save JSON splits
    os.makedirs("dataset/annotations", exist_ok=True)
    
    with open(out_train_json, "w", encoding="utf-8") as f:
        json.dump(train_df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
        
    with open(out_val_json, "w", encoding="utf-8") as f:
        json.dump(val_df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
        
    with open(out_test_json, "w", encoding="utf-8") as f:
        json.dump(test_df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    # 8. Create dataset_statistics.json
    attack_categories = sorted(master_df["attack_type"].unique().tolist())
    category_distribution = master_df["attack_type"].value_counts().to_dict()

    stats_data = {
        "dataset_name": "CMJD-10K",
        "version": "1.0",
        "safe_samples": int((master_df["label"] == "safe").sum()),
        "jailbreak_samples": int((master_df["label"] == "jailbreak").sum()),
        "train_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples": len(test_df),
        "attack_categories": attack_categories,
        "category_distribution": category_distribution
    }

    with open(out_stats_json_annot, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)
    with open(out_stats_json_proc, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)

    # 9. Print console output
    print("\n=========================================================")
    print("      CMJD-10K Master Text Dataset Merged Successfully   ")
    print("=========================================================")
    print(f"Total samples:                 {len(master_df):6d}")
    print(f"Safe count:                    {(master_df['label'] == 'safe').sum():6d}")
    print(f"Jailbreak count:               {(master_df['label'] == 'jailbreak').sum():6d}")
    print("\nSplit counts:")
    print(f"  - Train samples (70%):       {len(train_df):6d}")
    print(f"  - Validation samples (15%):  {len(val_df):6d}")
    print(f"  - Test samples (15%):        {len(test_df):6d}")
    
    print("\nAttack category distribution across entire CMJD-10K:")
    for cat, count in category_distribution.items():
        pct = (count / len(master_df)) * 100
        print(f"  - {cat:26s}: {count:5d} ({pct:5.1f}%)")

    print("\nOutput file paths:")
    print(f"1. Master CSV:        {out_master_csv}")
    print(f"2. Train JSON:        {out_train_json}")
    print(f"3. Validation JSON:   {out_val_json}")
    print(f"4. Test JSON:         {out_test_json}")
    print(f"5. Statistics JSON:   {out_stats_json_annot}")

if __name__ == "__main__":
    create_cmjd_text_dataset()
