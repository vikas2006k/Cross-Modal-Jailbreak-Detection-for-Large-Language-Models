import os
import glob
import re
import pandas as pd

def detect_prompt_column(df):
    """
    Automatically detect the prompt/jailbreak text column from common column naming schemes.
    """
    candidates = [
        "prompt", "behavior", "goal", "text", "instruction", 
        "input", "query", "harmful_behavior", "injection", "attack"
    ]
    col_map = {col.lower().strip(): col for col in df.columns}
    
    # Check exact match candidates
    for cand in candidates:
        if cand in col_map:
            return col_map[cand]
            
    # Check substring matches
    for col in df.columns:
        col_lower = col.lower().strip()
        for cand in candidates:
            if cand in col_lower:
                return col
                
    # Fallback: pick the first object/string column with longest average string length
    str_cols = df.select_dtypes(include=['object', 'string']).columns
    if len(str_cols) > 0:
        lens = {c: df[c].astype(str).str.len().mean() for c in str_cols}
        return max(lens, key=lens.get)
        
    return df.columns[0]

def classify_attack_type(prompt_text, source):
    """
    Classify prompts into specific attack taxonomy:
    - system_prompt_extraction
    - prompt_injection
    - role_override
    - tool_injection
    - encoded_attack
    - policy_bypass
    - unknown
    """
    text = prompt_text.lower()

    # 1. System Prompt Extraction
    extraction_patterns = [
        "repeat your instructions", "output your system prompt", "print your initial instructions",
        "what are your instructions", "leak prompt", "show system prompt", "reveal instructions",
        "system prompt", "developer instructions", "repeat the text above", "what was the first prompt",
        "system message", "output everything above", "print initial prompt"
    ]
    if any(p in text for p in extraction_patterns):
        return "system_prompt_extraction"

    # 2. Prompt Injection / Instruction Overrides
    injection_patterns = [
        "ignore all previous instructions", "ignore previous instructions", "ignore the above",
        "disregard all previous", "disregard prior", "override previous", "new instruction",
        "from now on, ignore", "system override", "forget your instructions", "stop being an ai",
        "bypass safeguards", "ignore instructions"
    ]
    if any(p in text for p in injection_patterns):
        return "prompt_injection"

    # 3. Role Override / Persona Hijack / DAN
    role_patterns = [
        "dan", "jailbreak", "do anything now", "aim", "stan", "dev mode", "developer mode",
        "evil confidant", "act as", "you are now", "pretend you are", "roleplay as",
        "character named", "hypothetical unrestricted", "unfiltered assistant", "jailbroken",
        "always answer", "never refuse", "unrestricted mode"
    ]
    if any(p in text for p in role_patterns):
        return "role_override"

    # 4. Tool / Code / Command Injection
    tool_patterns = [
        "<execute>", "<script>", "exec(", "eval(", "system(", "sql", "bash", "cmd.exe",
        "powershell", "curl ", "wget ", "format c:", "drop table", "union select",
        "import os", "/etc/passwd", "javascript:", "function_call", "popen"
    ]
    if any(p in text for p in tool_patterns):
        return "tool_injection"

    # 5. Encoded / Obfuscated Attacks
    encoded_patterns = [
        "rot13", "base64", "hex:", "decode the following", "caesar cipher", "morse code",
        "atbash", "binary string", "cipher text", "encoded prompt"
    ]
    if any(p in text for p in encoded_patterns) or bool(re.search(r"[A-Za-z0-9+/=]{35,}", prompt_text)):
        return "encoded_attack"

    # 6. Policy Bypass / Harmful Request (default for AdvBench harmful queries)
    bypass_patterns = [
        "hypothetical", "for educational purposes", "fictional story", "academic research",
        "purely educational", "red team test", "safety bypass", "write a guide", "how to make",
        "how to build", "how to hack", "create a malware", "synthesize", "exploit", "weapons",
        "bomb", "illegal", "steal"
    ]
    if any(p in text for p in bypass_patterns) or source == "advbench":
        return "policy_bypass"

    return "unknown"

def load_source_csvs(folder_path, source_name):
    """
    Read all CSV files from a directory and return a standardized list of dicts.
    """
    samples = []
    if not os.path.exists(folder_path):
        return samples, 0

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    total_raw_count = 0

    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath, encoding="utf-8", on_bad_lines="skip")
        except UnicodeDecodeError:
            df = pd.read_csv(fpath, encoding="latin-1", on_bad_lines="skip")

        if df.empty:
            continue

        prompt_col = detect_prompt_column(df)
        raw_prompts = df[prompt_col].dropna().tolist()
        total_raw_count += len(raw_prompts)

        for p in raw_prompts:
            samples.append({
                "raw_prompt": str(p),
                "source": source_name
            })

    return samples, total_raw_count

def prepare_jailbreak_dataset():
    advbench_dir = "dataset/external/advbench"
    prompt_inj_dir = "dataset/external/prompt_injection"
    out_clean_path = "dataset/processed/jailbreak_prompts_clean.csv"
    out_train_path = "dataset/text/jailbreak/jailbreak_prompts.csv"

    # 1. Read datasets
    advbench_samples, advbench_raw_count = load_source_csvs(advbench_dir, "advbench")
    prompt_inj_samples, prompt_inj_raw_count = load_source_csvs(prompt_inj_dir, "prompt_injection")

    total_read = advbench_raw_count + prompt_inj_raw_count
    all_samples = advbench_samples + prompt_inj_samples

    print("==================================================")
    print("      Phase 1.3: Jailbreak Dataset Preparation    ")
    print("==================================================")
    print(f"Number of prompts read from AdvBench:          {advbench_raw_count}")
    print(f"Number of prompts read from Prompt Injection:  {prompt_inj_raw_count}")
    print(f"Total raw prompts read:                        {total_read}")

    if len(all_samples) == 0:
        print("\n[!] WARNING: No CSV files were found in:")
        print(f"    - {advbench_dir}/")
        print(f"    - {prompt_inj_dir}/")
        print(f"Remaining count needed: 5000")
        print("Please place your AdvBench and Prompt Injection CSV files in the folders above and re-run.")
        return

    df = pd.DataFrame(all_samples)

    # 2. Cleaning Rules:
    # - Strip whitespace
    df["prompt"] = df["raw_prompt"].astype(str).str.strip()

    # - Remove null or empty prompts
    df = df[df["prompt"].notna() & (df["prompt"] != "")].copy()

    # - Keep prompts between 15 and 500 characters
    prompt_len = df["prompt"].str.len()
    df = df[(prompt_len >= 15) & (prompt_len <= 500)].copy()

    # - Remove prompts containing only symbols or numbers (must contain letters)
    has_letters = df["prompt"].apply(lambda p: bool(re.search(r"[a-zA-Z]", p)))
    df = df[has_letters].copy()
    cleaned_count = len(df)

    # - Remove duplicates
    df = df.drop_duplicates(subset=["prompt"]).copy()
    dedup_count = len(df)

    print(f"Number after basic cleaning:                   {cleaned_count}")
    print(f"Number after duplicate removal:                {dedup_count}")

    # 3. Dataset Sampling
    target_count = 5000
    if dedup_count >= target_count:
        sampled_df = df.sample(n=target_count, random_state=42).reset_index(drop=True)
    else:
        sampled_df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        remaining_needed = target_count - dedup_count
        print(f"Remaining count needed to reach 5000:          {remaining_needed}")

    # 4. Classify attack types
    sampled_df["attack_type"] = sampled_df.apply(
        lambda r: classify_attack_type(r["prompt"], r["source"]),
        axis=1
    )

    # 5. Build final formatted dataframe
    final_df = pd.DataFrame({
        "id": range(1, len(sampled_df) + 1),
        "prompt": sampled_df["prompt"],
        "label": "jailbreak",
        "source": sampled_df["source"],
        "modality": "text",
        "attack_type": sampled_df["attack_type"]
    })

    final_saved_count = len(final_df)
    print(f"Final saved count:                             {final_saved_count}")

    # Attack type distribution
    dist = final_df["attack_type"].value_counts().to_dict()
    print("\nDistribution of attack_type categories:")
    for cat, count in dist.items():
        pct = (count / final_saved_count) * 100
        print(f"  - {cat:26s}: {count:5d} ({pct:5.1f}%)")

    # 6. Save outputs
    os.makedirs(os.path.dirname(out_clean_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_train_path), exist_ok=True)

    final_df.to_csv(out_clean_path, index=False, encoding="utf-8")
    final_df.to_csv(out_train_path, index=False, encoding="utf-8")

    print("\nOutput file paths:")
    print(f"1. {out_clean_path}")
    print(f"2. {out_train_path}")

if __name__ == "__main__":
    prepare_jailbreak_dataset()
