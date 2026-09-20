import os
import pandas as pd
from datasets import load_dataset, concatenate_datasets

def download_official_prompt_injections():
    output_dir = "dataset/external/prompt_injection"
    output_file = os.path.join(output_dir, "prompt_injection.parquet")
    old_csv_file = os.path.join(output_dir, "prompt_injection.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Replace existing dataset inside directory
    if os.path.exists(old_csv_file):
        os.remove(old_csv_file)
        
    print("Downloading official 'deepset/prompt-injections' from Hugging Face...")
    ds = load_dataset("deepset/prompt-injections")
    
    # Combine available splits to capture all data
    splits = [ds[k] for k in ds.keys()]
    combined_ds = concatenate_datasets(splits) if len(splits) > 1 else splits[0]
    
    # Save as parquet preserving all original columns
    combined_ds.to_parquet(output_file)
    
    # File size calculation
    file_size_bytes = os.path.getsize(output_file)
    file_size_kb = file_size_bytes / 1024
    
    # Print metrics
    print("\nDownload & Save Complete!")
    print(f"Number of rows downloaded: {len(combined_ds)}")
    print(f"Column names:              {combined_ds.column_names}")
    print(f"Output file size:          {file_size_bytes:,} bytes ({file_size_kb:.2f} KB)")
    print(f"Output file path:          {output_file}")

if __name__ == "__main__":
    download_official_prompt_injections()
