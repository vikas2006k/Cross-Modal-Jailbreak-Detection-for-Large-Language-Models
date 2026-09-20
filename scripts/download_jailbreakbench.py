import os
import pandas as pd
from datasets import load_dataset, concatenate_datasets

def download_jailbreakbench():
    output_dir = "dataset/external/jailbreakbench"
    output_file = os.path.join(output_dir, "jailbreakbench.parquet")
    
    # 1. Create folder
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading official 'JailbreakBench' (JailbreakBench/JBB-Behaviors) from Hugging Face...")
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    
    # 2. Download complete dataset without sampling or preprocessing
    splits = [ds[k] for k in ds.keys()]
    combined_ds = concatenate_datasets(splits) if len(splits) > 1 else splits[0]
    
    # 3. Save as Parquet preserving all original columns
    combined_ds.to_parquet(output_file)
    
    # 4. Calculate file metrics
    file_size_bytes = os.path.getsize(output_file)
    file_size_kb = file_size_bytes / 1024
    
    # 5. Print metrics
    print("\nDownload & Save Complete!")
    print(f"Number of rows downloaded: {len(combined_ds)}")
    print(f"Column names:              {combined_ds.column_names}")
    print(f"Output file size:          {file_size_bytes:,} bytes ({file_size_kb:.2f} KB)")
    print(f"Output file path:          {output_file}")

if __name__ == "__main__":
    download_jailbreakbench()
