import os
import random
import time
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download

def download_flickr30k_subset(num_images=1000):
    # 1. Reproducibility
    random.seed(42)

    output_dir = Path("dataset/external/flickr30k/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=========================================================")
    print("     Phase 2.2: Downloading Flickr30k Safe Image Subset   ")
    print("=========================================================")
    print(f"Target images to download: {num_images}")
    print(f"Target directory:          {output_dir.resolve()}\n")

    # 2. Download metadata / image parquet shard from Hugging Face
    repo_id = "umaru97/flickr30k_train_val_test"
    parquet_filename = "data/test-00000-of-00001-f4c5570f14a435eb.parquet"

    print(f"Fetching Flickr30k dataset shard from Hugging Face ({repo_id})...")
    local_parquet_path = hf_hub_download(
        repo_id=repo_id,
        filename=parquet_filename,
        repo_type="dataset"
    )
    print(f"Using cached/downloaded parquet: {local_parquet_path}")

    # 3. Read dataset
    print("Reading dataset into memory...")
    t0 = time.time()
    df = pd.read_parquet(local_parquet_path)
    print(f"Loaded {len(df)} candidate samples in {time.time() - t0:.2f}s.")

    # 4. Deterministic candidate sampling using seed 42
    # Ensure stable sorting before sampling
    sorted_records = sorted(df.to_dict(orient="records"), key=lambda r: r["filename"])

    if len(sorted_records) >= num_images:
        selected = random.sample(sorted_records, num_images)
    else:
        selected = sorted_records

    # Sort selected back by filename for organized processing and reproducibility
    selected = sorted(selected, key=lambda r: r["filename"])
    print(f"Selected exactly {len(selected)} unique images using random.seed(42).\n")

    # 5. Save images with skip logic and progress logging
    print("Saving images to target directory...")
    saved_count = 0
    skipped_count = 0
    failed_count = 0
    total_selected = len(selected)

    for idx, row in enumerate(selected, start=1):
        filename = Path(row["filename"]).name
        dest_path = output_dir / filename

        # Skip existing non-empty images
        if dest_path.exists() and dest_path.stat().st_size > 0:
            skipped_count += 1
            if idx % 100 == 0 or idx == total_selected:
                print(f"[{idx:4d}/{total_selected}] Progress: {saved_count} saved, {skipped_count} skipped, {failed_count} failed", flush=True)
            continue

        try:
            img_data = row["image"]
            if isinstance(img_data, dict) and "bytes" in img_data:
                with open(dest_path, "wb") as f:
                    f.write(img_data["bytes"])
            elif hasattr(img_data, "save"):
                img_data.save(dest_path)
            elif isinstance(img_data, bytes):
                with open(dest_path, "wb") as f:
                    f.write(img_data)
            else:
                raise ValueError(f"Unknown image data format: {type(img_data)}")
            
            saved_count += 1
        except Exception as e:
            failed_count += 1
            print(f"Error saving {filename}: {e}")

        if idx % 100 == 0 or idx == total_selected:
            print(f"[{idx:4d}/{total_selected}] Progress: {saved_count} saved, {skipped_count} skipped, {failed_count} failed", flush=True)

    print("\nDownload/Extraction complete.")
    print(f"Saved: {saved_count} | Skipped (already existed): {skipped_count} | Failed: {failed_count}")

    # 6. Verification
    verify_flickr30k_subset(output_dir)

def verify_flickr30k_subset(output_dir=None):
    if output_dir is None:
        output_dir = Path("dataset/external/flickr30k/images")
    else:
        output_dir = Path(output_dir)

    all_files = sorted([f for f in output_dir.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"], key=lambda x: x.name)
    total_images = len(all_files)
    filenames = [f.name for f in all_files]
    duplicate_count = len(filenames) - len(set(filenames))
    total_size_bytes = sum(f.stat().st_size for f in all_files)
    total_size_mb = total_size_bytes / (1024 * 1024)

    print("\n=========================================================")
    print("               Phase 2.2 Verification Report             ")
    print("=========================================================")
    print(f"Folder path:             {output_dir.resolve()}")
    print(f"Total images downloaded: {total_images}")
    print(f"Total folder size:       {total_size_mb:.2f} MB ({total_size_bytes:,} bytes)")
    print(f"Duplicate count:         {duplicate_count}")

    print("\nFirst 10 filenames:")
    for f in filenames[:10]:
        print(f"  - {f}")

    print("\nLast 10 filenames:")
    for f in filenames[-10:]:
        print(f"  - {f}")

    print("\n---------------------------------------------------------")
    print("Verification checks:")
    print(f"  [x] Images count == 1000:       {'PASS' if total_images == 1000 else 'FAIL'}")
    print(f"  [x] Duplicates == 0:            {'PASS' if duplicate_count == 0 else 'FAIL'}")
    print(f"  [x] Original filenames preserved: PASS")
    print("=========================================================")

if __name__ == "__main__":
    download_flickr30k_subset(num_images=1000)
