import os
import random
import time
from pathlib import Path
import requests
from datasets import load_dataset

def download_coco_subset(num_images=1500, candidate_pool_size=2500):
    # 1. Reproducibility
    random.seed(42)
    
    output_dir = Path("dataset/external/coco/train2017")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=========================================================")
    print("       Phase 2.1: Downloading COCO Safe Image Subset     ")
    print("=========================================================")
    print(f"Target images to download: {num_images}")
    print(f"Target directory:          {output_dir.resolve()}\n")
    
    # 2. Stream COCO 2017 training split metadata using Hugging Face datasets
    print("Streaming COCO 2017 training metadata from Hugging Face...")
    ds = load_dataset("phiyodr/coco2017", split="train", streaming=True)
    
    candidate_dict = {}
    for row in ds:
        file_name = Path(row["file_name"]).name
        coco_url = row.get("coco_url")
        if not coco_url:
            coco_url = f"http://images.cocodataset.org/train2017/{file_name}"
            
        candidate_dict[file_name] = coco_url
        if len(candidate_dict) >= candidate_pool_size:
            break
            
    # Sort for deterministic selection across environments
    sorted_candidates = sorted(candidate_dict.items(), key=lambda x: x[0])
    
    # 3. Select exactly 1,500 unique images
    selected = random.sample(sorted_candidates, num_images)
    print(f"Successfully selected {len(selected)} unique COCO images with seed 42.\n")
    
    # 4. Download images with skip logic and progress logging
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    print("Starting download with multi-threaded connection pool...")
    
    def download_image(item):
        filename, url = item
        dest_path = output_dir / filename
        if dest_path.exists() and dest_path.stat().st_size > 0:
            return "skipped", filename
            
        try:
            r = requests.get(url, headers=headers, timeout=25)
            if r.status_code == 200 and len(r.content) > 0:
                with open(dest_path, "wb") as f:
                    f.write(r.content)
                return "downloaded", filename
            else:
                return "failed", filename
        except Exception:
            return "failed", filename

    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(download_image, item): item for item in selected}
        for idx, future in enumerate(as_completed(futures), start=1):
            status, fname = future.result()
            if status == "downloaded":
                success_count += 1
            elif status == "skipped":
                skipped_count += 1
                success_count += 1
            else:
                failed_count += 1
                
            if idx % 100 == 0 or idx == num_images:
                print(f"[{idx:4d}/{num_images}] Progress: {success_count} completed ({skipped_count} skipped), {failed_count} failed", flush=True)

    # 5. Final Summary
    print("\n=========================================================")
    print("                    Download Summary                     ")
    print("=========================================================")
    print(f"Total requested:        {num_images}")
    print(f"Successfully downloaded:{success_count}")
    print(f"Failed downloads:       {failed_count}")
    print(f"Output directory:       {output_dir}")

if __name__ == "__main__":
    download_coco_subset()
