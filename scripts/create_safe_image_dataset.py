import os
import shutil
from pathlib import Path
import pandas as pd

def assemble_safe_image_dataset():
    coco_dir = Path("dataset/external/coco/train2017")
    flickr_dir = Path("dataset/external/flickr30k/images")
    output_dir = Path("dataset/images/safe")
    metadata_dir = Path("dataset/processed")
    metadata_path = metadata_dir / "image_metadata.csv"

    # Ensure output directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print("=========================================================")
    print("      Phase 2.3: Assemble Safe Image Dataset             ")
    print("=========================================================")
    print(f"COCO source dir:     {coco_dir.resolve()}")
    print(f"Flickr30k source dir:{flickr_dir.resolve()}")
    print(f"Output directory:    {output_dir.resolve()}")
    print(f"Metadata file:       {metadata_path.resolve()}\n")

    # 1. Read input image files (sorted deterministically)
    coco_images = sorted([f for f in coco_dir.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"], key=lambda x: x.name)
    flickr_images = sorted([f for f in flickr_dir.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"], key=lambda x: x.name)

    print(f"Found {len(coco_images)} COCO images.")
    print(f"Found {len(flickr_images)} Flickr30k images.")

    if len(coco_images) != 1500:
        raise ValueError(f"Expected 1500 COCO images, but found {len(coco_images)}")
    if len(flickr_images) != 1000:
        raise ValueError(f"Expected 1000 Flickr30k images, but found {len(flickr_images)}")

    # 2. Copy and rename sequentially: safe_00001.jpg to safe_02500.jpg
    metadata_records = []
    current_id = 1
    coco_copied = 0
    flickr_copied = 0

    print("\nCopying and sequentially renaming COCO images (1..1500)...")
    for src_file in coco_images:
        new_filename = f"safe_{current_id:05d}.jpg"
        dest_file = output_dir / new_filename

        shutil.copy2(src_file, dest_file)
        coco_copied += 1

        metadata_records.append({
            "id": current_id,
            "file_name": new_filename,
            "label": "safe",
            "source": "coco",
            "modality": "image"
        })
        current_id += 1

    print(f"COCO copy completed: {coco_copied} images.")

    print("\nCopying and sequentially renaming Flickr30k images (1501..2500)...")
    for src_file in flickr_images:
        new_filename = f"safe_{current_id:05d}.jpg"
        dest_file = output_dir / new_filename

        shutil.copy2(src_file, dest_file)
        flickr_copied += 1

        metadata_records.append({
            "id": current_id,
            "file_name": new_filename,
            "label": "safe",
            "source": "flickr30k",
            "modality": "image"
        })
        current_id += 1

    print(f"Flickr30k copy completed: {flickr_copied} images.")

    total_copied = coco_copied + flickr_copied

    # 3. Create metadata CSV
    print(f"\nWriting metadata CSV to {metadata_path}...")
    df_metadata = pd.DataFrame(metadata_records)
    df_metadata.to_csv(metadata_path, index=False)
    print(f"Saved {len(df_metadata)} rows into {metadata_path}.")

    # 4. Verification
    verify_assembly(output_dir, metadata_path, coco_copied, flickr_copied, total_copied)

def verify_assembly(output_dir, metadata_path, coco_copied, flickr_copied, total_copied):
    jpg_files = sorted([f for f in output_dir.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"], key=lambda x: x.name)
    total_images_in_folder = len(jpg_files)
    filenames = [f.name for f in jpg_files]
    duplicate_filenames = len(filenames) - len(set(filenames))

    df_meta = pd.read_csv(metadata_path)
    metadata_rows = len(df_meta)

    # Check ID sequencing and filenames
    expected_filenames = [f"safe_{i:05d}.jpg" for i in range(1, 2501)]
    filenames_match = (filenames == expected_filenames)
    meta_filenames_match = (df_meta["file_name"].tolist() == expected_filenames)
    meta_id_match = (df_meta["id"].tolist() == list(range(1, 2501)))
    coco_source_match = ((df_meta.iloc[:1500]["source"] == "coco").all())
    flickr_source_match = ((df_meta.iloc[1500:]["source"] == "flickr30k").all())
    label_match = ((df_meta["label"] == "safe").all())
    modality_match = ((df_meta["modality"] == "image").all())

    folder_size_bytes = sum(f.stat().st_size for f in jpg_files)
    folder_size_mb = folder_size_bytes / (1024 * 1024)

    print("\n=========================================================")
    print("               Phase 2.3 Verification Report             ")
    print("=========================================================")
    print(f"COCO copied = {coco_copied}")
    print(f"Flickr30k copied = {flickr_copied}")
    print(f"Total copied = {total_copied}")
    print(f"Duplicate filenames = {duplicate_filenames}")
    print(f"Metadata rows = {metadata_rows}")
    print(f"Output folder path: {output_dir.resolve()}")
    print(f"Total folder size: {folder_size_mb:.2f} MB ({folder_size_bytes:,} bytes)")

    print("\nMetadata sample (first 5 rows):")
    print(df_meta.head(5).to_string(index=False))

    print("\nMetadata transition sample (rows 1499-1503):")
    print(df_meta.iloc[1498:1503].to_string(index=False))

    print("\nMetadata sample (last 5 rows):")
    print(df_meta.tail(5).to_string(index=False))

    print("\n---------------------------------------------------------")
    print("Integrity checks:")
    print(f"  [x] Total copied == 2500:             {'PASS' if total_copied == 2500 else 'FAIL'}")
    print(f"  [x] Images in folder == 2500:         {'PASS' if total_images_in_folder == 2500 else 'FAIL'}")
    print(f"  [x] Duplicate filenames == 0:         {'PASS' if duplicate_filenames == 0 else 'FAIL'}")
    print(f"  [x] Filenames match safe_XXXXX.jpg:   {'PASS' if filenames_match and meta_filenames_match else 'FAIL'}")
    print(f"  [x] Metadata IDs 1..2500:             {'PASS' if meta_id_match else 'FAIL'}")
    print(f"  [x] First 1500 source == coco:        {'PASS' if coco_source_match else 'FAIL'}")
    print(f"  [x] Next 1000 source == flickr30k:    {'PASS' if flickr_source_match else 'FAIL'}")
    print(f"  [x] All labels == safe:               {'PASS' if label_match else 'FAIL'}")
    print(f"  [x] All modality == image:            {'PASS' if modality_match else 'FAIL'}")
    print("=========================================================")

if __name__ == "__main__":
    assemble_safe_image_dataset()
