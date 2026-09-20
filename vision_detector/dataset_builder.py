"""
Dataset construction and organization pipeline for the CMJD Vision Detector.
Organizes clean safe images and synthesizes realistic typographic visual jailbreaks,
generating train/val/test splits and dataset metadata.csv.
"""

import csv
from pathlib import Path
import random
import shutil
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_SAFE_DIR = PROJECT_ROOT / "dataset" / "images" / "safe"
TEXT_DATASET_CSV = PROJECT_ROOT / "dataset" / "processed" / "cmjd_text_dataset.csv"
ADVBENCH_CSV = PROJECT_ROOT / "dataset" / "external" / "advbench" / "harmful_behaviors.csv"

VISION_DATASET_DIR = PROJECT_ROOT / "vision_detector" / "dataset"
SAFE_OUTPUT_DIR = VISION_DATASET_DIR / "safe"
JAILBREAK_OUTPUT_DIR = VISION_DATASET_DIR / "jailbreak"
METADATA_CSV = VISION_DATASET_DIR / "metadata.csv"

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def get_jailbreak_prompts(max_prompts: int = 1500) -> List[Tuple[str, str]]:
    """
    Retrieves high-severity jailbreak prompts from the CMJD text dataset and AdvBench.
    Returns a list of (prompt, attack_subtype).
    """
    prompts = []

    # Load from CMJD processed text dataset
    if TEXT_DATASET_CSV.exists():
        df = pd.read_csv(TEXT_DATASET_CSV)
        jailbreak_df = df[df["label"] == 1]
        for _, row in jailbreak_df.iterrows():
            p = str(row["prompt"]).strip()
            cat = str(row.get("category", "prompt_injection")).strip()
            if p:
                prompts.append((p, cat))

    # Load from AdvBench if available
    if ADVBENCH_CSV.exists():
        df_adv = pd.read_csv(ADVBENCH_CSV)
        col = "goal" if "goal" in df_adv.columns else df_adv.columns[0]
        for val in df_adv[col].dropna():
            p = str(val).strip()
            if p:
                prompts.append((p, "direct_harmful_behavior"))

    random.seed(42)
    random.shuffle(prompts)
    return prompts[:max_prompts]


def render_typographic_jailbreak(
    prompt: str,
    output_path: Path,
    bg_image_path: Path = None,
    style_idx: int = 0
) -> None:
    """
    Renders an adversarial prompt onto an image to simulate realistic visual jailbreak attacks:
    - Style 0: System banner / UI prompt injection dialog
    - Style 1: Semi-transparent text overlay on natural background image
    - Style 2: Obfuscated high-contrast typographic card
    - Style 3: Terminal / Code editor payload injection
    """
    width, height = 512, 512

    if bg_image_path and bg_image_path.exists() and style_idx in [1, 2]:
        try:
            with Image.open(bg_image_path) as bg:
                base_img = bg.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        except Exception:
            base_img = Image.new("RGB", (width, height), color=(30, 30, 35))
    else:
        # Synthetic card background
        if style_idx == 0:
            base_img = Image.new("RGB", (width, height), color=(245, 245, 250))
        elif style_idx == 3:
            base_img = Image.new("RGB", (width, height), color=(18, 22, 28))
        else:
            base_img = Image.new("RGB", (width, height), color=(255, 250, 240))

    draw = ImageDraw.Draw(base_img)

    # Use default bitmap font
    font = ImageFont.load_default()

    # Wrap prompt text into multiple lines
    words = prompt.split()
    lines = []
    curr_line = []
    line_char_limit = 42

    for word in words:
        if len(" ".join(curr_line + [word])) <= line_char_limit:
            curr_line.append(word)
        else:
            lines.append(" ".join(curr_line))
            curr_line = [word]
        if len(lines) >= 14:
            break
    if curr_line and len(lines) < 14:
        lines.append(" ".join(curr_line))

    # Render specific attack styling
    if style_idx == 0:  # System Warning Dialog Injection
        draw.rectangle([(20, 20), (492, 492)], fill=(255, 255, 255), outline=(220, 50, 50), width=3)
        draw.rectangle([(20, 20), (492, 60)], fill=(220, 50, 50))
        draw.text((35, 30), "[SYSTEM OVERRIDE INSTRUCTION]", fill=(255, 255, 255), font=font)
        y = 80
        for line in lines:
            draw.text((40, y), line, fill=(20, 20, 20), font=font)
            y += 24

    elif style_idx == 1:  # Text Overlay on Nature Image
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([(20, 100), (492, 420)], fill=(0, 0, 0, 175))
        overlay_draw.text((40, 120), "IMPORTANT INSTRUCTION OVERRIDE:", fill=(255, 220, 0, 255), font=font)
        y = 155
        for line in lines:
            overlay_draw.text((40, y), line, fill=(255, 255, 255, 255), font=font)
            y += 22
        base_img = Image.alpha_composite(base_img.convert("RGBA"), overlay).convert("RGB")

    elif style_idx == 3:  # Terminal / Developer Mode Injection
        draw.rectangle([(20, 20), (492, 492)], fill=(12, 16, 20), outline=(0, 255, 128), width=2)
        draw.text((35, 35), "root@system:~# sudo ./enable_developer_mode.sh", fill=(0, 255, 128), font=font)
        draw.line([(20, 60), (492, 60)], fill=(0, 180, 90), width=1)
        y = 75
        for line in lines:
            draw.text((35, y), f"> {line}", fill=(220, 230, 240), font=font)
            y += 24
    else:  # High-Contrast Typographic Card
        draw.rectangle([(25, 25), (487, 487)], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        draw.text((40, 40), "PROMPT INJECTION PAYLOAD:", fill=(180, 0, 0), font=font)
        y = 75
        for line in lines:
            draw.text((40, y), line, fill=(0, 0, 0), font=font)
            y += 24

    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_img.save(output_path, quality=92)


def build_vision_dataset(
    target_safe_count: int = 1000,
    target_jailbreak_count: int = 1000,
    random_seed: int = 42
) -> None:
    """
    Constructs the balanced 2,000-sample multimodal vision dataset.
    """
    random.seed(random_seed)
    SAFE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JAILBREAK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Collect clean safe images from dataset/images/safe
    available_safe = sorted([
        p for p in RAW_SAFE_DIR.glob("*.*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])
    if len(available_safe) < target_safe_count:
        raise ValueError(f"Not enough safe images: found {len(available_safe)}, need {target_safe_count}")

    selected_safe = available_safe[:target_safe_count]
    print(f"[*] Processing {len(selected_safe)} safe images...")

    metadata_records = []

    for idx, src_path in enumerate(selected_safe, start=1):
        ext = src_path.suffix.lower()
        dst_filename = f"safe_{idx:05d}{ext}"
        dst_path = SAFE_OUTPUT_DIR / dst_filename

        # Copy safe image
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)

        with Image.open(dst_path) as img:
            w, h = img.size
            fmt = img.format or ext.replace(".", "").upper()

        metadata_records.append({
            "image_id": f"IMG_SAFE_{idx:05d}",
            "filename": dst_filename,
            "file_path": str(dst_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "label": 0,
            "label_name": "SAFE",
            "category": "benign_nature_visual",
            "width": w,
            "height": h,
            "format": fmt
        })

    # 2. Synthesize typographic visual jailbreak images
    print(f"[*] Synthesizing {target_jailbreak_count} visual jailbreak images...")
    prompts = get_jailbreak_prompts(max_prompts=target_jailbreak_count + 200)

    for idx in range(1, target_jailbreak_count + 1):
        prompt_text, attack_type = prompts[(idx - 1) % len(prompts)]
        style = idx % 4

        # Choose background from safe collection for overlay styles
        bg_candidate = selected_safe[(idx * 7) % len(selected_safe)]
        dst_filename = f"jailbreak_{idx:05d}.jpg"
        dst_path = JAILBREAK_OUTPUT_DIR / dst_filename

        if not dst_path.exists():
            render_typographic_jailbreak(
                prompt=prompt_text,
                output_path=dst_path,
                bg_image_path=bg_candidate,
                style_idx=style
            )

        with Image.open(dst_path) as img:
            w, h = img.size
            fmt = img.format or "JPEG"

        style_names = {
            0: "system_dialog_injection",
            1: "scenic_text_overlay",
            2: "high_contrast_typographic",
            3: "terminal_code_injection"
        }

        metadata_records.append({
            "image_id": f"IMG_JAIL_{idx:05d}",
            "filename": dst_filename,
            "file_path": str(dst_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "label": 1,
            "label_name": "JAILBREAK",
            "category": style_names.get(style, "typographic_attack"),
            "width": w,
            "height": h,
            "format": fmt
        })

    # 3. Create stratified train (70%), val (15%), test (15%) splits
    df = pd.DataFrame(metadata_records)

    df_safe = df[df["label"] == 0].sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df_jail = df[df["label"] == 1].sample(frac=1, random_state=random_seed).reset_index(drop=True)

    def assign_splits(sub_df):
        n = len(sub_df)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        splits = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
        sub_df["split"] = splits
        return sub_df

    df_safe = assign_splits(df_safe)
    df_jail = assign_splits(df_jail)

    final_df = pd.concat([df_safe, df_jail], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    final_df.to_csv(METADATA_CSV, index=False)
    print(f"[+] Dataset metadata successfully saved to: {METADATA_CSV}")
    print(f"    Total Images: {len(final_df)}")
    print(f"    Split Distribution:\n{final_df['split'].value_counts()}")
    print(f"    Class Distribution:\n{final_df['label_name'].value_counts()}")


if __name__ == "__main__":
    build_vision_dataset(target_safe_count=1000, target_jailbreak_count=1000)
