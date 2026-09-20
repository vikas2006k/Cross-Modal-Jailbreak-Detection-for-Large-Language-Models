"""
Image preprocessing and normalization utilities for the CMJD Vision Detector.
Handles image loading, format normalization, and PyTorch / CLIP tensor transformation.
"""

from pathlib import Path
from typing import Union, Tuple, Optional
import torch
from torchvision import transforms
from PIL import Image, ImageOps


# Standard CLIP normalization constants (OpenAI CLIP ViT-B/32)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_SIZE = 224


def get_vision_transforms(is_training: bool = False) -> transforms.Compose:
    """
    Returns standard torchvision transformation pipeline compatible with CLIP ViT-B/32.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Loads an image from disk, normalizes EXIF orientation, and converts to RGB.
    Supports PNG, JPG, JPEG, and WEBP formats.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at path: {path}")

    try:
        with Image.open(path) as img:
            # Transpose based on EXIF orientation tag if present
            img = ImageOps.exif_transpose(img)
            # Ensure consistent 3-channel RGB format
            return img.convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load and process image {path}: {str(e)}")
