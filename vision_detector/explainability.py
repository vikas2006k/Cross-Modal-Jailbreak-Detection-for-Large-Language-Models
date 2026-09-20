"""
Explainability module for the CMJD Vision Detector.
Extracts Vision Transformer (ViT-B/32) self-attention saliency maps and Grad-CAM activations,
highlighting suspicious typographic prompt injections and adversarial image regions.
Generates publication-ready 300 DPI composite figures.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Tuple, Optional, List, Union
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection, CLIPProcessor

from vision_detector.preprocessing import load_image

FIGURES_DIR = PROJECT_ROOT / "vision_detector" / "metrics" / "figures"
DEFAULT_CLIP_ID = "openai/clip-vit-base-patch32"


class VisionExplainabilityEngine:
    """
    Computes spatial attention saliency maps and Grad-CAM visualizations
    for CLIP ViT-B/32 image inputs.
    """

    def __init__(self, model_id: str = DEFAULT_CLIP_ID, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[*] Initializing VisionExplainabilityEngine on {self.device}...")
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            model_id,
            attn_implementation="eager"
        )
        self.model.to(self.device)
        self.model.eval()

    def generate_attention_map(
        self,
        image: Union[str, Path, Image.Image]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts multi-head self-attention weights from the final ViT layer (Layer 11)
        and computes the CLS-to-patch spatial saliency map.

        Returns:
            rgb_image (np.ndarray): Original image resized to 224x224 (0.0 to 1.0).
            heatmap (np.ndarray): Normalized 2D spatial attention matrix (0.0 to 1.0) resized to 224x224.
        """
        if isinstance(image, (str, Path)):
            pil_img = load_image(image)
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Preprocess for CLIP (224x224)
        inputs = self.processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, output_attentions=True)
            # outputs.attentions is a tuple of 12 tensors: each (batch_size, 12_heads, 50, 50)
            last_layer_attn = outputs.attentions[-1]  # shape: (1, 12, 50, 50)

            # Average attention across all 12 heads
            avg_attn = last_layer_attn.mean(dim=1).squeeze(0)  # shape: (50, 50)

            # CLS token is index 0. Attention from CLS to all 49 visual patches (tokens 1 to 49)
            cls_to_patches = avg_attn[0, 1:]  # shape: (49,)

            # Reshape into 7x7 spatial grid (224 / 32 = 7)
            grid_size = 7
            spatial_attn = cls_to_patches.reshape(1, 1, grid_size, grid_size)

            # Bilinear upsample to 224x224
            upsampled = F.interpolate(spatial_attn, size=(224, 224), mode="bicubic", align_corners=False)
            heatmap = upsampled.squeeze().cpu().numpy()

            # Min-max normalize heatmap to [0, 1]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # Normalized RGB numpy array
        resized_pil = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
        rgb_image = np.array(resized_pil, dtype=np.float32) / 255.0

        return rgb_image, heatmap

    def visualize_saliency(
        self,
        image: Union[str, Path, Image.Image],
        output_path: Path,
        title: str = "Visual Jailbreak Attention Saliency",
        label: str = "JAILBREAK"
    ) -> None:
        """
        Generates and saves a 3-panel 300 DPI publication-grade figure:
        [1] Input Image | [2] Attention Heatmap | [3] Composite Overlay
        """
        rgb_img, heatmap = self.generate_attention_map(image)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

        # Panel 1: Original Image
        axes[0].imshow(rgb_img)
        axes[0].set_title("Input Image", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Saliency Heatmap
        im1 = axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("ViT Layer 11 Attention Map", fontsize=12, fontweight="bold")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: Composite Overlay
        axes[2].imshow(rgb_img)
        axes[2].imshow(heatmap, cmap="jet", alpha=0.55)
        axes[2].set_title(f"Overlay ({label})", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[+] Saliency visualization saved to: {output_path}")
