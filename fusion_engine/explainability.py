"""
Multimodal Explainability Engine for Cross-Modal Jailbreak Detection.
Combines:
1. OCR Bounding Box Localization
2. Vision Transformer (ViT-B/32) Layer 11 Attention Saliency Heatmap
3. DistilBERT Token-Level Attribution for Extracted Text
4. Modality Contribution Bar / Donut Chart (Vision % vs Text %)
Generates publication-quality 300 DPI multi-panel figures.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, Optional, List, Union, Tuple
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from vision_detector.explainability import VisionExplainabilityEngine
from backend.inference import TextInferenceEngine
from transformers import AutoModelForSequenceClassification

FIGURES_DIR = PROJECT_ROOT / "fusion_engine" / "metrics" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


class MultimodalExplainabilityEngine:
    """
    Unified multimodal explainability generator.
    Produces comprehensive visual attribution, OCR localization, token saliency,
    and cross-modal risk allocation diagrams.
    """

    _instance: Optional["MultimodalExplainabilityEngine"] = None

    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[*] Initializing MultimodalExplainabilityEngine on {self.device}...")
        self.vision_explainer = VisionExplainabilityEngine(device=str(self.device))
        self.text_engine = TextInferenceEngine.get_instance()
        self.eager_text_model = AutoModelForSequenceClassification.from_pretrained(
            self.text_engine.model_path,
            attn_implementation="eager"
        ).to(self.device)
        self.eager_text_model.eval()
        print("[+] MultimodalExplainabilityEngine ready.")


    @classmethod
    def get_instance(cls, device: Optional[str] = None) -> "MultimodalExplainabilityEngine":
        if cls._instance is None:
            cls._instance = cls(device=device)
        return cls._instance

    def draw_ocr_boxes(
        self,
        image: Image.Image,
        bounding_boxes: List[Any],
        is_jailbreak: bool
    ) -> Image.Image:
        """Draws OCR bounding boxes on image copy."""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        color = "#e63946" if is_jailbreak else "#2a9d8f"  # Red for jailbreak, teal for safe

        for box in bounding_boxes:
            try:
                # Box format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                coords = [(int(p[0]), int(p[1])) for p in box]
                draw.polygon(coords, outline=color, width=4)
            except Exception:
                pass

        return img_copy

    def compute_text_token_salience(self, text: str) -> Tuple[List[str], List[float]]:
        """
        Extracts token-level attribution from DistilBERT's final self-attention layer.
        Computes attention from [CLS] to all sequence tokens.
        """
        if not text.strip():
            return ["[EMPTY]"], [1.0]

        tokenizer = self.text_engine.tokenizer
        model = getattr(self, "eager_text_model", self.text_engine.model)

        inputs = tokenizer(
            text,
            max_length=128,
            padding=False,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            if outputs.attentions and len(outputs.attentions) > 0:
                last_attn = outputs.attentions[-1]
                avg_attn = last_attn.mean(dim=1).squeeze(0)
                cls_attn = avg_attn[0].cpu().numpy()
            else:
                seq_len = inputs["input_ids"].shape[1]
                cls_attn = np.ones(seq_len) / seq_len


        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))
        # Remove special tokens for clean display if desired, or keep for context
        salience = cls_attn.tolist()

        # Normalize to sum to 1.0
        total = sum(salience) or 1.0
        norm_salience = [s / total for s in salience]

        # Truncate to top 15 tokens if too long
        if len(tokens) > 15:
            sorted_indices = sorted(range(len(norm_salience)), key=lambda i: norm_salience[i], reverse=True)[:15]
            sorted_indices = sorted(sorted_indices)  # preserve word order
            tokens = [tokens[i] for i in sorted_indices]
            norm_salience = [norm_salience[i] for i in sorted_indices]

        return tokens, norm_salience

    def generate_multimodal_explanation(
        self,
        image: Image.Image,
        ocr_result: Dict[str, Any],
        vision_result: Dict[str, Any],
        text_result: Dict[str, Any],
        fusion_result: Any,
        output_prefix: str = "multimodal_explanation"
    ) -> Dict[str, str]:
        """
        Generates a 4-panel publication-grade 300 DPI composite explainability figure:
        [1] OCR Bounding Boxes on Input Image
        [2] Vision Transformer Layer 11 Attention Heatmap Overlay
        [3] DistilBERT Token Attribution (Saliency) for Extracted Text
        [4] Cross-Modal Modality Contribution Breakdown
        """
        fig = plt.figure(figsize=(18, 12), dpi=300)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

        # -------------------------------------------------------------
        # Panel 1: OCR Bounding Boxes Overlay
        # -------------------------------------------------------------
        ax1 = fig.add_subplot(gs[0, 0])
        ocr_boxes = ocr_result.get("bounding_boxes", [])
        is_jailbreak = (fusion_result.prediction == "JAILBREAK")
        img_with_boxes = self.draw_ocr_boxes(image, ocr_boxes, is_jailbreak)
        ax1.imshow(img_with_boxes)
        num_boxes = len(ocr_boxes)
        ax1.set_title(
            f"Panel A: OCR Text Detection ({num_boxes} Region{'s' if num_boxes != 1 else ''})",
            fontsize=13, fontweight="bold", pad=10
        )
        ax1.axis("off")

        # -------------------------------------------------------------
        # Panel 2: ViT Attention Saliency Overlay
        # -------------------------------------------------------------
        ax2 = fig.add_subplot(gs[0, 1])
        rgb_img, heatmap = self.vision_explainer.generate_attention_map(image)
        ax2.imshow(rgb_img)
        im2 = ax2.imshow(heatmap, cmap="jet", alpha=0.55)
        ax2.set_title(
            "Panel B: Vision Transformer Attention Map (ViT-B/32)",
            fontsize=13, fontweight="bold", pad=10
        )
        ax2.axis("off")
        cbar = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)

        # -------------------------------------------------------------
        # Panel 3: DistilBERT Token Attribution Bar Chart
        # -------------------------------------------------------------
        ax3 = fig.add_subplot(gs[1, 0])
        text_evaluated = text_result.get("prompt_evaluated", "") or ocr_result.get("extracted_text", "")
        tokens, salience = self.compute_text_token_salience(text_evaluated)

        y_pos = np.arange(len(tokens))
        colors = plt.cm.viridis(np.array(salience) / (max(salience) if max(salience) > 0 else 1.0))
        bars = ax3.barh(y_pos, salience, color=colors, edgecolor="black", linewidth=0.6, height=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(tokens, fontsize=10, fontfamily="monospace")
        ax3.invert_yaxis()  # top-down reading order
        ax3.set_xlabel("Normalized Self-Attention Weight", fontsize=11, fontweight="bold")
        ax3.set_title(
            "Panel C: DistilBERT Token Attribution (Extracted Text)",
            fontsize=13, fontweight="bold", pad=10
        )
        ax3.grid(axis="x", linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # Panel 4: Modality Contribution Breakdown
        # -------------------------------------------------------------
        ax4 = fig.add_subplot(gs[1, 1])
        v_risk = float(vision_result.get("risk_score", 0.0))
        t_risk = float(text_result.get("risk_score", 0.0))
        f_risk = float(fusion_result.risk_score)

        # Calculate relative weights based on fusion resolution
        total_risk = v_risk + t_risk
        if total_risk > 0:
            v_pct = (v_risk / total_risk) * 100.0
            t_pct = (t_risk / total_risk) * 100.0
        else:
            v_pct, t_pct = 50.0, 50.0

        modalities = ["Vision Branch", "Text Branch"]
        pcts = [v_pct, t_pct]
        bar_colors = ["#3a86ff", "#ff006e"]

        y_indices = [0, 1]
        bars4 = ax4.barh(y_indices, pcts, color=bar_colors, edgecolor="black", linewidth=0.8, height=0.55)
        ax4.set_yticks(y_indices)
        ax4.set_yticklabels(modalities, fontsize=12, fontweight="bold")
        ax4.set_xlim(0, 115)
        ax4.set_xlabel("Relative Modality Contribution (%)", fontsize=11, fontweight="bold")
        ax4.set_title(
            f"Panel D: Cross-Modal Contribution (Fused Risk: {f_risk:.1f}%)",
            fontsize=13, fontweight="bold", pad=10
        )
        ax4.grid(axis="x", linestyle="--", alpha=0.5)

        for bar, pct, raw_risk in zip(bars4, pcts, [v_risk, t_risk]):
            ax4.text(
                pct + 2, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}% (Raw: {raw_risk:.1f}%)",
                va="center", ha="left", fontsize=10, fontweight="bold"
            )

        # Super title with prediction verdict
        verdict_color = "#e63946" if is_jailbreak else "#2a9d8f"
        plt.suptitle(
            f"Cross-Modal Jailbreak Attribution Report | Verdict: {fusion_result.prediction} "
            f"({fusion_result.attack_category}) | Fused Risk: {f_risk:.1f}%",
            fontsize=16, fontweight="bold", y=0.98, color=verdict_color
        )

        composite_filename = f"{output_prefix}.png"
        composite_path = FIGURES_DIR / composite_filename
        fig.savefig(composite_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"[+] Multimodal explanation figure saved to: {composite_path}")

        return {
            "composite_explanation": str(composite_path)
        }
