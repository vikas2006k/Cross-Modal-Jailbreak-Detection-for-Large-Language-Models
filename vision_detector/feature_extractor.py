from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Union, Optional
import torch
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModelWithProjection
from PIL import Image

from vision_detector.preprocessing import load_image


DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


class CLIPFeatureExtractor:
    """
    Feature extractor utilizing frozen CLIP ViT-B/32 visual encoder.
    Outputs L2-normalized 512-dimensional visual embeddings.
    """

    _instance: Optional["CLIPFeatureExtractor"] = None

    def __init__(self, model_id: str = DEFAULT_CLIP_MODEL_ID, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_id = model_id

        print(f"[*] Loading CLIP ViT-B/32 feature extractor ({model_id}) on {self.device}...")
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        # Strictly freeze backbone weights
        for param in self.model.parameters():
            param.requires_grad = False

        self.embedding_dim = self.model.config.projection_dim  # 512 for ViT-B/32
        print(f"[+] CLIP Vision Encoder loaded successfully. Embedding dimension: {self.embedding_dim}")

    @classmethod
    def get_instance(cls, model_id: str = DEFAULT_CLIP_MODEL_ID, device: Optional[str] = None) -> "CLIPFeatureExtractor":
        if cls._instance is None:
            cls._instance = cls(model_id=model_id, device=device)
        return cls._instance

    def extract_features(
        self,
        images: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]],
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Extract normalized visual embeddings for single or multiple images.
        Returns tensor of shape (N, 512).
        """
        if not isinstance(images, list):
            image_list = [images]
        else:
            image_list = images

        # Load and convert all images to PIL RGB
        pil_images = []
        for item in image_list:
            if isinstance(item, (str, Path)):
                pil_images.append(load_image(item))
            elif isinstance(item, Image.Image):
                pil_images.append(item.convert("RGB"))
            else:
                raise ValueError(f"Unsupported image type in list: {type(item)}")

        all_embeddings = []

        with torch.inference_mode():
            for i in range(0, len(pil_images), batch_size):
                batch = pil_images[i : i + batch_size]
                inputs = self.processor(images=batch, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)

                # Extract projected image features
                outputs = self.model(pixel_values=pixel_values)
                image_embeds = outputs.image_embeds  # shape: (batch_size, 512)

                # L2 normalize
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                all_embeddings.append(image_embeds.cpu())

        return torch.cat(all_embeddings, dim=0)
