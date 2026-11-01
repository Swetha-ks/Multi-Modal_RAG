"""
embedder.py — CLIP Multimodal Embedding Module
Converts product images and text into 512D vectors
"""

import torch
import open_clip
from PIL import Image
import requests
from io import BytesIO
import numpy as np


class MultimodalEmbedder:

    def __init__(self, model_name="ViT-B-32", pretrained="openai"):
        print("Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        print(f"✅ CLIP loaded on {self.device.upper()}")

    def embed_text(self, text: str) -> list:
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            features = self.model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0].tolist()

    def embed_image(self, image: Image.Image) -> list:
        with torch.no_grad():
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            features = self.model.encode_image(tensor)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0].tolist()

    def embed_image_from_url(self, url: str) -> list:
        try:
            r = requests.get(url, timeout=10)
            img = Image.open(BytesIO(r.content)).convert("RGB")
            return self.embed_image(img)
        except:
            return None

    def embed_combined(self, text: str, image: Image.Image = None) -> list:
        """Combine text + image into one vector (KEY INNOVATION)"""
        tv = np.array(self.embed_text(text))
        if image is not None:
            iv = np.array(self.embed_image(image))
            combined = (tv + iv) / 2
            return (combined / np.linalg.norm(combined)).tolist()
        return tv.tolist()
