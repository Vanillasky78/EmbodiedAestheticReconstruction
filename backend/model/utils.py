# model/utils.py
import torch
import clip
import numpy as np
from PIL import Image

def load_clip_model(model_name: str, pretrained: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model, preprocess = clip.load(name=model_name, device=device, jit=False)
    model.eval()
    return model, preprocess

def preprocess_image(image: Image.Image, preprocess):
    return preprocess(image).unsqueeze(0)

def cosine_similarity(a: np.ndarray, b: np.ndarray):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T).squeeze()
