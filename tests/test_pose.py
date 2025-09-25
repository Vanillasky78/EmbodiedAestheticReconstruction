# tests/test_pose.py
import numpy as np
from PIL import Image
from core.pose import pose_to_embedding

def test_pose_returns_fixed_dim():
    img = Image.new("RGB", (64, 64), color=(200, 200, 200))
    emb = pose_to_embedding(img)
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1
