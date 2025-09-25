# tests/test_personalise.py
import numpy as np
from core.personalise import personality_vector, fuse_embeddings

def test_fuse_normalises():
    pose = np.random.randn(66).astype(np.float32)
    pers = np.array([0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    fused = fuse_embeddings(pose, pers)
    assert np.isclose(np.linalg.norm(fused), 1.0, atol=1e-3)
