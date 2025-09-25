import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

STYLE_NAMES = [
    "Organic Flux",
    "Geo Deconstruct",
    "Noisy Bloom",
    "Monochrome Ritual",
    "Chromatic Drift",
]

def _seed_from_name(name: str) -> int:
    return abs(hash(name)) % (2**32 - 1)

def _style_proto(name: str, dim: int) -> np.ndarray:
    rng = np.random.default_rng(_seed_from_name(name))
    v = rng.normal(0, 1, size=(dim,))
    return v / (np.linalg.norm(v) + 1e-8)

def pick_style(fused_emb: np.ndarray):
    if fused_emb.size == 0:
        return STYLE_NAMES[0], 0.0
    vecs = np.stack([_style_proto(s, fused_emb.size) for s in STYLE_NAMES], 0)
    sims = cosine_similarity(vecs, fused_emb.reshape(1, -1)).reshape(-1)
    i = int(np.argmax(sims))
    return STYLE_NAMES[i], float(sims[i])
