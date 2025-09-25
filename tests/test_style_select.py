# tests/test_style_select.py
import numpy as np
from core.style_select import pick_style

def test_pick_style_runs():
    emb = np.random.randn(66).astype(np.float32)
    name, sim = pick_style(emb)
    assert isinstance(name, str)
    assert isinstance(sim, float)
