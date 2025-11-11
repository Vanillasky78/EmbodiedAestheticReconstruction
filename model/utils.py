# model/utils.py
# -----------------------------------------------------------------------------
# Low-level utilities for image I/O and vector math.
# -----------------------------------------------------------------------------

from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Read an image from raw bytes and convert to RGB PIL Image.
    This function raises if the bytes are not a valid image.
    """
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-10) -> np.ndarray:
    """
    L2-normalize an array along the given axis.

    Args:
        x: array of shape (..., D) or (N, D) etc.
        axis: axis to normalize across.
        eps: numerical stability term.

    Returns:
        Array with unit L2 norm along `axis`.
    """
    norm = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    return x / (norm + eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between rows of `a` and a single vector `b`.

    Assumes both `a` and `b` are already L2-normalized.
    Then cosine similarity reduces to a dot product.

    Args:
        a: (N, D) matrix (database embeddings).
        b: (D,) vector (query embedding).

    Returns:
        (N,) similarity scores.
    """
    # If b has shape (D, 1), squeeze to (D,)
    if b.ndim == 2 and b.shape[1] == 1:
        b = b[:, 0]
    return a @ b  # (N, D) · (D,) -> (N,)
