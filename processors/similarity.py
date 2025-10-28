"""
processors/similarity.py
----------------------------------------
Utilities for similarity scoring between vectors.
Currently we use cosine similarity.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_scores(query_vec, candidate_matrix):
    """
    Compute cosine similarity between a single query vector
    and a matrix of candidate vectors.

    Args:
        query_vec (np.ndarray): shape (1, D) or (D,)
        candidate_matrix (np.ndarray): shape (N, D)

    Returns:
        scores (np.ndarray): shape (N,), higher = more similar
    """

    # ensure proper shape
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)

    scores = cosine_similarity(query_vec, candidate_matrix)[0]
    return scores


def top_k_indices(scores, k=3):
    """
    Return the indices of the top-k highest scores.
    Args:
        scores (np.ndarray): shape (N,)
        k (int): number of top items to return
    Returns:
        idxs (list[int]): indices sorted by score desc
    """
    sorted_idx = np.argsort(scores)[::-1]
    return sorted_idx[:k]
