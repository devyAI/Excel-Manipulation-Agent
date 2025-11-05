from __future__ import annotations

import numpy as np


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between rows of a and rows of b."""
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.dot(a_norm, b_norm.T)


def top_k_similar(a_vec: np.ndarray, candidates: np.ndarray, k: int = 3):
    sims = cosine_similarity_matrix(a_vec, candidates)[0]
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]


