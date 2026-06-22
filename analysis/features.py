"""Fixed-width feature encodings for variable-height (tokens, 2816) matrices."""
from __future__ import annotations

import numpy as np

from .load_data import Sample


def pooled_stats(matrix: np.ndarray) -> np.ndarray:
    """Statistical pooling across the token axis -> (5 * width,) vector.

    mean / max / min / std / median per neuron, concatenated. Token-count
    invariant, which matters here since height varies (6-8 tokens).
    """
    return np.concatenate([
        matrix.mean(axis=0),
        matrix.max(axis=0),
        matrix.min(axis=0),
        matrix.std(axis=0),
        np.median(matrix, axis=0),
    ])


def last_token(matrix: np.ndarray) -> np.ndarray:
    """Representation of the final token position (next-token prediction site)."""
    return matrix[-1]


def first_token(matrix: np.ndarray) -> np.ndarray:
    return matrix[0]


def flatten_padded(matrix: np.ndarray, max_height: int) -> np.ndarray:
    """Zero-pad/truncate to max_height rows, then flatten."""
    h, w = matrix.shape
    if h >= max_height:
        out = matrix[:max_height]
    else:
        out = np.vstack([matrix, np.zeros((max_height - h, w))])
    return out.flatten()


def build_feature_matrix(samples: list[Sample], method: str = "pooled") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (X, y, sample_names). y: 1 = math, 0 = others."""
    names = [s.name for s in samples]
    y = np.array([1 if s.label == "math" else 0 for s in samples])

    if method == "pooled":
        X = np.stack([pooled_stats(s.matrix) for s in samples])
    elif method == "last_token":
        X = np.stack([last_token(s.matrix) for s in samples])
    elif method == "first_token":
        X = np.stack([first_token(s.matrix) for s in samples])
    elif method == "flatten_padded":
        max_h = max(s.matrix.shape[0] for s in samples)
        X = np.stack([flatten_padded(s.matrix, max_h) for s in samples])
    else:
        raise ValueError(f"unknown method: {method}")

    return X, y, names
