"""Spatially spread query initialization constrained to a binary specimen mask."""
import numpy as np


def sample_masked_grid(mask: np.ndarray, budget: int) -> np.ndarray:
    """Return up to `budget` points (x, y), all within mask.

    Uses a farthest-point heuristic on valid pixels to encourage broad coverage.
    This simple implementation is intended for small images / initial experiments.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if budget <= 0:
        raise ValueError("budget must be positive")
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("mask contains no foreground pixels")
    candidates = np.column_stack([xs, ys]).astype(float)
    n = min(budget, len(candidates))
    selected = np.empty((n, 2), dtype=float)
    center = candidates.mean(axis=0)
    first = np.argmin(np.sum((candidates - center) ** 2, axis=1))
    selected[0] = candidates[first]
    d2 = np.sum((candidates - selected[0]) ** 2, axis=1)
    d2[first] = -1.0
    for i in range(1, n):
        idx = int(np.argmax(d2))
        selected[i] = candidates[idx]
        d2 = np.minimum(d2, np.sum((candidates - candidates[idx]) ** 2, axis=1))
        d2[idx] = -1.0
    return selected
