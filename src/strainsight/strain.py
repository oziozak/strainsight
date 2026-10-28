"""Simple two-point strain baseline, not a full spatial strain estimator."""
import numpy as np


def axial_engineering_strain(initial_xy: np.ndarray, tracked_xy: np.ndarray) -> float:
    """Relative change in x-distance between two material points.

    Assumes axial loading aligned with the x axis, with no out-of-plane motion.
    For rotated specimens use a specimen-aligned coordinate frame.
    """
    a = np.asarray(initial_xy, dtype=float)
    b = np.asarray(tracked_xy, dtype=float)
    if a.shape != (2, 2) or b.shape != (2, 2):
        raise ValueError("Expected two points as arrays of shape (2, 2)")
    length0 = a[1, 0] - a[0, 0]
    if np.isclose(length0, 0):
        raise ValueError("Initial axial separation must be nonzero")
    return float((b[1, 0] - b[0, 0] - length0) / length0)
