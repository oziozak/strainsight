"""Simple two-point strain baseline, not a full spatial strain estimator."""
import numpy as np


def k_nearest_strain(reference: np.ndarray, tracked: np.ndarray,
                     visibility: np.ndarray, evaluation_grid: np.ndarray,
                     k: int = 20) -> np.ndarray:
    """Unweighted local affine displacement fit, returning Green–Lagrange E.

    At least six finite visible neighbours and a design condition number below
    1e6 are required. Undefined positions remain NaN for coverage reporting.
    Neighbour distances are measured in the reference configuration.
    """
    valid = visibility & np.isfinite(tracked).all(axis=1)
    points = reference[valid]
    displacement = tracked[valid] - points
    strain = np.full((len(evaluation_grid), 2, 2), np.nan)
    count = min(k, len(points))
    if count < 6:
        return strain
    for index, position in enumerate(evaluation_grid):
        distances = np.sum((points - position)**2, axis=1)
        neighbours = np.argpartition(distances, count - 1)[:count]
        design = np.column_stack((points[neighbours] - position, np.ones(count)))
        if np.linalg.cond(design) > 1e6:
            continue
        coefficients, _, rank, _ = np.linalg.lstsq(design, displacement[neighbours], rcond=None)
        if rank < 3:
            continue
        gradient = np.eye(2) + coefficients[:2].T
        strain[index] = (gradient.T @ gradient - np.eye(2)) / 2
    return strain


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
