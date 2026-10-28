"""Shared local affine strain estimators in the reference configuration."""
import numpy as np

MINIMUM_NEIGHBOURS = 6
CONDITION_LIMIT = 1e6
BASELINE_RADIUS_SPACINGS = 2.5
# Exact square-lattice count in a 2.5-spacing disk centred on a lattice site,
# including the centre. Unlike round(pi * 2.5**2), this preserves the stencil.
BASELINE_K = 21


def baseline_neighbour_counts(baseline: np.ndarray, evaluation_grid: np.ndarray,
                              radius: float) -> np.ndarray:
    """Calibrate k from the intact lattice baseline's fixed-radius stencil.

    Freeze these counts across placement methods and frames, before visibility
    filtering. This accounts for lattice offset, evaluation phase and mask edges.
    Counts below MINIMUM_NEIGHBOURS leave the primary estimate undefined too.
    """
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    points = baseline[np.isfinite(baseline).all(axis=1)]
    return np.array([np.count_nonzero(np.sum((points - position)**2, axis=1) <= radius**2)
                     for position in evaluation_grid], dtype=int)


def k_nearest_strain(reference: np.ndarray, tracked: np.ndarray,
                     visibility: np.ndarray, evaluation_grid: np.ndarray,
                     k: int | np.ndarray = BASELINE_K) -> np.ndarray:
    """Green–Lagrange E from the k nearest finite visible query points.

    The default k matches a radius of 2.5 baseline spacings at an interior
    square-lattice site. For an independent evaluation grid, pass counts from
    baseline_neighbour_counts, frozen across placement methods and frames.
    Extra local density shrinks the k-nearest support.
    """
    counts = np.asarray(k)
    if counts.shape not in ((), (len(evaluation_grid),)) or counts.dtype.kind not in "iu" or np.any(counts < 0):
        raise ValueError("k must be a nonnegative integer or one integer per evaluation position")
    return _local_affine_strain(reference, tracked, visibility, evaluation_grid, k=k)


def fixed_radius_strain(reference: np.ndarray, tracked: np.ndarray,
                        visibility: np.ndarray, evaluation_grid: np.ndarray,
                        radius: float) -> np.ndarray:
    """Green–Lagrange E from all finite visible query points within radius.

    Use radius = BASELINE_RADIUS_SPACINGS * baseline spacing, even when local
    query density increases. Distances are in the reference configuration.
    """
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    return _local_affine_strain(reference, tracked, visibility, evaluation_grid, radius=radius)


def _local_affine_strain(reference: np.ndarray, tracked: np.ndarray,
                         visibility: np.ndarray, evaluation_grid: np.ndarray,
                         *, k: int | np.ndarray | None = None, radius: float | None = None) -> np.ndarray:
    """Unweighted affine displacement fit with shared observability rules.

    Fewer than MINIMUM_NEIGHBOURS, rank deficiency, or a design condition
    number above CONDITION_LIMIT leave NaN for explicit coverage reporting.
    """
    valid = visibility & np.isfinite(reference).all(axis=1) & np.isfinite(tracked).all(axis=1)
    points = reference[valid]
    displacement = tracked[valid] - points
    strain = np.full((len(evaluation_grid), 2, 2), np.nan)
    if len(points) < MINIMUM_NEIGHBOURS:
        return strain
    counts = np.broadcast_to(k, (len(evaluation_grid),)) if k is not None else None
    for index, position in enumerate(evaluation_grid):
        distances = np.sum((points - position)**2, axis=1)
        if counts is not None:
            count = min(int(counts[index]), len(points))
            if count < MINIMUM_NEIGHBOURS:
                continue
            neighbours = np.argpartition(distances, count - 1)[:count]
        else:
            assert radius is not None
            neighbours = np.flatnonzero(distances <= radius**2)
            count = len(neighbours)
        if count < MINIMUM_NEIGHBOURS:
            continue
        design = np.column_stack((points[neighbours] - position, np.ones(count)))
        if np.linalg.cond(design) > CONDITION_LIMIT:
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
