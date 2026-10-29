"""Adaptive placement: geometry first, then coarse measurements only."""
from dataclasses import dataclass

import numpy as np

from .sampling import lattice_baseline
from .strain import (BASELINE_RADIUS_SPACINGS, MINIMUM_NEIGHBOURS,
                     baseline_neighbour_counts, fixed_radius_strain)
from .tracking import Tracks


@dataclass(frozen=True)
class AdaptivePlan:
    """Geometry fixed before the coarse pass; regions are coarse-point cells."""
    coarse_points: np.ndarray
    coverage_points: np.ndarray
    candidates: np.ndarray
    candidate_regions: np.ndarray
    budget: int
    spacing: float
    baseline_counts: np.ndarray


@dataclass(frozen=True)
class RefinementConfig:
    """Provisional weights; tuning and freezing on dev sequences is separate."""
    gradient_weight: float = 1.0
    coverage_weight: float = 0.1
    smoothing_frames: int = 3

    def __post_init__(self) -> None:
        weights = (self.gradient_weight, self.coverage_weight)
        if not all(np.isfinite(w) and w >= 0 for w in weights) or sum(weights) == 0:
            raise ValueError("refinement weights must be finite, nonnegative and not both zero")
        if not isinstance(self.smoothing_frames, int) or self.smoothing_frames < 1:
            raise ValueError("smoothing_frames must be a positive integer")


@dataclass(frozen=True)
class AdaptivePlacement:
    query_points: np.ndarray
    region_scores: np.ndarray
    refinement_counts: np.ndarray
    visibility_gate_active: bool


def prepare_adaptive(mask: np.ndarray, spacing: float,
                     offset: tuple[float, float]) -> AdaptivePlan:
    """Place a geometry-only floor with six neighbours at each region centre.

    The coarse lattice has spacing 1.5s. Refinement uses its nested square
    sub-lattice at s/2 and the same absolute offset. Infeasible geometries
    raise rather than quietly violate the floor or query budget.
    """
    baseline = lattice_baseline(mask, spacing, offset)
    budget = len(baseline)
    coarse = lattice_baseline(mask, 1.5 * spacing, offset)
    if len(coarse) == 0 or len(coarse) > budget:
        raise ValueError("adaptive geometry needs a nonempty coarse pass within the query budget")
    fine_offset = (offset[0] % (spacing / 2), offset[1] % (spacing / 2))
    fine = lattice_baseline(mask, spacing / 2, fine_offset)
    # Integer sub-lattice indices avoid floating-point duplicates at coarse sites.
    indices = np.rint((fine - offset) / (spacing / 2)).astype(int)
    candidates = fine[np.any(indices % 3 != 0, axis=1)]
    probes = coarse
    radius = BASELINE_RADIUS_SPACINGS * spacing
    counts = baseline_neighbour_counts(coarse, probes, radius)
    available = np.ones(len(candidates), dtype=bool)
    floor: list[np.ndarray] = []
    while np.any(counts < MINIMUM_NEIGHBOURS):
        if len(coarse) + len(floor) >= budget:
            raise ValueError("coverage floor exceeds the query budget")
        target = probes[int(np.argmin(counts))]
        distances = np.sum((candidates - target)**2, axis=1)
        eligible = np.flatnonzero(available & (distances <= radius**2))
        if not len(eligible):
            raise ValueError("sub-lattice cannot satisfy the minimum-neighbour coverage floor")
        deficient = probes[counts < MINIMUM_NEIGHBOURS]
        deficits = MINIMUM_NEIGHBOURS - counts[counts < MINIMUM_NEIGHBOURS]
        # Prefer a query that repairs several deficient locations at once.
        repairs = np.sum((np.sum((candidates[eligible, None] - deficient[None])**2, axis=-1)
                          <= radius**2) * deficits, axis=1)
        chosen = int(eligible[np.argmax(repairs)])
        floor.append(candidates[chosen])
        available[chosen] = False
        counts += np.sum((probes - candidates[chosen])**2, axis=1) <= radius**2
    remaining = candidates[available]
    if len(coarse) + len(floor) + len(remaining) < budget:
        raise ValueError("sub-lattice cannot fill the query budget")
    regions = np.array([np.argmin(np.sum((coarse - point)**2, axis=1))
                        for point in remaining], dtype=int)
    return AdaptivePlan(coarse, np.asarray(floor).reshape(-1, 2), remaining, regions,
                        budget, spacing, baseline_neighbour_counts(baseline, coarse, radius))


def adaptive_queries(plan: AdaptivePlan, tracks: Tracks,
                     config: RefinementConfig = RefinementConfig()) -> AdaptivePlacement:
    """Place queries from coarse trajectories and visibility, without truth.

    Score = (w_gradient * s * ||grad E|| + w_coverage * geometry_deficit)
    * local_visible_fraction. Apply a trailing mean before taking the maximum
    over frames. Coverage is geometric so track loss never earns a bonus.
    """
    reference = plan.coarse_points
    if (tracks.positions.ndim != 3 or tracks.positions.shape[1:] != reference.shape
            or tracks.visibility.shape != tracks.positions.shape[:2]
            or len(tracks.positions) == 0):
        raise ValueError("coarse tracker must return positions and visibility for its query points")
    radius = BASELINE_RADIUS_SPACINGS * plan.spacing
    distances = np.sum((reference[:, None] - reference[None, :])**2, axis=-1)
    neighbourhood = distances <= radius**2
    counts = neighbourhood.sum(axis=1)
    coverage = np.clip(1 - counts / np.maximum(plan.baseline_counts, 1), 0, 1)
    valid = tracks.visibility & np.isfinite(tracks.positions).all(axis=-1)
    gates = valid.astype(float) @ neighbourhood.T / counts
    per_frame = []
    for positions, visibility, gate in zip(tracks.positions, valid, gates):
        strain = fixed_radius_strain(reference, positions, visibility, reference, radius)
        defined = np.isfinite(strain).all(axis=(-1, -2))
        gradient = np.zeros(len(reference))
        for region, centre in enumerate(reference):
            neighbours = np.flatnonzero(neighbourhood[region] & defined)
            if len(neighbours) < MINIMUM_NEIGHBOURS:
                continue
            design = np.column_stack((reference[neighbours] - centre, np.ones(len(neighbours))))
            coefficients, _, rank, _ = np.linalg.lstsq(design, strain[neighbours].reshape(-1, 4), rcond=None)
            if rank == 3:
                gradient[region] = np.linalg.norm(coefficients[:2])
        per_frame.append((config.gradient_weight * plan.spacing * gradient
                          + config.coverage_weight * coverage) * gate)
    scores = np.asarray(per_frame)
    smoothed = np.stack([scores[max(0, frame - config.smoothing_frames + 1):frame + 1].mean(axis=0)
                         for frame in range(len(scores))])
    region_scores = smoothed.max(axis=0)
    needed = plan.budget - len(plan.coarse_points) - len(plan.coverage_points)
    capacities = np.bincount(plan.candidate_regions, minlength=len(reference))
    allocations = _proportional_counts(region_scores, capacities, needed)
    initial = np.vstack((plan.coarse_points, plan.coverage_points))
    selected: list[int] = []
    for region, count in enumerate(allocations):
        group = np.flatnonzero(plan.candidate_regions == region)
        # Spread each region's allocation, including the coarse/floor geometry.
        separation = np.min(np.sum((plan.candidates[group, None] - initial[None])**2, axis=-1), axis=1)
        for _ in range(count):
            local = int(np.argmax(separation))
            chosen = int(group[local])
            selected.append(chosen)
            separation = np.minimum(separation, np.sum((plan.candidates[group] - plan.candidates[chosen])**2, axis=1))
            separation[local] = -1
    queries = np.vstack((initial, plan.candidates[selected]))
    return AdaptivePlacement(queries, region_scores, allocations, bool(np.any(gates < 1)))


def _proportional_counts(scores: np.ndarray, capacities: np.ndarray, budget: int) -> np.ndarray:
    """Largest remainders with capacity caps; zero scores share the remainder."""
    allocated = np.zeros(len(scores), dtype=int)
    while budget:
        available = capacities > allocated
        weights = np.where(available, scores, 0)
        if not weights.any():
            weights = available.astype(float)
        quotas = budget * weights / weights.sum()
        capped = available & (quotas >= capacities - allocated)
        if capped.any():
            # Redistribute capped shares before performing any integer rounding.
            budget -= int((capacities[capped] - allocated[capped]).sum())
            allocated[capped] = capacities[capped]
            continue
        whole = np.floor(quotas).astype(int)
        remaining = budget - int(whole.sum())
        eligible = np.flatnonzero(available)
        order = eligible[np.argsort(-(quotas[eligible] - whole[eligible]), kind="stable")]
        allocated += whole
        allocated[order[:remaining]] += 1
        break
    return allocated
