import numpy as np
import pytest

from strainsight.adaptive import RefinementConfig, adaptive_queries, prepare_adaptive
from strainsight.specimen import dog_bone, points_in_mask
from strainsight.tracking import Tracks


@pytest.mark.parametrize("visible", [False, True])
def test_placement_spends_exact_budget_without_being_given_ground_truth(visible):
    mask = dog_bone().sampling_region
    plan = prepare_adaptive(mask, 8, (7., 6.))
    # No frames, sequence, motion, truth or tracker is available to placement.
    positions = np.broadcast_to(plan.coarse_points, (4, *plan.coarse_points.shape)).copy()
    visibility = np.full(positions.shape[:2], visible)
    if not visible:
        positions[:] = np.nan
    placement = adaptive_queries(plan, Tracks(positions, visibility),
                                 RefinementConfig(coverage_weight=0))
    assert len(placement.query_points) == plan.budget
    assert len(np.unique(placement.query_points, axis=0)) == plan.budget
    assert points_in_mask(mask, placement.query_points).all()
    np.testing.assert_allclose(placement.region_scores, 0, atol=1e-12)
    assert np.isfinite(placement.region_scores).all()
    assert placement.visibility_gate_active is (not visible)
    assert placement.refinement_counts.sum() == plan.budget - len(plan.coarse_points) - len(plan.coverage_points)


def test_proportional_allocation_redistributes_when_a_region_fills_its_sub_lattice():
    plan = prepare_adaptive(dog_bone().sampling_region, 8, (0., 0.))
    reference = plan.coarse_points
    positions = np.broadcast_to(reference, (4, *reference.shape)).copy()
    positions[1:, :, 0] += 6 * np.exp(-(reference[:, 0] - 256)**2 / 288)
    placement = adaptive_queries(plan, Tracks(positions, np.ones(positions.shape[:2], dtype=bool)),
                                 RefinementConfig(coverage_weight=0))
    capacity = np.bincount(plan.candidate_regions, minlength=len(reference))
    quota = placement.refinement_counts.sum() * placement.region_scores / placement.region_scores.sum()
    assert np.any(quota > capacity)
    assert np.all(placement.refinement_counts <= capacity)
    assert np.any(placement.refinement_counts == capacity)
    assert len(np.unique(placement.query_points, axis=0)) == plan.budget


def test_infeasible_geometry_rejects_coverage_instead_of_exceeding_budget():
    with pytest.raises(ValueError, match="coverage floor exceeds the query budget"):
        prepare_adaptive(np.ones((3, 3), dtype=bool), 2, (0., 0.))


def test_proportional_rounding_happens_after_capacity_redistribution():
    plan = prepare_adaptive(dog_bone().sampling_region, 8., (1.2, 2.3))
    reference = plan.coarse_points
    positions = np.broadcast_to(reference, (4, *reference.shape)).copy()
    positions[1:, :, 0] += 6 * np.exp(-(reference[:, 0] - 256)**2 / (2 * 8**2))
    placement = adaptive_queries(plan, Tracks(positions, np.ones(positions.shape[:2], dtype=bool)),
                                 RefinementConfig(coverage_weight=0))
    # Independent worked allocation: constrained quota 7.121063 rounds to 7.
    region = np.isclose(reference, [229.2, 170.3]).all(axis=1)
    assert placement.refinement_counts[region].tolist() == [7]
    capacities = np.bincount(plan.candidate_regions, minlength=len(reference))
    saturated = placement.refinement_counts == capacities
    remaining = placement.refinement_counts[~saturated].sum()
    # Unsaturated regions must share the remaining budget in score proportions,
    # with less than one query of integer-rounding error at every region.
    quotas = remaining * placement.region_scores[~saturated] / placement.region_scores[~saturated].sum()
    assert np.max(np.abs(placement.refinement_counts[~saturated] - quotas)) < 1
