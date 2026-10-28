import numpy as np
import pytest

from strainsight.strain import (MINIMUM_NEIGHBOURS, baseline_neighbour_counts,
                               fixed_radius_strain, k_nearest_strain)


@pytest.mark.parametrize("variant", ["k_nearest", "fixed_radius"])
def test_affine_displacement_recovers_green_lagrange_strain(variant):
    x, y = np.meshgrid(np.arange(-4, 5, dtype=float), np.arange(-4, 5, dtype=float))
    reference = np.column_stack((x.ravel(), y.ravel()))
    # F=[[1.1, .2], [0, 1]], plus translation: E=[[.105, .11], [.11, .02]].
    tracked = reference @ np.array([[1.1, 0], [0.2, 1]]) + [3, -7]
    evaluation = np.array([[0., 0.], [0.3, -0.7], [1., 2.]])
    visibility = np.ones(len(reference), dtype=bool)
    if variant == "k_nearest":
        strain = k_nearest_strain(reference, tracked, visibility, evaluation)
    else:
        strain = fixed_radius_strain(reference, tracked, visibility, evaluation, radius=2.5)
    np.testing.assert_allclose(strain, np.broadcast_to([[0.105, 0.11], [0.11, 0.02]],
                                                     strain.shape), atol=1e-12)


def test_variants_coincide_at_interior_lattice_sites_for_non_affine_displacement():
    x, y = np.meshgrid(np.arange(-8, 9, dtype=float), np.arange(-8, 9, dtype=float))
    reference = np.column_stack((x.ravel(), y.ravel()))
    tracked = reference + np.column_stack((0.1 * reference[:, 0]**3,
                                          0.2 * reference[:, 1]**2))
    visibility = np.ones(len(reference), dtype=bool)
    evaluation = np.array([[0., 0.], [1., -2.], [-3., 3.]])
    nearest = k_nearest_strain(reference, tracked, visibility, evaluation)
    radius = fixed_radius_strain(reference, tracked, visibility, evaluation, radius=2.5)
    assert np.isfinite(nearest).all()
    np.testing.assert_allclose(nearest, radius, atol=1e-12)


@pytest.mark.parametrize("offset", [(0., 0.), (.3, .7)])
def test_baseline_calibration_matches_radius_on_independent_grid_and_at_edges(offset):
    x, y = np.meshgrid(np.arange(-8., 9.) + offset[0], np.arange(-8., 9.) + offset[1])
    reference = np.column_stack((x.ravel(), y.ravel()))
    tracked = reference.copy()
    tracked[:, 0] += .01 * reference[:, 0]**3
    ex, ey = np.meshgrid(np.arange(-10., 11., .5), [-9., -.5, 0., .5, 9.])
    evaluation = np.column_stack((ex.ravel(), ey.ravel()))
    counts = baseline_neighbour_counts(reference, evaluation, radius=2.5)
    visibility = np.ones(len(reference), dtype=bool)
    nearest = k_nearest_strain(reference, tracked, visibility, evaluation, k=counts)
    radius = fixed_radius_strain(reference, tracked, visibility, evaluation, radius=2.5)
    assert np.isfinite(radius).any()
    assert np.isnan(radius).any()
    np.testing.assert_allclose(nearest, radius, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("variant", ["k_nearest", "fixed_radius"])
def test_gaussian_band_has_small_nonzero_error_that_decreases_with_query_density(variant):
    # Analytic displacement u_x = 2 exp(-x²/(2*12²)), u_y = 0.
    # Differentiate it directly for truth; do not approximate the derivative.
    ex, ey = np.meshgrid(np.arange(-20., 21., 2.), [-4., 0., 4.])
    evaluation = np.column_stack((ex.ravel(), ey.ravel()))
    derivative = -evaluation[:, 0] / 72 * np.exp(-evaluation[:, 0]**2 / 288)
    truth = np.zeros((len(evaluation), 2, 2))
    truth[:, 0, 0] = derivative + derivative**2 / 2
    errors = []
    counts = None
    for spacing in (4., 2., 1.):
        x, y = np.meshgrid(np.arange(-40., 41., spacing), np.arange(-16., 17., spacing))
        reference = np.column_stack((x.ravel(), y.ravel()))
        tracked = reference.copy()
        tracked[:, 0] += 2 * np.exp(-reference[:, 0]**2 / 288)
        visibility = np.ones(len(reference), dtype=bool)
        if variant == "k_nearest":
            # Calibrate on the coarse baseline once, then freeze k as local
            # density grows, as it would for refinement points at this budget.
            if counts is None:
                counts = baseline_neighbour_counts(reference, evaluation, radius=2.5 * spacing)
            strain = k_nearest_strain(reference, tracked, visibility, evaluation, k=counts)
        else:
            # Along the budget curve the baseline spacing decreases too.
            # At a single budget fixed-radius retains its physical radius.
            strain = fixed_radius_strain(reference, tracked, visibility, evaluation,
                                         radius=2.5 * spacing)
        assert np.isfinite(strain).all()
        errors.append(float(np.mean(np.abs(strain - truth))))
    assert all(1e-8 < error < 0.01 for error in errors)
    assert errors[1] < errors[0] / 2
    assert errors[2] < errors[1] / 2


@pytest.mark.parametrize("variant", ["k_nearest", "fixed_radius"])
@pytest.mark.parametrize("support", ["six", "five", "collinear", "ill_conditioned", "empty"])
def test_variants_share_minimum_neighbours_and_conditioning_rules(variant, support):
    assert MINIMUM_NEIGHBOURS == 6
    reference = np.array([[-1., -.5], [0., -.5], [1., -.5],
                          [-1., .5], [0., .5], [1., .5]])
    if support == "five":
        reference = reference[:5]
    elif support == "collinear":
        reference[:, 1] = 0
    elif support == "ill_conditioned":
        # Full rank, but condition number is above 10^6.
        reference[:, 1] *= 1e-8
    elif support == "empty":
        reference = reference[:0]
    tracked = reference * [1.1, 1.]
    visibility = np.ones(len(reference), dtype=bool)
    evaluation = np.array([[0., 0.]])
    if variant == "k_nearest":
        strain = k_nearest_strain(reference, tracked, visibility, evaluation)
    else:
        strain = fixed_radius_strain(reference, tracked, visibility, evaluation, radius=2.5)
    if support == "six":
        np.testing.assert_allclose(strain, np.array([[[.105, 0], [0, 0]]]), atol=1e-12)
    else:
        assert np.isnan(strain).all()


@pytest.mark.parametrize("variant", ["k_nearest", "fixed_radius"])
def test_only_finite_visible_query_points_contribute_in_reference_coordinates(variant):
    reference = np.array([[-1., -.5], [0., -.5], [1., -.5],
                          [-1., .5], [0., .5], [1., .5],
                          [0., 0.], [0., 0.], [np.nan, 0.]])
    # A large translation must not move points outside the reference radius.
    tracked = reference * [1.1, 1.] + [100, -50]
    tracked[6] = [1e6, 1e6]  # Invisible, even though the coordinates are finite.
    tracked[7] = np.nan
    tracked[8] = 0
    visibility = np.ones(len(reference), dtype=bool)
    visibility[6] = False
    evaluation = np.array([[0., 0.]])
    if variant == "k_nearest":
        strain = k_nearest_strain(reference, tracked, visibility, evaluation)
    else:
        strain = fixed_radius_strain(reference, tracked, visibility, evaluation, radius=2.5)
    np.testing.assert_allclose(strain, np.array([[[.105, 0], [0, 0]]]), atol=1e-12)


def test_fixed_radius_reports_coverage_loss_without_expanding_to_distant_points():
    reference = np.array([[-1., -.5], [0., -.5], [1., -.5],
                          [-1., .5], [0., .5], [1., .5]])
    tracked = reference * [1.1, 1.]
    visibility = np.ones(len(reference), dtype=bool)
    evaluation = np.array([[0., 0.], [10., 0.]])
    strain = fixed_radius_strain(reference, tracked, visibility, evaluation, radius=2.5)
    np.testing.assert_allclose(strain[0], np.array([[.105, 0], [0, 0]]), atol=1e-12)
    assert np.isnan(strain[1]).all()
