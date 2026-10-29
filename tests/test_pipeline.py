from dataclasses import replace

import numpy as np
import pytest

from strainsight.experiment import ExperimentConfig, run_experiment
from strainsight.adaptive import RefinementConfig
from strainsight.dataset import frozen_dataset
from strainsight.results import ResultsStore
from strainsight.sampling import lattice_baseline
from strainsight.strain import MINIMUM_NEIGHBOURS, baseline_neighbour_counts
from strainsight.synthetic import UniformExtension
from strainsight.tracking import OracleTracker, Tracks


def test_adaptive_oracle_run_keeps_baseline_budget_and_calibration(tmp_path):
    store = ResultsStore(tmp_path)
    config = ExperimentConfig(frame_count=2, score_every=1, spacing=8,
                              lattice_offset=(1.5, 2.5))
    baseline = run_experiment(config, OracleTracker(), store)
    adaptive = run_experiment(replace(config, method="adaptive"), OracleTracker(), store)
    assert adaptive.record.query_count == baseline.record.query_count
    assert len(np.unique(adaptive.query_points, axis=0)) == baseline.record.query_count
    np.testing.assert_array_equal(adaptive.evaluation_grid, baseline.evaluation_grid)
    assert adaptive.record.configuration["k"] == baseline.record.configuration["k"]
    assert adaptive.record.configuration["method"] == "adaptive"
    qx, qy = np.floor(adaptive.query_points + .5).astype(int).T
    assert adaptive.sequence.specimen.sampling_region[qy, qx].all()
    np.testing.assert_allclose(adaptive.strain[-1, :, 0, 0], .105, atol=1e-12)
    np.testing.assert_allclose(adaptive.fixed_radius_strain[-1, :, 0, 0], .105, atol=1e-12)
    assert store.read(adaptive.record.run_id) == adaptive.record


def test_adaptive_coarse_compute_is_recorded_but_only_measurement_tracks_report_strain(tmp_path):
    class TwoPassTracker:
        name = "two_pass_fixture"
        support_grid_count = 36

        def __init__(self):
            self.queries = []

        def track(self, frames, query_points):
            self.queries.append(query_points.copy())
            # Deliberately disagree with truth and with the measurement pass.
            stretch = 1.4 if len(self.queries) == 1 else 1.2
            tracks = OracleTracker(UniformExtension(stretch)).track(frames, query_points)
            return replace(tracks, runtime_seconds=.25 if len(self.queries) == 1 else .5,
                           peak_vram_bytes=200 if len(self.queries) == 1 else 100)

    tracker = TwoPassTracker()
    store = ResultsStore(tmp_path)
    result = run_experiment(ExperimentConfig(method="adaptive", frame_count=2, score_every=1),
                            tracker, store)
    assert len(tracker.queries) == 2
    coarse, measurement = tracker.queries
    np.testing.assert_array_equal(result.query_points, measurement)
    for point in coarse:
        assert np.count_nonzero(np.all(measurement == point, axis=1)) == 1
    np.testing.assert_allclose(result.strain[-1, :, 0, 0], .22, atol=1e-12)
    np.testing.assert_allclose(result.fixed_radius_strain[-1, :, 0, 0], .22, atol=1e-12)
    assert result.record.tracker_runtime_seconds == .75
    assert result.record.peak_vram_bytes == 200
    adaptive = result.record.configuration["adaptive"]
    assert adaptive["coarse_query_count"] == len(coarse)
    assert adaptive["coarse_runtime_seconds"] == .25
    assert adaptive["measurement_runtime_seconds"] == .5
    assert store.read(result.record.run_id) == result.record


def test_adaptive_refinement_follows_only_the_coarse_measured_band(tmp_path):
    class MeasuredBandTracker:
        name = "measured_band_fixture"
        support_grid_count = 0

        def track(self, frames, query_points):
            positions = np.broadcast_to(query_points, (len(frames), *query_points.shape)).copy()
            # A displacement band supplied as measurements, without a motion/truth API.
            band = 6 * np.exp(-(query_points[:, 0] - 256)**2 / (2 * 12**2))
            positions[1:, :, 0] += band
            return Tracks(positions, np.ones(positions.shape[:2], dtype=bool))

    result = run_experiment(ExperimentConfig(method="adaptive", frame_count=5, score_every=4),
                            MeasuredBandTracker(), ResultsStore(tmp_path))
    plan = result.adaptive_plan
    assert plan is not None
    score = np.asarray(result.record.configuration["adaptive"]["region_scores"])
    near = np.abs(plan.coarse_points[:, 0] - 256) < 40
    far = np.abs(plan.coarse_points[:, 0] - 256) > 80
    assert score[near].mean() > 2 * score[far].mean()
    # Region counts account for the fillets' greater area away from the band.
    counts = np.asarray(result.record.configuration["adaptive"]["refinement_counts"])
    assert counts[near].mean() > counts[far].mean()
    quotas = counts.sum() * score / score.sum()
    assert np.max(np.abs(counts - quotas)) < 1  # Integer rounding only.


@pytest.mark.parametrize("spacing", [4., 5.3, 6.4, 8.])
@pytest.mark.parametrize("offset", [(0., 0.), (1.2, 2.3)])
def test_adaptive_geometry_floor_and_nested_lattice_across_budget_curve(tmp_path, spacing, offset):
    result = run_experiment(ExperimentConfig(method="adaptive", spacing=spacing, lattice_offset=offset,
                                             frame_count=2, score_every=1),
                            OracleTracker(), ResultsStore(tmp_path))
    plan = result.adaptive_plan
    assert plan is not None
    mask = result.sequence.specimen.sampling_region
    baseline = lattice_baseline(mask, spacing, offset)
    np.testing.assert_array_equal(plan.coarse_points, lattice_baseline(mask, 1.5 * spacing, offset))
    assert len(result.query_points) == len(baseline)
    assert len(np.unique(result.query_points, axis=0)) == len(baseline)
    # Every point is on the same nested square lattice with the shared offset.
    lattice_indices = (result.query_points - offset) / (spacing / 2)
    np.testing.assert_allclose(lattice_indices, np.rint(lattice_indices), atol=1e-12)
    counts = baseline_neighbour_counts(np.vstack((plan.coarse_points, plan.coverage_points)),
                                       plan.coarse_points, 2.5 * spacing)
    assert counts.min() >= MINIMUM_NEIGHBOURS
    # The geometry-only floor should leave most of the budget for score allocation.
    assert len(plan.coverage_points) / len(baseline) <= .085
    assert result.record.configuration["adaptive"]["visibility_gate_active"] is False
    assert "inactive" in result.record.configuration["adaptive"]["visibility_gate_note"]


def test_adaptive_track_uncertainty_only_suppresses_scores_and_does_not_change_floor(tmp_path):
    class TranslationTracker:
        name = "translation_fixture"
        support_grid_count = 0

        def __init__(self, lost):
            self.lost = lost

        def track(self, frames, query_points):
            positions = np.broadcast_to(query_points, (len(frames), *query_points.shape)).copy()
            positions += [7, -3]
            visibility = np.ones(positions.shape[:2], dtype=bool)
            if self.lost:
                visibility[:, query_points[:, 0] < 256] = False
            return Tracks(positions, visibility)

    config = ExperimentConfig(method="adaptive", frame_count=2, score_every=1,
                               refinement=RefinementConfig(gradient_weight=0, coverage_weight=1))
    store = ResultsStore(tmp_path)
    full = run_experiment(config, TranslationTracker(False), store)
    lost = run_experiment(config, TranslationTracker(True), store)
    full_score = np.asarray(full.record.configuration["adaptive"]["region_scores"])
    lost_score = np.asarray(lost.record.configuration["adaptive"]["region_scores"])
    assert np.all(lost_score <= full_score)
    assert np.any(lost_score < full_score)
    assert lost.record.configuration["adaptive"]["visibility_gate_active"] is True
    assert full.adaptive_plan is not None and lost.adaptive_plan is not None
    np.testing.assert_array_equal(full.adaptive_plan.coverage_points, lost.adaptive_plan.coverage_points)
    assert lost.record.query_count == full.record.query_count


def test_adaptive_temporal_max_finds_late_bands_and_smoothing_suppresses_spikes(tmp_path):
    class IntermittentBandTracker:
        name = "intermittent_band_fixture"
        support_grid_count = 0

        def __init__(self, band_frames):
            self.band_frames = band_frames

        def track(self, frames, query_points):
            positions = np.broadcast_to(query_points, (len(frames), *query_points.shape)).copy()
            band = 6 * np.exp(-(query_points[:, 0] - 256)**2 / (2 * 12**2))
            positions[self.band_frames, :, 0] += band
            return Tracks(positions, np.ones(positions.shape[:2], dtype=bool))

    config = ExperimentConfig(method="adaptive", frame_count=8, score_every=7,
                               refinement=RefinementConfig(coverage_weight=0))
    store = ResultsStore(tmp_path)
    scores = []
    for frames in ([2], [2, 3, 4], [5, 6, 7]):
        result = run_experiment(config, IntermittentBandTracker(frames), store)
        scores.append(np.asarray(result.record.configuration["adaptive"]["region_scores"]))
    # A three-frame band survives the mean, a single-frame spike is diluted.
    np.testing.assert_allclose(scores[0], scores[1] / 3, atol=1e-12)
    np.testing.assert_allclose(scores[1], scores[2], atol=1e-12)
    assert scores[2].max() > .01


def test_adaptive_rejects_a_coarse_pass_with_missing_frames(tmp_path):
    class TruncatedTracker(OracleTracker):
        def track(self, frames, query_points):
            tracks = super().track(frames, query_points)
            # Malformed coarse output must fail before starting measurement.
            return replace(tracks, positions=tracks.positions[:1], visibility=tracks.visibility[:1])

    with pytest.raises(ValueError, match="coarse tracker must return.*given query points"):
        run_experiment(ExperimentConfig(method="adaptive", frame_count=2, score_every=1),
                       TruncatedTracker(), ResultsStore(tmp_path))


def test_uniform_extension_run_recovers_green_lagrange_and_persists(tmp_path):
    config = ExperimentConfig(frame_count=5, score_every=4, lattice_offset=(1.2, 2.3))
    store = ResultsStore(tmp_path)
    result = run_experiment(config, OracleTracker(), store)

    # 10% stretch gives 0.5 * (1.1**2 - 1) = 0.105, not 0.1.
    np.testing.assert_allclose(result.strain[-1, :, 0, 0], 0.105, atol=1e-12)
    np.testing.assert_allclose(result.strain[-1, :, 1, 1], -0.04545454545454547, atol=1e-12)
    np.testing.assert_allclose(result.strain[-1, :, 0, 1], 0, atol=1e-12)
    np.testing.assert_allclose(result.fixed_radius_strain, result.strain, atol=1e-12)
    mae = result.record.metrics["full_field_exx_mae"]
    assert mae is not None and mae < 1e-12
    assert result.record.metrics["coverage_loss"] == 0
    secondary_mae = result.record.metrics["fixed_radius_exx_mae"]
    assert secondary_mae is not None and secondary_mae < 1e-12
    assert result.record.metrics["fixed_radius_coverage_loss"] == 0
    assert result.record.configuration["fixed_radius"] == 16
    assert len(result.record.configuration["k"]) == len(result.evaluation_grid)
    assert result.record.query_count == len(result.query_points)
    assert result.record.query_count == len(np.unique(result.query_points, axis=0))
    assert result.record.runtime_seconds > 0
    assert store.read(result.record.run_id) == result.record
    assert store.read(result.record.run_id).configuration["lattice_offset"] == [1.2, 2.3]


def test_lattice_budget_and_scoring_use_the_reference_regions(tmp_path):
    result = run_experiment(ExperimentConfig(frame_count=2, score_every=1, spacing=8),
                            OracleTracker(), ResultsStore(tmp_path))
    specimen = result.sequence.specimen
    assert specimen.specimen_mask.shape == (384, 512)
    # Fixed dimensions at pixel centres: length=380 and gauge width=64.
    assert np.flatnonzero(specimen.specimen_mask[192])[[0, -1]].tolist() == [66, 446]
    assert np.flatnonzero(specimen.specimen_mask[:, 256])[[0, -1]].tolist() == [160, 224]
    assert np.flatnonzero(specimen.specimen_mask[:, 75])[[0, -1]].tolist() == [141, 243]
    assert np.flatnonzero(specimen.eroded_mask[:, 256])[[0, -1]].tolist() == [164, 220]
    assert specimen.specimen_mask[192, 75]  # Grip is visible but never sampled.
    assert not specimen.sampling_region[192, 75]
    assert specimen.sampling_region[192, 100]  # Fillet can receive query points.
    assert not specimen.gauge_section[192, 100]
    assert np.all(specimen.eroded_mask <= specimen.specimen_mask)
    assert np.all(specimen.sampling_region <= specimen.eroded_mask)
    # Enumerated count for this lattice: central rows plus circular fillets.
    assert result.record.query_count == 317
    assert np.all(result.query_points % 8 == 0)
    qx, qy = np.floor(result.query_points + 0.5).astype(int).T
    assert specimen.sampling_region[qy, qx].all()
    ex, ey = result.evaluation_grid.astype(int).T
    assert (specimen.eroded_mask & specimen.gauge_section)[ey, ex].all()
    assert len(result.evaluation_grid) == 975


def test_pipeline_rejects_wrong_direction_inverse_warp(tmp_path):
    class WrongDirection(UniformExtension):
        def inverse(self, points, time):
            return self.forward(points, time)

    class MustNotTrack(OracleTracker):
        def track(self, frames, query_points):
            pytest.fail("a rejected synthetic sequence must not reach tracking")

    with pytest.raises(ValueError, match="inverse mapping residual.*0.01"):
        run_experiment(ExperimentConfig(frame_count=2, score_every=1), MustNotTrack(),
                       ResultsStore(tmp_path), motion=WrongDirection())


def test_results_store_refuses_to_replace_completed_run(tmp_path):
    store = ResultsStore(tmp_path)
    result = run_experiment(ExperimentConfig(frame_count=2, score_every=1), OracleTracker(), store)
    changed = replace(result.record, query_count=1)
    with pytest.raises(FileExistsError, match="immutable run"):
        store.write(changed)
    assert ResultsStore(tmp_path).read(result.record.run_id) == result.record


def test_injected_tracker_measurements_determine_strain_not_ground_truth(tmp_path):
    class ShearTracker:
        name = "affine_fixture"
        support_grid_count = 42

        def track(self, frames, query_points):
            # Independent affine example: F=[[1.1, .2], [0, 1]].
            final = query_points @ np.array([[1.1, 0], [0.2, 1]]) + [3, -7]
            return Tracks(np.stack([query_points, final]), np.ones((2, len(query_points)), dtype=bool))

    result = run_experiment(ExperimentConfig(frame_count=2, score_every=1), ShearTracker(),
                            ResultsStore(tmp_path))
    np.testing.assert_allclose(result.strain[0], np.broadcast_to([[0.105, 0.11], [0.11, 0.02]],
                                                               result.strain[0].shape), atol=1e-12)
    assert result.tracks.positions.shape == (2, result.record.query_count, 2)
    assert result.record.configuration["tracker"] == "affine_fixture"
    assert result.record.configuration["support_grid_count"] == 42
    assert result.record.query_count == 523


def test_variants_coincide_on_offset_lattice_baseline_for_non_affine_tracks(tmp_path):
    class NonAffineTracker:
        name = "non_affine_fixture"
        support_grid_count = 0

        def track(self, frames, query_points):
            final = query_points.copy()
            final[:, 0] += .0001 * (query_points[:, 0] - 256)**3
            return Tracks(np.stack([query_points, final]), np.ones((2, len(query_points)), dtype=bool))

    result = run_experiment(ExperimentConfig(frame_count=2, score_every=1, spacing=8,
                                             lattice_offset=(1.5, 2.5)),
                            NonAffineTracker(), ResultsStore(tmp_path))
    counts = result.record.configuration["k"]
    assert len(counts) == len(result.evaluation_grid)
    assert min(counts) < max(counts)
    np.testing.assert_allclose(result.strain, result.fixed_radius_strain, atol=1e-12)


def test_lost_tracks_count_against_budget_and_show_as_coverage_loss(tmp_path):
    class LostTracker(OracleTracker):
        name = "lost_fixture"

        def track(self, frames, query_points):
            tracked = super().track(frames, query_points)
            tracked.visibility[-1] = False
            tracked.positions[-1] = np.nan
            return tracked

    result = run_experiment(ExperimentConfig(frame_count=3, score_every=1), LostTracker(),
                            ResultsStore(tmp_path))
    assert result.record.query_count == 523
    assert result.record.metrics["coverage_loss"] == 0.5
    assert result.record.metrics["fixed_radius_coverage_loss"] == 0.5
    secondary_mae = result.record.metrics["fixed_radius_exx_mae"]
    assert secondary_mae is not None and secondary_mae < 1e-12
    mae = result.record.metrics["full_field_exx_mae"]
    assert mae is not None and mae < 1e-12
    assert result.record.metrics["tracking_failure_fraction"] == 0.5
    assert np.isnan(result.strain[-1]).all()
    assert np.isnan(result.fixed_radius_strain[-1]).all()


def test_tracker_resources_and_settings_are_preserved_in_run_record(tmp_path):
    class MeasuredTracker(OracleTracker):
        name = "measured_fixture"
        support_grid_count = 36

        def track(self, frames, query_points):
            tracks = super().track(frames, query_points)
            return replace(tracks, runtime_seconds=0.25, peak_vram_bytes=123456,
                           configuration={"support_grid_size": 6, "visibility_threshold": 0.9,
                                          "confidence_used": False})

    store = ResultsStore(tmp_path)
    result = run_experiment(ExperimentConfig(frame_count=2, score_every=1),
                            MeasuredTracker(), store)
    record = store.read(result.record.run_id)
    assert record.tracker_runtime_seconds == 0.25
    assert record.peak_vram_bytes == 123456
    assert record.configuration["tracker_settings"] == {
        "support_grid_size": 6, "visibility_threshold": 0.9, "confidence_used": False}
    assert record.configuration["support_grid_count"] == 36
    assert record.query_count == 523  # Auxiliary support grid is excluded.


@pytest.mark.parametrize("available", ["none", "five", "collinear"])
def test_unobservable_strain_is_missing_instead_of_a_zero_error(tmp_path, available):
    class InsufficientTracker(OracleTracker):
        def track(self, frames, query_points):
            tracked = super().track(frames, query_points)
            tracked.visibility[:] = False
            if available == "five":
                tracked.visibility[:, :5] = True
            elif available == "collinear":
                tracked.visibility[:, np.isclose(query_points[:, 1], 192)] = True
            return tracked

    store = ResultsStore(tmp_path)
    result = run_experiment(ExperimentConfig(frame_count=2, score_every=1), InsufficientTracker(), store)
    assert result.record.metrics["coverage_loss"] == 1
    assert result.record.metrics["full_field_exx_mae"] is None
    assert result.record.metrics["fixed_radius_coverage_loss"] == 1
    assert result.record.metrics["fixed_radius_exx_mae"] is None
    assert np.isnan(result.strain).all()
    assert np.isnan(result.fixed_radius_strain).all()
    assert store.read(result.record.run_id) == result.record


def test_sequence_and_evaluation_grid_are_independent_of_query_placement(tmp_path):
    store = ResultsStore(tmp_path)
    first = run_experiment(ExperimentConfig(frame_count=2, score_every=1), OracleTracker(), store)
    second = run_experiment(ExperimentConfig(frame_count=2, score_every=1, spacing=8,
                                             lattice_offset=(1.5, 2.5)), OracleTracker(), store)
    assert second.record.configuration["fixed_radius"] == 20
    np.testing.assert_array_equal(first.sequence.frames, second.sequence.frames)
    np.testing.assert_array_equal(first.evaluation_grid, second.evaluation_grid)
    assert first.record.configuration["frames_sha256"] == second.record.configuration["frames_sha256"]
    assert not np.array_equal(first.sequence.frames[0], first.sequence.frames[-1])
    assert np.std(first.sequence.frames[0][first.sequence.specimen.specimen_mask]) > 10
    assert first.sequence.max_inversion_residual <= 0.01


def test_fixed_radius_coverage_is_reported_separately_when_tracks_survive_only_locally(tmp_path):
    class LocalTracker(OracleTracker):
        def track(self, frames, query_points):
            tracked = super().track(frames, query_points)
            # Six query points in two rows near the centre of the gauge section.
            tracked.visibility[:] = ((np.abs(query_points[:, 0] - 256) < 7)
                                     & (query_points[:, 1] > 185)
                                     & (query_points[:, 1] <= 192))
            return tracked

    store = ResultsStore(tmp_path)
    result = run_experiment(ExperimentConfig(frame_count=2, score_every=1), LocalTracker(), store)
    assert result.tracks.visibility[-1].sum() == 6
    assert result.record.query_count == 523
    assert result.record.metrics["coverage_loss"] == 0
    coverage_loss = result.record.metrics["fixed_radius_coverage_loss"]
    assert coverage_loss is not None and 0.9 < coverage_loss < 1
    secondary_mae = result.record.metrics["fixed_radius_exx_mae"]
    assert secondary_mae is not None and secondary_mae < 1e-12
    centre = np.all(result.evaluation_grid == [256, 192], axis=1)
    far = result.evaluation_grid[:, 0] < 200
    assert np.isfinite(result.fixed_radius_strain[:, centre]).all()
    assert np.isnan(result.fixed_radius_strain[:, far]).all()
    assert store.read(result.record.run_id) == result.record


def test_oracle_tracks_only_the_requested_material_positions(tmp_path):
    config = ExperimentConfig(frame_count=2, score_every=1, final_stretch=1.2)
    result = run_experiment(config, OracleTracker(UniformExtension(1.2)), ResultsStore(tmp_path))
    np.testing.assert_allclose(result.tracks.positions[0], result.query_points, atol=1e-12)
    np.testing.assert_allclose(result.tracks.positions[-1, :, 0],
                               256 + 1.2 * (result.query_points[:, 0] - 256), atol=1e-12)
    np.testing.assert_allclose(result.tracks.positions[-1, :, 1],
                               192 + 0.9128709291752769 * (result.query_points[:, 1] - 192), atol=1e-12)
    assert result.tracks.visibility.all()
    assert result.tracks.positions.shape == (2, result.record.query_count, 2)
    np.testing.assert_allclose(result.strain[-1, :, 0, 0], 0.22, atol=1e-12)


def test_pipeline_rejects_an_oracle_configured_for_a_different_sequence(tmp_path):
    config = ExperimentConfig(frame_count=2, score_every=1, final_stretch=1.2)
    with pytest.raises(ValueError, match="oracle motion must match the synthetic sequence"):
        run_experiment(config, OracleTracker(), ResultsStore(tmp_path))


def test_frozen_necking_run_records_generator_provenance_and_spatial_truth(tmp_path):
    dataset = frozen_dataset()
    case = dataset.dev[0]
    config = ExperimentConfig.from_case(case, spacing=8)
    store = ResultsStore(tmp_path)
    result = run_experiment(config, OracleTracker(case.motion), store)
    recorded = store.read(result.record.run_id).configuration
    assert recorded["generator_version"] == dataset.identity.version
    assert recorded["generator_sha256"] == dataset.identity.sha256
    assert recorded["synthetic_case"]["seed"] == case.seed
    assert recorded["synthetic_case"]["speckle_size"] == case.speckle_size
    assert recorded["synthetic_case"]["noise_sigma"] == case.noise_sigma
    assert recorded["synthetic_case"]["blur_sigma"] == case.blur_sigma
    assert recorded["dataset_split"] == "dev"
    assert recorded["k_calibration"] == "intact_lattice_baseline"
    assert recorded["fixed_radius"] == 20
    np.testing.assert_allclose(result.fixed_radius_strain, result.strain, atol=1e-12)
    assert result.record.metrics["fixed_radius_coverage_loss"] == 0
    assert result.record.metrics["fixed_radius_exx_mae"] == pytest.approx(
        result.record.metrics["full_field_exx_mae"], abs=1e-12)
    assert result.scored_frames.tolist() == [4, 8, 12, 16, 20, 24, 28]
    assert result.ground_truth_strain[-1, :, 0, 0].ptp() > 0.1
    truth = result.sequence.ground_truth(28)
    ex, ey = result.evaluation_grid.astype(int).T
    np.testing.assert_allclose(result.ground_truth_strain[-1], truth.strain[ey, ex], atol=1e-12)
    np.testing.assert_allclose(result.tracks.positions[-1],
                               case.motion.forward(result.query_points, 1), atol=1e-12)


def test_necking_oracle_must_match_the_entire_spatial_motion(tmp_path):
    case = frozen_dataset().dev[0]
    wrong_motion = replace(case.motion, band_center=case.band_center + 8)
    with pytest.raises(ValueError, match="oracle motion must match the synthetic sequence"):
        run_experiment(ExperimentConfig.from_case(case), OracleTracker(wrong_motion), ResultsStore(tmp_path))
