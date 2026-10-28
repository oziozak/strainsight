from dataclasses import replace

import numpy as np
import pytest

from strainsight.experiment import ExperimentConfig, run_experiment
from strainsight.results import ResultsStore
from strainsight.synthetic import UniformExtension
from strainsight.tracking import OracleTracker, Tracks


def test_uniform_extension_run_recovers_green_lagrange_and_persists(tmp_path):
    config = ExperimentConfig(frame_count=5, score_every=4, lattice_offset=(1.2, 2.3))
    store = ResultsStore(tmp_path)
    result = run_experiment(config, OracleTracker(), store)

    # 10% stretch gives 0.5 * (1.1**2 - 1) = 0.105, not 0.1.
    np.testing.assert_allclose(result.strain[-1, :, 0, 0], 0.105, atol=1e-12)
    np.testing.assert_allclose(result.strain[-1, :, 1, 1], -0.04545454545454547, atol=1e-12)
    np.testing.assert_allclose(result.strain[-1, :, 0, 1], 0, atol=1e-12)
    mae = result.record.metrics["full_field_exx_mae"]
    assert mae is not None and mae < 1e-12
    assert result.record.metrics["coverage_loss"] == 0
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
    mae = result.record.metrics["full_field_exx_mae"]
    assert mae is not None and mae < 1e-12
    assert result.record.metrics["tracking_failure_fraction"] == 0.5
    assert np.isnan(result.strain[-1]).all()


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
    assert np.isnan(result.strain).all()
    assert store.read(result.record.run_id) == result.record


def test_sequence_and_evaluation_grid_are_independent_of_query_placement(tmp_path):
    store = ResultsStore(tmp_path)
    first = run_experiment(ExperimentConfig(frame_count=2, score_every=1), OracleTracker(), store)
    second = run_experiment(ExperimentConfig(frame_count=2, score_every=1, spacing=8,
                                             lattice_offset=(1.5, 2.5)), OracleTracker(), store)
    np.testing.assert_array_equal(first.sequence.frames, second.sequence.frames)
    np.testing.assert_array_equal(first.evaluation_grid, second.evaluation_grid)
    assert first.record.configuration["frames_sha256"] == second.record.configuration["frames_sha256"]
    assert not np.array_equal(first.sequence.frames[0], first.sequence.frames[-1])
    assert np.std(first.sequence.frames[0][first.sequence.specimen.specimen_mask]) > 10
    assert first.sequence.max_inversion_residual <= 0.01


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
