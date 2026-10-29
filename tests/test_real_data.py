"""Generated fixtures only: no DIC Challenge images or annotations."""
import json

import numpy as np
import pytest

from strainsight.real_data import HandMaskSegmenter, Sample6ROI, prepare_sample6
from strainsight.real_experiment import RealDataConfig, run_sample6
from strainsight.results import ResultsStore
from strainsight.tracking import Tracks


@pytest.fixture
def sample6_input(tmp_path):
    frames = np.full((3, 1100, 1500), 11, dtype=np.uint8)
    frames[:, 30:1054, 50:1415] = 29
    specimen = np.zeros((1024, 1365), dtype=bool)
    specimen[350:674, 100:1265] = True
    sampling = specimen.copy()
    sampling[:, :300] = sampling[:, 1065:] = False
    gauge = sampling.copy()
    gauge[:, :400] = gauge[:, 965:] = False
    path = tmp_path / "hand-mask.npz"
    np.savez(path, specimen_mask=specimen, sampling_region=sampling, gauge_section=gauge)
    return frames, Sample6ROI(50, 30), HandMaskSegmenter(path)


def test_preparation_crops_before_tracking_and_preserves_reference_geometry(sample6_input):
    frames, roi, segmenter = sample6_input
    data = prepare_sample6(frames, roi, segmenter, edges_verified=True)
    assert data.frames.shape == (3, 384, 512)
    assert np.all(data.frames == 29)  # Full-frame border must never reach the tracker.
    np.testing.assert_allclose(roi.to_full_frame([[255.5, 191.5]]), [[732, 541.5]])
    points = np.array([[0., 0.], [511, 383], [123.25, 200.75]])
    np.testing.assert_allclose(roi.to_tracker_input(roi.to_full_frame(points)), points, atol=1e-12)
    assert data.specimen.specimen_mask[131, 200]
    assert not data.specimen.eroded_mask[131, 200]
    assert data.specimen.eroded_mask[136, 200]
    assert not data.specimen.sampling_region[150, 50]  # Annotated grip.
    assert data.specimen.sampling_region[150, 120]  # Annotated fillet.
    assert not data.specimen.gauge_section[150, 120]
    assert data.specimen.gauge_section[150, 200]


def test_hand_mask_path_rejects_missing_region_annotations(sample6_input, tmp_path):
    frames, roi, _ = sample6_input
    path = tmp_path / "incomplete.npz"
    np.savez(path, specimen_mask=np.ones((1024, 1365), dtype=bool))
    with pytest.raises(ValueError, match="specimen_mask, sampling_region and gauge_section"):
        prepare_sample6(frames, roi, HandMaskSegmenter(path), edges_verified=True)


class DensitySensitiveTracker:
    """Independent affine fixture with a deliberate query-density bias."""

    name = "density_sensitive_fixture"
    support_grid_count = 36

    def track(self, frames, query_points):
        assert frames.shape == (3, 384, 512)
        assert np.all(frames == 29)
        extension = .08 if len(query_points) > 1000 else .10
        positions = np.broadcast_to(query_points, (3, *query_points.shape)).copy()
        for frame in range(3):
            positions[frame, :, 0] += frame / 2 * extension * (query_points[:, 0] - 256)
            positions[frame] += [3 * frame, -2 * frame]
        return Tracks(positions, np.ones(positions.shape[:2], dtype=bool),
                      runtime_seconds=.25, peak_vram_bytes=1234,
                      configuration={"support_grid_size": 6})


def test_real_runs_report_dense_consistency_and_grip_sanity_with_equal_budgets(sample6_input, tmp_path):
    data = prepare_sample6(*sample6_input, edges_verified=True)
    store = ResultsStore(tmp_path / "results")
    result = run_sample6(data, DensitySensitiveTracker(), store, np.array([5., 6., 7.]),
                        RealDataConfig(score_every=1, evaluation_spacing=16))
    assert result.baseline.record.query_count == result.adaptive.record.query_count
    assert result.dense_reference.record.query_count >= 4 * result.baseline.record.query_count
    for run in (result.baseline, result.adaptive):
        np.testing.assert_allclose(run.strain[-1, :, 0, 0], .105, atol=1e-12)
        # E(1.05)-E(1.04)=.01045; E(1.10)-E(1.08)=.0218; mean=.016125.
        assert run.record.metrics["dense_reference_loading_axis_mae"] == pytest.approx(.016125)
        assert run.record.metrics["fixed_radius_dense_reference_loading_axis_mae"] == pytest.approx(.016125)
        assert run.record.metrics["coverage_loss"] == 0
        assert run.record.metrics["dense_comparison_coverage_loss"] == 0
        assert "full_field_exx_mae" not in run.record.metrics
        assert run.record.metrics["grip_extension_trend_correlation"] == pytest.approx(1)
        np.testing.assert_allclose(run.global_elongation, [0, .05, .10], atol=1e-12)
        np.testing.assert_allclose(run.full_frame_tracks.positions[0], run.full_frame_query_points)
        centre = np.array([[255.5, 191.5]])
        np.testing.assert_allclose(data.roi.to_full_frame(centre), [[732, 541.5]])
        assert run.record.configuration["dense_reference_run_id"] == result.dense_reference.record.run_id
        assert run.record.configuration["downscale_factor_xy"] == [1365 / 512, 1024 / 384]
        assert "downscaled" in run.record.configuration["limitations"][0]
        assert store.read(run.record.run_id) == run.record
    assert result.baseline.record.tracker_runtime_seconds == .25
    assert result.adaptive.record.tracker_runtime_seconds == .5
    assert result.adaptive.record.peak_vram_bytes == 1234
    assert result.adaptive.record.configuration["adaptive"]["coarse_query_count"] > 0
    np.testing.assert_allclose(result.dense_reference.strain[-1, :, 0, 0], .0832, atol=1e-12)
    site = np.flatnonzero(np.all(result.baseline.query_points == [200, 160], axis=1)).item()
    np.testing.assert_allclose(result.baseline.full_frame_tracks.positions[1, site],
                               [584.5693359375, 452.1666666666667], atol=1e-10)


def test_external_manifest_preserves_frame_order_and_grip_alignment(tmp_path):
    from strainsight.real_data import load_sample6_manifest

    pytest.importorskip("PIL")
    # PGM bytes are generated, with no image-writing library or source dataset.
    (tmp_path / "frame10.pgm").write_bytes(b"P5\n2 2\n255\n" + bytes([17] * 4))
    (tmp_path / "frame2.pgm").write_bytes(b"P5\n2 2\n255\n" + bytes([31] * 4))
    path = tmp_path / "sequence.json"
    path.write_text(json.dumps({"frames": ["frame10.pgm", "frame2.pgm"],
                               "grip_extension": [4.5, 6.5], "grip_extension_units": "mm"}))
    source = load_sample6_manifest(path)
    assert source.frames[:, 0, 0].tolist() == [17, 31]
    assert source.frame_ids == ["frame10.pgm", "frame2.pgm"]
    assert source.grip_extension.tolist() == [4.5, 6.5]
    assert source.grip_extension_units == "mm"
    path.write_text(json.dumps({"frames": ["frame10.pgm", "frame2.pgm"],
                               "grip_extension": [4.5], "grip_extension_units": "mm"}))
    with pytest.raises(ValueError, match="one-to-one"):
        load_sample6_manifest(path)


@pytest.mark.parametrize("lost_method", ["dense", "budgeted"])
def test_missing_dense_or_method_estimates_stay_unavailable(sample6_input, tmp_path, lost_method):
    class LostTracker(DensitySensitiveTracker):
        def track(self, frames, query_points):
            tracks = super().track(frames, query_points)
            if (len(query_points) > 1000) == (lost_method == "dense"):
                tracks.visibility[:] = False
                tracks.positions[:] = np.nan
            return tracks

    data = prepare_sample6(*sample6_input, edges_verified=True)
    store = ResultsStore(tmp_path / "results")
    result = run_sample6(data, LostTracker(), store, np.array([0., 1., 2.]),
                        RealDataConfig(score_every=1, evaluation_spacing=16))
    for run in (result.baseline, result.adaptive):
        assert run.record.metrics["dense_reference_loading_axis_mae"] is None
        assert run.record.metrics["fixed_radius_dense_reference_loading_axis_mae"] is None
        assert run.record.metrics["dense_comparison_coverage_loss"] == 1
        assert run.record.metrics["coverage_loss"] == (0 if lost_method == "dense" else 1)
        assert store.read(run.record.run_id) == run.record
        if lost_method == "budgeted":
            assert run.record.metrics["grip_extension_trend_correlation"] is None
            assert run.record.configuration["global_elongation"] == [None, None, None]


def test_preparation_requires_human_edge_check_and_roi_sized_annotation(sample6_input):
    frames, roi, segmenter = sample6_input
    with pytest.raises(ValueError, match="human verification"):
        prepare_sample6(frames, roi, segmenter, edges_verified=False)
    with pytest.raises(ValueError, match="must fit"):
        prepare_sample6(frames, Sample6ROI(300, 30), segmenter, edges_verified=True)
    with pytest.raises(ValueError, match="smaller than the full frame"):
        prepare_sample6(frames[:, 30:1054, 50:1415], Sample6ROI(0, 0), segmenter, edges_verified=True)


def test_rgb_preparation_keeps_channel_order_and_segments_reference_roi_once(sample6_input):
    frames, roi, hand_mask = sample6_input
    rgb = np.stack((frames, frames + 1, frames + 2), axis=-1)

    class ReferenceSegmenter:
        def __init__(self):
            self.references = []

        def segment(self, reference):
            self.references.append(reference)
            return hand_mask.segment(reference)

    segmenter = ReferenceSegmenter()
    data = prepare_sample6(rgb, roi, segmenter, edges_verified=True)
    assert data.frames.shape == (3, 384, 512, 3)
    np.testing.assert_array_equal(data.frames[0, 0, 0], [29, 30, 31])
    assert len(segmenter.references) == 1
    assert segmenter.references[0].shape == (1024, 1365, 3)
