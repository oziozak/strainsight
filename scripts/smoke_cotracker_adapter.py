"""Opt-in GPU check of the adapter through the experiment pipeline.

    uv run python scripts/smoke_cotracker_adapter.py [--checkpoint PATH]

Not collected by pytest. Uses the real offline checkpoint on CUDA and writes
immutable records under outputs/. Model load is excluded from tracker timing.
"""
import argparse
from pathlib import Path

import numpy as np

from strainsight.cotracker import CoTracker3Tracker
from strainsight.experiment import ExperimentConfig, run_experiment
from strainsight.results import ResultsStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/cotracker_adapter_smoke"))
    args = parser.parse_args()
    tracker = CoTracker3Tracker(args.checkpoint)
    store = ResultsStore(args.results_dir)

    for spacing, expected_count in [(8.0, 317), (6.4, 523)]:
        result = run_experiment(ExperimentConfig(spacing=spacing), tracker, store)
        record = store.read(result.record.run_id)
        assert record == result.record
        assert record.query_count == expected_count
        assert result.tracks.positions.shape == (32, expected_count, 2)
        assert result.tracks.visibility.shape == (32, expected_count)
        assert result.tracks.visibility.dtype == np.bool_
        np.testing.assert_allclose(result.tracks.positions[0], result.query_points, atol=1e-4)
        assert result.tracks.visibility[0].all()
        assert record.configuration["support_grid_count"] == 36
        assert record.configuration["tracker_settings"]["support_grid_size"] == 6
        assert record.configuration["tracker_settings"]["visibility_threshold"] == 0.9
        assert record.configuration["tracker_settings"]["confidence_used"] is False
        assert record.tracker_runtime_seconds is not None and record.tracker_runtime_seconds > 0
        assert record.runtime_seconds >= record.tracker_runtime_seconds
        assert record.peak_vram_bytes > 0
        expected = result.sequence.motion.forward(result.query_points, 1.0)
        visible = result.tracks.visibility[-1]
        assert visible.mean() > 0.5, "most queries should survive uniform extension"
        error = np.linalg.norm(result.tracks.positions[-1, visible] - expected[visible], axis=-1)
        assert np.median(error) < 3, f"unexpected displacement error: {np.median(error)} px"
        print(f"{record.run_id}: N={record.query_count}, support=36, "
              f"tracker={record.tracker_runtime_seconds:.3f}s, "
              f"peak={record.peak_vram_bytes / 2**20:.1f}MiB, median error={np.median(error):.3f}px")

    # Exercise RGB and the predictor's resize/backscale path independently of
    # the native grayscale tracker input; positions must remain in input pixels.
    rgb = np.repeat(result.sequence.frames[..., None], 3, axis=-1)
    rgb = np.repeat(np.repeat(rgb, 2, axis=1), 2, axis=2)
    queries = result.query_points * 2
    tracks = tracker.track(rgb, queries)
    np.testing.assert_allclose(tracks.positions[0], queries, atol=1e-4)
    visible = tracks.visibility[-1]
    assert visible.mean() > 0.5
    error = np.linalg.norm(tracks.positions[-1, visible] - expected[visible] * 2, axis=-1)
    assert np.median(error) < 6, f"unexpected resized displacement error: {np.median(error)} px"
    print(f"RGB 1024x768: median error={np.median(error):.3f}px; adapter smoke passed")


if __name__ == "__main__":
    main()
