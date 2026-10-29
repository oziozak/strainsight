"""Run Sample6 using external images, a human annotation, and CoTracker3."""
import argparse
import json
from pathlib import Path

import numpy as np

from .real_data import HandMaskSegmenter, Sample6ROI, load_sample6_manifest, prepare_sample6
from .real_experiment import RealDataConfig, run_sample6
from .results import ResultsStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="external JSON with ordered frames and aligned grip extension")
    parser.add_argument("--mask", type=Path, required=True,
                        help="external NPZ: Boolean specimen_mask, sampling_region, gauge_section in ROI pixels")
    parser.add_argument("--roi", type=int, nargs=2, required=True, metavar=("X", "Y"),
                        help="top-left of the fixed 1365x1024 specimen ROI in full-frame pixels")
    parser.add_argument("--edges-verified", action="store_true", required=True,
                        help="confirm you checked both lateral specimen edges stay inside this ROI in every frame")
    parser.add_argument("--spacing", type=float, default=8)
    parser.add_argument("--offset", type=float, nargs=2, default=(0., 0.))
    parser.add_argument("--dense-spacing", type=float, default=2)
    parser.add_argument("--score-every", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/sample6"))
    args = parser.parse_args()
    config = RealDataConfig(spacing=args.spacing, lattice_offset=tuple(args.offset),
                            dense_spacing=args.dense_spacing, score_every=args.score_every)
    source = load_sample6_manifest(args.manifest)
    data = prepare_sample6(source.frames, Sample6ROI(*args.roi), HandMaskSegmenter(args.mask),
                           edges_verified=args.edges_verified)
    from .cotracker import CoTracker3Tracker

    result = run_sample6(data, CoTracker3Tracker(args.checkpoint), ResultsStore(args.results_dir),
                         source.grip_extension, config, frame_ids=source.frame_ids,
                         grip_extension_units=source.grip_extension_units)
    tracks_dir = args.results_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    for run in (result.dense_reference, result.baseline, result.adaptive):
        path = tracks_dir / f"{run.record.run_id}.npz"
        np.savez_compressed(path, query_points_full_frame=run.full_frame_query_points,
                            positions_full_frame=run.full_frame_tracks.positions,
                            visibility=run.full_frame_tracks.visibility,
                            query_points_tracker_input=run.query_points,
                            positions_tracker_input=run.tracks.positions)
        print(json.dumps({"method": run.record.configuration["method"], "run_id": run.record.run_id,
                          "query_count": run.record.query_count, "metrics": run.record.metrics,
                          "tracks": str(path)}, allow_nan=False))
    for limitation in result.baseline.record.configuration["limitations"]:
        print(f"Limitation: {limitation}")
    print(f"Results store: {args.results_dir / 'runs.sqlite3'}")


if __name__ == "__main__":
    main()
