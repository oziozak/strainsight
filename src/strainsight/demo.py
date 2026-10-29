"""Run the uniform-extension tracer bullet and save an immutable run record."""
import argparse
from pathlib import Path

from .experiment import ExperimentConfig, run_experiment
from .results import ResultsStore
from .tracking import OracleTracker, Tracker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    parser.add_argument("--spacing", type=float, default=6.4)
    parser.add_argument("--method", choices=("lattice_baseline", "adaptive"), default="lattice_baseline")
    parser.add_argument("--offset", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"))
    parser.add_argument("--tracker", choices=("oracle", "cotracker"), default="oracle")
    parser.add_argument("--checkpoint", type=Path, help="local CoTracker3 offline checkpoint")
    parser.add_argument("--visualize", type=Path, nargs="?", const=Path("outputs/tracking.html"),
                        metavar="HTML", help="save interactive track playback (default: outputs/tracking.html)")
    args = parser.parse_args()
    if args.checkpoint is not None and args.tracker != "cotracker":
        parser.error("--checkpoint requires --tracker cotracker")
    config = ExperimentConfig(spacing=args.spacing, lattice_offset=tuple(args.offset), method=args.method)
    store = ResultsStore(args.results_dir)
    tracker: Tracker
    if args.tracker == "cotracker":
        from .cotracker import CoTracker3Tracker

        tracker = CoTracker3Tracker(args.checkpoint)
    else:
        tracker = OracleTracker()
    result = run_experiment(config, tracker, store)
    record = store.read(result.record.run_id)
    print(f"Run: {record.run_id}")
    print(f"Method: {record.configuration['method']}")
    print(f"Lattice baseline query budget: {record.query_count}")
    print(f"Full-field E_XX MAE: {record.metrics['full_field_exx_mae']}")
    print(f"Coverage loss: {record.metrics['coverage_loss']}")
    print(f"Fixed-radius E_XX MAE: {record.metrics['fixed_radius_exx_mae']}")
    print(f"Fixed-radius coverage loss: {record.metrics['fixed_radius_coverage_loss']}")
    print(f"Runtime: {record.runtime_seconds:.3f} s")
    print(f"Tracker runtime: {record.tracker_runtime_seconds} s")
    print(f"Peak VRAM: {record.peak_vram_bytes / 2**20:.1f} MiB")
    print(f"Immutable results: {store.path}")
    if args.visualize is not None:
        from .visualization import write_tracking_view

        path = write_tracking_view(result, args.visualize)
        print(f"Track playback: {path.resolve()}")


if __name__ == "__main__":
    main()
