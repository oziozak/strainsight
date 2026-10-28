"""Run the uniform-extension tracer bullet and save an immutable run record."""
import argparse
from pathlib import Path

from .experiment import ExperimentConfig, run_experiment
from .results import ResultsStore
from .tracking import OracleTracker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    parser.add_argument("--spacing", type=float, default=6.4)
    parser.add_argument("--offset", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"))
    args = parser.parse_args()
    config = ExperimentConfig(spacing=args.spacing, lattice_offset=tuple(args.offset))
    store = ResultsStore(args.results_dir)
    result = run_experiment(config, OracleTracker(), store)
    record = store.read(result.record.run_id)
    print(f"Run: {record.run_id}")
    print(f"Lattice baseline query budget: {record.query_count}")
    print(f"Full-field E_XX MAE: {record.metrics['full_field_exx_mae']}")
    print(f"Coverage loss: {record.metrics['coverage_loss']}")
    print(f"Runtime: {record.runtime_seconds:.3f} s")
    print(f"Immutable results: {store.path}")


if __name__ == "__main__":
    main()
