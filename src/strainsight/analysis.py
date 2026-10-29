"""Read-only evaluation of the pre-registered decision rule over selected runs."""
from collections import defaultdict
from dataclasses import dataclass, replace
from math import fsum
from typing import Sequence

import numpy as np

from .results import ResultsStore, RunRecord


BOOTSTRAP_RESAMPLES = 10_000
CONFIDENCE_LEVEL = .95
NONINFERIORITY_MARGIN = .05


@dataclass(frozen=True, order=True)
class ComparisonCondition:
    spacing: float
    tracker: str
    estimator: str


NOMINAL_CONDITION = ComparisonCondition(6.4, "cotracker3_offline", "k_nearest")
_SEQUENCE_FIELDS = ("synthetic_case", "speckle_stratum", "generator_version", "generator_sha256",
                    "frames_sha256", "frame_count", "score_every", "scored_frames")
_PAIR_FIELDS = _SEQUENCE_FIELDS + (
    "tracker_settings", "support_grid_count", "erosion_px", "evaluation_spacing",
    "k", "k_calibration", "fixed_radius", "minimum_neighbours", "condition_limit",
)


@dataclass(frozen=True)
class MetricComparison:
    pair_count: int
    sequence_count: int
    baseline_mean: float | None
    adaptive_mean: float | None
    difference: float | None
    relative_change: float | None


@dataclass(frozen=True)
class PrimaryDecision:
    condition: ComparisonCondition
    pair_count: int
    sequence_count: int
    high_strain: MetricComparison
    high_strain_ci: tuple[float, float]
    high_strain_improves: bool
    full_field: MetricComparison
    noninferiority_margin: float
    full_field_noninferior: bool

    @property
    def passes(self) -> bool:
        return self.high_strain_improves and self.full_field_noninferior


@dataclass(frozen=True)
class ComparisonSummary:
    pair_count: int
    sequence_count: int
    metrics: dict[str, MetricComparison]

    @property
    def full_field(self) -> MetricComparison:
        return self.metrics["full_field_exx_mae"]

    @property
    def high_strain(self) -> MetricComparison:
        return self.metrics["high_strain_exx_mae"]


@dataclass(frozen=True)
class DescriptiveComparison:
    condition: ComparisonCondition
    pooled: ComparisonSummary
    per_stratum: dict[str, ComparisonSummary]


@dataclass(frozen=True)
class AnalysisReport:
    run_ids: tuple[str, ...]
    seed: int
    bootstrap_resamples: int
    confidence_level: float
    primary: PrimaryDecision
    descriptive: tuple[DescriptiveComparison, ...]


@dataclass(frozen=True)
class _Pair:
    sequence: int
    offset: tuple[float, float]
    baseline: RunRecord
    adaptive: RunRecord

    @property
    def kind(self) -> str:
        return self.baseline.configuration["synthetic_case"]["kind"]


def _summary(pairs: list[_Pair]) -> ComparisonSummary:
    names = {"full_field_exx_mae", "high_strain_exx_mae"}
    for pair in pairs:
        names.update(pair.baseline.metrics)
        names.update(pair.adaptive.metrics)
    metrics = {name: _metric([p for p in pairs if p.kind == "necking"]
                             if name == "high_strain_exx_mae" else pairs, name)
               for name in sorted(names)}
    return ComparisonSummary(len(pairs), len({p.sequence for p in pairs}), metrics)


def _estimator_records(record: RunRecord) -> list[RunRecord]:
    """Expose the runner's bundled fixed-radius metrics as a separate comparison."""
    if record.configuration["estimator"] != "k_nearest":
        return [record]
    primary = {name: value for name, value in record.metrics.items()
               if not name.startswith("fixed_radius_")}
    records = [replace(record, metrics=primary)]
    secondary = {name.removeprefix("fixed_radius_"): value
                 for name, value in record.metrics.items() if name.startswith("fixed_radius_")}
    if secondary:
        if "exx_mae" in secondary:
            secondary["full_field_exx_mae"] = secondary.pop("exx_mae")
        config = dict(record.configuration, estimator="fixed_radius")
        records.append(replace(record, configuration=config, metrics=secondary))
    return records


def _metric(pairs: list[_Pair], name: str) -> MetricComparison:
    baseline = [pair.baseline.metrics.get(name) for pair in pairs]
    adaptive = [pair.adaptive.metrics.get(name) for pair in pairs]
    baseline_mean = _mean(baseline)
    adaptive_mean = _mean(adaptive)
    difference = None
    relative = None
    if baseline_mean is not None and adaptive_mean is not None:
        # Use the paired statistic directly, avoiding cancellation of rounded means.
        difference = fsum(a - b for a, b in zip(adaptive, baseline)
                          if a is not None and b is not None) / len(pairs)
        if baseline_mean != 0:
            relative = difference / baseline_mean
        elif adaptive_mean == 0:
            relative = 0.
    return MetricComparison(len(pairs), len({pair.sequence for pair in pairs}),
                            baseline_mean, adaptive_mean, difference, relative)


def _mean(values: list[float | None]) -> float | None:
    if not values or any(value is None or not np.isfinite(value) for value in values):
        return None
    return fsum(value for value in values if value is not None) / len(values)


def _cluster_interval(pairs: list[_Pair], seed: int) -> tuple[float, float]:
    clusters: dict[int, list[float]] = defaultdict(list)
    for pair in pairs:
        baseline = pair.baseline.metrics["high_strain_exx_mae"]
        adaptive = pair.adaptive.metrics["high_strain_exx_mae"]
        assert baseline is not None and adaptive is not None
        clusters[pair.sequence].append(adaptive - baseline)
    sums = np.array([fsum(values) for values in clusters.values()])
    counts = np.array([len(values) for values in clusters.values()])
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.integers(len(clusters), size=(BOOTSTRAP_RESAMPLES, len(clusters)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def _require_mae(pairs: list[_Pair], metric: str) -> None:
    for pair in pairs:
        for record in (pair.baseline, pair.adaptive):
            value = record.metrics.get(metric)
            if value is None or not np.isfinite(value) or value < 0:
                raise ValueError(f"{record.run_id}: nominal {metric} must be finite and nonnegative")


def _validate_metadata(record: RunRecord) -> None:
    config = record.configuration
    try:
        case = config["synthetic_case"]
        offset = config["lattice_offset"]
        valid = (
            isinstance(case, dict) and type(case["seed"]) is int
            and case["kind"] in {"necking", "uniform", "smooth"}
            and config["speckle_stratum"] in {"1.5–3 px", "3–5 px"}
            and np.isfinite(config["spacing"]) and config["spacing"] > 0
            and len(offset) == 2 and np.isfinite(offset).all()
            and isinstance(config["tracker"], str) and bool(config["tracker"])
            and config["estimator"] in {"k_nearest", "fixed_radius"}
            and config["method"] in {"lattice_baseline", "adaptive"}
            and type(record.query_count) is int and record.query_count > 0
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{record.run_id}: invalid or missing test-run metadata")


def analyze_results(store: ResultsStore, run_ids: Sequence[str], *, seed: int = 0) -> AnalysisReport:
    """Analyze only selected completed runs; see docs/decision_rule.md for the contract."""
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("duplicate selected run IDs")
    cells: dict[tuple[ComparisonCondition, int, tuple[float, float]], dict[str, RunRecord]] = {}
    sequences: dict[int, RunRecord] = {}
    for run_id in run_ids:
        record = store.read(run_id)
        config = record.configuration
        if config.get("dataset_split") in {"dev", "custom"}:
            continue
        if config.get("dataset_split") != "test":
            raise ValueError(f"{run_id}: invalid or missing dataset_split metadata")
        _validate_metadata(record)
        sequence = config["synthetic_case"]["seed"]
        previous = sequences.setdefault(sequence, record)
        for field in _SEQUENCE_FIELDS:
            if previous.configuration.get(field) != config.get(field):
                raise ValueError(f"inconsistent sequence {sequence}: {field} differs in {run_id}")
        for observation in _estimator_records(record):
            config = observation.configuration
            condition = ComparisonCondition(config["spacing"], config["tracker"], config["estimator"])
            key = (condition, config["synthetic_case"]["seed"], tuple(config["lattice_offset"]))
            method = config["method"]
            methods = cells.setdefault(key, {})
            if method in methods:
                raise ValueError(f"ambiguous pair: duplicate {method} at {key}")
            methods[method] = observation
    groups: dict[ComparisonCondition, list[_Pair]] = defaultdict(list)
    for (condition, sequence, offset), methods in sorted(cells.items()):
        if set(methods) != {"lattice_baseline", "adaptive"}:
            raise ValueError(f"incomplete pair at {(condition, sequence, offset)}")
        baseline, adaptive = methods["lattice_baseline"], methods["adaptive"]
        if baseline.query_count != adaptive.query_count:
            raise ValueError(f"query budgets differ at {(condition, sequence, offset)}")
        for field in _PAIR_FIELDS:
            if baseline.configuration.get(field) != adaptive.configuration.get(field):
                raise ValueError(f"paired {field} differs at {(condition, sequence, offset)}")
        groups[condition].append(_Pair(sequence, offset, methods["lattice_baseline"], methods["adaptive"]))
    pairs = groups.get(NOMINAL_CONDITION, [])
    if not pairs:
        raise ValueError("selected runs contain no nominal CoTracker k-nearest pairs at 6.4 px")
    necking = [pair for pair in pairs if pair.kind == "necking"]
    if len({pair.sequence for pair in necking}) < 2:
        raise ValueError("nominal bootstrap requires at least two independent necking test sequences")
    if not any(pair.kind in {"uniform", "smooth"} for pair in pairs):
        raise ValueError("nominal full-field comparison requires control sequences")
    _require_mae(pairs, "full_field_exx_mae")
    _require_mae(necking, "high_strain_exx_mae")
    full = _metric(pairs, "full_field_exx_mae")
    high = _metric(necking, "high_strain_exx_mae")
    interval = _cluster_interval(necking, seed)
    assert full.adaptive_mean is not None and full.baseline_mean is not None
    improves = high.difference is not None and high.difference < 0 and interval[1] < 0
    decision = PrimaryDecision(NOMINAL_CONDITION, len(pairs), len({p.sequence for p in pairs}),
                               high, interval, improves, full, NONINFERIORITY_MARGIN,
                               full.adaptive_mean <= (1 + NONINFERIORITY_MARGIN) * full.baseline_mean)
    descriptive = []
    for condition, condition_pairs in sorted(groups.items()):
        strata: dict[str, list[_Pair]] = defaultdict(list)
        for pair in condition_pairs:
            strata[pair.baseline.configuration["speckle_stratum"]].append(pair)
        descriptive.append(DescriptiveComparison(condition, _summary(condition_pairs),
                           {stratum: _summary(values) for stratum, values in sorted(strata.items())}))
    return AnalysisReport(tuple(run_ids), seed, BOOTSTRAP_RESAMPLES, CONFIDENCE_LEVEL,
                          decision, tuple(descriptive))
