"""Synthetic experiment pipeline, with an injected tracker."""
from dataclasses import asdict, dataclass
import hashlib
import json
from time import perf_counter
from uuid import uuid4

import numpy as np

from .adaptive import AdaptivePlan, RefinementConfig, adaptive_queries, prepare_adaptive
from .dataset import SyntheticCase, frozen_dataset, generator_identity
from .results import ResultsStore, RunRecord
from .sampling import lattice_baseline
from .specimen import dog_bone
from .strain import (BASELINE_RADIUS_SPACINGS, CONDITION_LIMIT, MINIMUM_NEIGHBOURS,
                     baseline_neighbour_counts, fixed_radius_strain, k_nearest_strain)
from .synthetic import SyntheticSequence, TensileMotion, UniformExtension, render_sequence
from .tracking import OracleTracker, Tracker, Tracks


@dataclass(frozen=True)
class ExperimentConfig:
    spacing: float = 6.4
    lattice_offset: tuple[float, float] = (0.0, 0.0)
    frame_count: int = 32
    score_every: int = 4
    final_stretch: float = 1.1
    seed: int = 0
    speckle_size: float = 2.5
    synthetic_case: SyntheticCase | None = None
    method: str = "lattice_baseline"
    refinement: RefinementConfig = RefinementConfig()

    def __post_init__(self) -> None:
        if self.method not in ("lattice_baseline", "adaptive"):
            raise ValueError("method must be lattice_baseline or adaptive")
        if self.score_every < 1 or self.frame_count <= self.score_every:
            raise ValueError("sequence must include a reference and at least one scored frame")
        if not np.isfinite(self.speckle_size) or not 1.5 <= self.speckle_size <= 5:
            raise ValueError("speckle size must be between 1.5 and 5 tracker-input px")
        UniformExtension(self.final_stretch)
        if self.synthetic_case is not None:
            case = self.synthetic_case
            if (self.frame_count, self.score_every, self.seed, self.speckle_size) != (
                    case.frame_count, case.score_every, case.seed, case.speckle_size):
                raise ValueError("experiment configuration must match its synthetic case; use from_case")

    @classmethod
    def from_case(cls, case: SyntheticCase, *, spacing: float = 6.4,
                  lattice_offset: tuple[float, float] = (0., 0.),
                  method: str = "lattice_baseline",
                  refinement: RefinementConfig = RefinementConfig()) -> "ExperimentConfig":
        return cls(spacing=spacing, lattice_offset=lattice_offset, frame_count=case.frame_count,
                   score_every=case.score_every, seed=case.seed, speckle_size=case.speckle_size,
                   final_stretch=float(np.sqrt(1 + 2 * case.uniform_strain)), synthetic_case=case,
                   method=method, refinement=refinement)


@dataclass(frozen=True)
class ExperimentResult:
    record: RunRecord
    sequence: SyntheticSequence
    query_points: np.ndarray
    tracks: Tracks
    evaluation_grid: np.ndarray
    scored_frames: np.ndarray
    strain: np.ndarray
    ground_truth_strain: np.ndarray
    fixed_radius_strain: np.ndarray
    adaptive_plan: AdaptivePlan | None = None
    coarse_tracks: Tracks | None = None


def _track_queries(tracker: Tracker, frames: np.ndarray, query_points: np.ndarray,
                   label: str = "tracker") -> Tracks:
    tracks = tracker.track(frames, query_points.copy())
    if (tracks.positions.shape != (len(frames), len(query_points), 2)
            or tracks.visibility.shape != (len(frames), len(query_points))):
        raise ValueError(f"{label} must return positions and visibility only for the given query points")
    return tracks


def run_experiment(config: ExperimentConfig, tracker: Tracker, store: ResultsStore,
                   *, motion: UniformExtension | TensileMotion | None = None) -> ExperimentResult:
    started = perf_counter()
    identity = generator_identity()
    specimen = dog_bone()
    query_points = lattice_baseline(specimen.sampling_region, config.spacing, config.lattice_offset)
    if len(query_points) < MINIMUM_NEIGHBOURS:
        raise ValueError("lattice baseline must contain at least six query points")
    baseline = query_points
    plan = (prepare_adaptive(specimen.sampling_region, config.spacing, config.lattice_offset)
            if config.method == "adaptive" else None)
    case = config.synthetic_case
    expected_motion = case.motion if case is not None else UniformExtension(config.final_stretch)
    motion = motion or expected_motion
    if asdict(motion) != asdict(expected_motion):
        raise ValueError("motion must match the configured synthetic sequence")
    if isinstance(tracker, OracleTracker) and asdict(tracker.motion) != asdict(motion):
        raise ValueError("oracle motion must match the synthetic sequence")
    sequence = render_sequence(specimen, motion, config.frame_count, config.seed, config.speckle_size,
                               case.noise_sigma if case else 0, case.blur_sigma if case else 0)
    coarse_tracks = None
    placement = None
    if plan is not None:
        coarse_tracks = _track_queries(tracker, sequence.frames, plan.coarse_points, "coarse tracker")
        placement = adaptive_queries(plan, coarse_tracks, config.refinement)
        query_points = placement.query_points
    tracks = _track_queries(tracker, sequence.frames, query_points)
    # Fixed 4 px evaluation grid, independent of the placement spacing/offset.
    evaluation_grid = lattice_baseline(specimen.gauge_section & specimen.eroded_mask, 4, (0, 0))
    scored_frames = np.arange(config.score_every, config.frame_count, config.score_every)
    radius = BASELINE_RADIUS_SPACINGS * config.spacing
    # Calibrate on the intact baseline geometry, before per-frame track loss.
    # Reuse these counts unchanged when comparing another placement method.
    k = baseline_neighbour_counts(baseline, evaluation_grid, radius)
    strain = np.stack([k_nearest_strain(query_points, tracks.positions[frame],
                                       tracks.visibility[frame], evaluation_grid, k)
                       for frame in scored_frames])
    secondary_strain = np.stack([fixed_radius_strain(query_points, tracks.positions[frame],
                                                    tracks.visibility[frame], evaluation_grid, radius)
                                 for frame in scored_frames])
    truth = np.stack([motion.strain_at(evaluation_grid, frame / (config.frame_count - 1))
                      for frame in scored_frames])
    metrics: dict[str, float | None] = {
        "tracking_failure_fraction": float(1 - (tracks.visibility[scored_frames] &
            np.isfinite(tracks.positions[scored_frames]).all(axis=-1)).mean())}
    for field, error_metric, coverage_metric in (
        (strain, "full_field_exx_mae", "coverage_loss"),
        (secondary_strain, "fixed_radius_exx_mae", "fixed_radius_coverage_loss"),
    ):
        defined = np.isfinite(field).all(axis=(-2, -1))
        errors = np.abs(field[..., 0, 0] - truth[..., 0, 0])
        metrics[error_metric] = float(errors[defined].mean()) if defined.any() else None
        metrics[coverage_metric] = float(1 - defined.mean())
    configuration = json.loads(json.dumps(asdict(config)))
    dataset = frozen_dataset()
    split = "dev" if case in dataset.dev else "test" if case in dataset.test else "custom"
    configuration.update({"method": config.method, "tracker": tracker.name,
                          "support_grid_count": tracker.support_grid_count,
                          "tracker_settings": tracks.configuration,
                          "generator_version": identity.version, "generator_sha256": identity.sha256,
                          "dataset_split": split, "numpy_version": np.__version__,
                          "estimator": "k_nearest",
                          "k": k.tolist(), "k_calibration": "intact_lattice_baseline",
                          "fixed_radius": radius,
                          "secondary_estimator": "fixed_radius",
                          "minimum_neighbours": MINIMUM_NEIGHBOURS, "condition_limit": CONDITION_LIMIT,
                          "erosion_px": 4, "evaluation_spacing": 4,
                          "scored_frames": scored_frames.tolist(),
                          "speckle_stratum": "1.5–3 px" if config.speckle_size < 3 else "3–5 px",
                          "max_inversion_residual_px": sequence.max_inversion_residual,
                          "frames_sha256": hashlib.sha256(sequence.frames.tobytes()).hexdigest()})
    runtime = tracks.runtime_seconds
    peak_vram = tracks.peak_vram_bytes
    if plan is not None and coarse_tracks is not None and placement is not None:
        configuration["adaptive"] = {
            "coarse_spacing": 1.5 * config.spacing,
            "refinement_spacing": config.spacing / 2,
            "coarse_query_count": len(plan.coarse_points),
            "coverage_query_count": len(plan.coverage_points),
            "coarse_runtime_seconds": coarse_tracks.runtime_seconds,
            "measurement_runtime_seconds": tracks.runtime_seconds,
            "coarse_peak_vram_bytes": coarse_tracks.peak_vram_bytes,
            "measurement_peak_vram_bytes": tracks.peak_vram_bytes,
            "coarse_tracker_settings": coarse_tracks.configuration,
            "region_scores": placement.region_scores.tolist(),
            "refinement_counts": placement.refinement_counts.tolist(),
            "visibility_gate_active": placement.visibility_gate_active,
            "visibility_gate_note": ("inactive: oracle tracking has full visibility"
                                     if tracker.name == "oracle" and not placement.visibility_gate_active
                                     else "local fraction of finite visible coarse tracks; suppresses only"),
            "score_reduction": "maximum of trailing-mean per-frame scores",
        }
        runtime = (runtime + coarse_tracks.runtime_seconds
                   if runtime is not None and coarse_tracks.runtime_seconds is not None else None)
        peak_vram = max(peak_vram, coarse_tracks.peak_vram_bytes)
    record = RunRecord(uuid4().hex, configuration, metrics, len(query_points), perf_counter() - started,
                       tracker_runtime_seconds=runtime, peak_vram_bytes=peak_vram)
    store.write(record)
    return ExperimentResult(record, sequence, query_points, tracks, evaluation_grid,
                            scored_frames, strain, truth, secondary_strain, plan, coarse_tracks)
