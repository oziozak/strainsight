"""The uniform-extension experiment pipeline, with an injected tracker."""
from dataclasses import asdict, dataclass
import hashlib
import json
from time import perf_counter
from uuid import uuid4

import numpy as np

from .results import ResultsStore, RunRecord
from .sampling import lattice_baseline
from .specimen import dog_bone
from .strain import (BASELINE_RADIUS_SPACINGS, CONDITION_LIMIT, MINIMUM_NEIGHBOURS,
                     baseline_neighbour_counts, fixed_radius_strain, k_nearest_strain)
from .synthetic import GENERATOR_VERSION, SyntheticSequence, UniformExtension, render_uniform_extension
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

    def __post_init__(self) -> None:
        if self.score_every < 1 or self.frame_count <= self.score_every:
            raise ValueError("sequence must include a reference and at least one scored frame")
        if not np.isfinite(self.speckle_size) or not 1.5 <= self.speckle_size <= 5:
            raise ValueError("speckle size must be between 1.5 and 5 tracker-input px")
        UniformExtension(self.final_stretch)


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


def run_experiment(config: ExperimentConfig, tracker: Tracker, store: ResultsStore,
                   *, motion: UniformExtension | None = None) -> ExperimentResult:
    started = perf_counter()
    specimen = dog_bone()
    query_points = lattice_baseline(specimen.sampling_region, config.spacing, config.lattice_offset)
    if len(query_points) < MINIMUM_NEIGHBOURS:
        raise ValueError("lattice baseline must contain at least six query points")
    motion = motion or UniformExtension(config.final_stretch)
    if motion.final_stretch != config.final_stretch:
        raise ValueError("motion must match the configured final stretch")
    if isinstance(tracker, OracleTracker) and tracker.motion.final_stretch != motion.final_stretch:
        raise ValueError("oracle motion must match the synthetic sequence")
    sequence = render_uniform_extension(specimen, motion, config.frame_count,
                                       config.seed, config.speckle_size)
    tracks = tracker.track(sequence.frames, query_points.copy())
    if tracks.positions.shape != (config.frame_count, len(query_points), 2) or tracks.visibility.shape != (config.frame_count, len(query_points)):
        raise ValueError("tracker must return positions and visibility only for the given query points")
    # Fixed 4 px evaluation grid, independent of the placement spacing/offset.
    evaluation_grid = lattice_baseline(specimen.gauge_section & specimen.eroded_mask, 4, (0, 0))
    scored_frames = np.arange(config.score_every, config.frame_count, config.score_every)
    radius = BASELINE_RADIUS_SPACINGS * config.spacing
    # Calibrate on the intact baseline geometry, before per-frame track loss.
    # Reuse these counts unchanged when comparing another placement method.
    k = baseline_neighbour_counts(query_points, evaluation_grid, radius)
    strain = np.stack([k_nearest_strain(query_points, tracks.positions[frame],
                                       tracks.visibility[frame], evaluation_grid, k)
                       for frame in scored_frames])
    secondary_strain = np.stack([fixed_radius_strain(query_points, tracks.positions[frame],
                                                    tracks.visibility[frame], evaluation_grid, radius)
                                 for frame in scored_frames])
    truth = np.stack([np.broadcast_to(motion.strain(frame / (config.frame_count - 1)),
                                     (len(evaluation_grid), 2, 2)) for frame in scored_frames])
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
    configuration.update({"method": "lattice_baseline", "tracker": tracker.name,
                          "support_grid_count": tracker.support_grid_count,
                          "generator_version": GENERATOR_VERSION, "estimator": "k_nearest",
                          "k": k.tolist(), "k_calibration": "intact_lattice_baseline",
                          "fixed_radius": radius,
                          "secondary_estimator": "fixed_radius",
                          "minimum_neighbours": MINIMUM_NEIGHBOURS, "condition_limit": CONDITION_LIMIT,
                          "erosion_px": 4, "evaluation_spacing": 4,
                          "scored_frames": scored_frames.tolist(),
                          "speckle_stratum": "1.5–3 px" if config.speckle_size < 3 else "3–5 px",
                          "max_inversion_residual_px": sequence.max_inversion_residual,
                          "frames_sha256": hashlib.sha256(sequence.frames.tobytes()).hexdigest()})
    record = RunRecord(uuid4().hex, configuration, metrics, len(query_points), perf_counter() - started)
    store.write(record)
    return ExperimentResult(record, sequence, query_points, tracks, evaluation_grid,
                            scored_frames, strain, truth, secondary_strain)
