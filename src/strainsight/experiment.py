"""Synthetic experiment pipeline, with an injected tracker."""
from dataclasses import asdict, dataclass
import hashlib
import json
from time import perf_counter
from uuid import uuid4

import numpy as np

from .dataset import SyntheticCase, frozen_dataset, generator_identity
from .results import ResultsStore, RunRecord, StrainSample
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

    def __post_init__(self) -> None:
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
                  lattice_offset: tuple[float, float] = (0., 0.)) -> "ExperimentConfig":
        return cls(spacing=spacing, lattice_offset=lattice_offset, frame_count=case.frame_count,
                   score_every=case.score_every, seed=case.seed, speckle_size=case.speckle_size,
                   final_stretch=float(np.sqrt(1 + 2 * case.uniform_strain)), synthetic_case=case)


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
                   *, motion: UniformExtension | TensileMotion | None = None) -> ExperimentResult:
    started = perf_counter()
    identity = generator_identity()
    specimen = dog_bone()
    query_points = lattice_baseline(specimen.sampling_region, config.spacing, config.lattice_offset)
    if len(query_points) < MINIMUM_NEIGHBOURS:
        raise ValueError("lattice baseline must contain at least six query points")
    case = config.synthetic_case
    expected_motion = case.motion if case is not None else UniformExtension(config.final_stretch)
    motion = motion or expected_motion
    if asdict(motion) != asdict(expected_motion):
        raise ValueError("motion must match the configured synthetic sequence")
    if isinstance(tracker, OracleTracker) and asdict(tracker.motion) != asdict(motion):
        raise ValueError("oracle motion must match the synthetic sequence")
    sequence = render_sequence(specimen, motion, config.frame_count, config.seed, config.speckle_size,
                               case.noise_sigma if case else 0, case.blur_sigma if case else 0)
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
    configuration.update({"method": "lattice_baseline", "tracker": tracker.name,
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
    strain_samples = {}
    for estimator, field in (("k_nearest", strain), ("fixed_radius", secondary_strain)):
        strain_samples[estimator] = [
            StrainSample(int(frame), float(x), float(y),
                         float(tensor[0, 0]) if np.isfinite(tensor).all() else None,
                         float(tensor[1, 1]) if np.isfinite(tensor).all() else None,
                         float(tensor[0, 1]) if np.isfinite(tensor).all() else None)
            for frame, tensors in zip(scored_frames, field)
            for (x, y), tensor in zip(evaluation_grid, tensors)]
    record = RunRecord(uuid4().hex, configuration, metrics, len(query_points), perf_counter() - started,
                       tracker_runtime_seconds=tracks.runtime_seconds,
                       peak_vram_bytes=tracks.peak_vram_bytes, strain_samples=strain_samples)
    store.write(record)
    return ExperimentResult(record, sequence, query_points, tracks, evaluation_grid,
                            scored_frames, strain, truth, secondary_strain)
