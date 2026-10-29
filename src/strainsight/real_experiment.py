"""Sample6 consistency evaluation using the shared placement and strain code."""
from dataclasses import asdict, dataclass, replace
from time import perf_counter
from uuid import uuid4

import numpy as np

from .adaptive import RefinementConfig, adaptive_queries, prepare_adaptive
from .real_data import PreparedSample6
from .results import ResultsStore, RunRecord, StrainSample
from .sampling import lattice_baseline
from .specimen import points_in_mask
from .strain import (BASELINE_RADIUS_SPACINGS, CONDITION_LIMIT, MINIMUM_NEIGHBOURS,
                     baseline_neighbour_counts, fixed_radius_strain, k_nearest_strain)
from .tracking import Tracker, Tracks


@dataclass(frozen=True)
class RealDataConfig:
    spacing: float = 8
    lattice_offset: tuple[float, float] = (0., 0.)
    dense_spacing: float = 2
    evaluation_spacing: float = 4
    score_every: int = 4
    refinement: RefinementConfig = RefinementConfig()

    def __post_init__(self) -> None:
        if (not np.isfinite(self.spacing) or not np.isfinite(self.dense_spacing)
                or not 0 < self.dense_spacing <= self.spacing / 2):
            raise ValueError("dense spacing must be positive and at most half the baseline spacing")
        if not np.isfinite(self.evaluation_spacing) or self.evaluation_spacing <= 0:
            raise ValueError("evaluation spacing must be finite and positive")
        if not isinstance(self.score_every, int) or self.score_every < 1:
            raise ValueError("score_every must be a positive integer")


@dataclass(frozen=True)
class RealDataRun:
    record: RunRecord
    query_points: np.ndarray
    tracks: Tracks
    full_frame_query_points: np.ndarray
    full_frame_tracks: Tracks
    strain: np.ndarray
    fixed_radius_strain: np.ndarray
    global_elongation: np.ndarray


@dataclass(frozen=True)
class Sample6Result:
    baseline: RealDataRun
    adaptive: RealDataRun
    dense_reference: RealDataRun
    evaluation_grid: np.ndarray
    scored_frames: np.ndarray


def _track(tracker: Tracker, frames: np.ndarray, queries: np.ndarray) -> Tracks:
    tracks = tracker.track(frames, queries.copy())
    expected = (len(frames), len(queries))
    if (tracks.positions.shape != (*expected, 2) or tracks.visibility.shape != expected
            or tracks.visibility.dtype != np.bool_):
        raise ValueError("tracker must return positions and Boolean visibility for the given query points")
    return tracks


def _elongation(queries: np.ndarray, tracks: Tracks, gauge: np.ndarray) -> np.ndarray:
    """Global affine loading-axis displacement slope, with translation/shear removed."""
    inside = points_in_mask(gauge, queries)
    elongation = np.full(len(tracks.positions), np.nan)
    for frame, positions in enumerate(tracks.positions):
        valid = inside & tracks.visibility[frame] & np.isfinite(positions).all(axis=1)
        if valid.sum() < MINIMUM_NEIGHBOURS:
            continue
        reference = queries[valid]
        design = np.column_stack((reference - reference.mean(axis=0), np.ones(valid.sum())))
        if np.linalg.cond(design) > CONDITION_LIMIT:
            continue
        coefficients, _, rank, _ = np.linalg.lstsq(design, positions[valid, 0] - reference[:, 0], rcond=None)
        if rank == 3:
            elongation[frame] = coefficients[0]
    return elongation


def _trend_correlation(elongation: np.ndarray, extension: np.ndarray) -> float | None:
    valid = np.isfinite(elongation) & np.isfinite(extension)
    if valid.sum() < 3 or np.ptp(elongation[valid]) < 1e-12 or np.ptp(extension[valid]) < 1e-12:
        return None
    return float(np.corrcoef(elongation[valid], extension[valid])[0, 1])


def run_sample6(data: PreparedSample6, tracker: Tracker, store: ResultsStore,
                grip_extension: np.ndarray, config: RealDataConfig = RealDataConfig(), *,
                frame_ids: list[str] | None = None,
                grip_extension_units: str = "machine_units") -> Sample6Result:
    """Run dense reference, baseline and adaptive on identical cropped inputs.

    Grip extension must already be aligned one-to-one with the supplied frames.
    Dense agreement is consistency only; there is no real-data ground truth.
    """
    extension = np.asarray(grip_extension, dtype=float)
    if extension.shape != (len(data.frames),) or not np.isfinite(extension).all():
        raise ValueError("grip extension must be finite and aligned one-to-one with frames")
    if frame_ids is not None and (len(frame_ids) != len(data.frames) or len(set(frame_ids)) != len(frame_ids)):
        raise ValueError("frame IDs must be distinct and aligned one-to-one with frames")
    if not grip_extension_units.strip():
        raise ValueError("grip extension units must be stated")
    scored = np.arange(config.score_every, len(data.frames), config.score_every)
    if not len(scored):
        raise ValueError("sequence must contain at least one scored frame")
    specimen = data.specimen
    baseline = lattice_baseline(specimen.sampling_region, config.spacing, config.lattice_offset)
    if len(baseline) < MINIMUM_NEIGHBOURS:
        raise ValueError("lattice baseline must contain at least six query points")
    dense_offset = tuple(float(v % config.dense_spacing) for v in config.lattice_offset)
    dense = lattice_baseline(specimen.sampling_region, config.dense_spacing,
                             (dense_offset[0], dense_offset[1]))
    if len(dense) < 4 * len(baseline):
        raise ValueError("dense reference must contain at least four times the baseline query count")
    plan = prepare_adaptive(specimen.sampling_region, config.spacing, config.lattice_offset)
    grid = lattice_baseline(specimen.gauge_section & specimen.eroded_mask,
                            config.evaluation_spacing, (0, 0))
    if not len(grid):
        raise ValueError("eroded gauge section must contain evaluation positions")
    radius = BASELINE_RADIUS_SPACINGS * config.spacing
    k = baseline_neighbour_counts(baseline, grid, radius)
    shared = {
        **asdict(config), "lattice_offset": list(config.lattice_offset),
        "dataset": "TelecentricSample6", "dataset_split": "real",
        "frame_count": len(data.frames), "loading_axis": "x", "tracker": tracker.name,
        "frame_ids": frame_ids if frame_ids is not None else [str(i) for i in range(len(data.frames))],
        "support_grid_count": tracker.support_grid_count, "erosion_px": 4,
        "frames_sha256": data.frames_sha256, "masks_sha256": data.masks_sha256,
        "mask_source": "external_reference_annotation", "edges_verified_across_sequence": True,
        "roi": {"x": data.roi.x, "y": data.roi.y, "width": data.roi.width, "height": data.roi.height},
        "full_frame_shape": list(data.full_frame_shape), "tracker_input_shape": list(data.frames.shape[1:3]),
        "downscale_factor_xy": data.roi.downscale_xy.tolist(),
        "coordinate_mapping": "full_xy = (input_xy + 0.5) * downscale_xy - 0.5 + roi_origin",
        "strain_coordinate_system": "tracker_input_px", "numpy_version": np.__version__,
        "estimator": "k_nearest", "secondary_estimator": "fixed_radius", "k": k.tolist(),
        "k_calibration": "intact_lattice_baseline", "fixed_radius": radius,
        "minimum_neighbours": MINIMUM_NEIGHBOURS, "condition_limit": CONDITION_LIMIT,
        "scored_frames": scored.tolist(), "dense_reference_role": "consistency_only_not_ground_truth",
        "grip_extension_role": "global_trend_sanity_only_not_extensometer",
        "grip_extension_delta": (extension - extension[0]).tolist(),
        "grip_extension_units": grip_extension_units,
        "global_elongation_definition": "global affine loading-axis displacement slope over visible gauge queries",
        "limitations": [
            "1365x1024 ROI downscaled by %.6g in x and %.6g in y; speckles and narrow strain bands may be unresolved."
            % tuple(data.roi.downscale_xy),
            "1365x1024 approximates 4:3; x/y scale factors differ slightly.",
            "Dense reference shares tracker errors and is a consistency target, not ground truth.",
            "Machine grip extension includes grip/system compliance; correlation is sanity only.",
            "Lateral-edge containment over the sequence is human verified from the external images.",
        ],
    }
    runs: dict[str, RealDataRun] = {}
    for method, queries in (("dense_reference", dense), ("lattice_baseline", baseline), ("adaptive", baseline)):
        started = perf_counter()
        adaptive_configuration = {}
        coarse = None
        if method == "adaptive":
            coarse = _track(tracker, data.frames, plan.coarse_points)
            placement = adaptive_queries(plan, coarse, config.refinement)
            queries = placement.query_points
            adaptive_configuration = {
                "coarse_query_count": len(plan.coarse_points), "coverage_query_count": len(plan.coverage_points),
                "coarse_runtime_seconds": coarse.runtime_seconds, "coarse_peak_vram_bytes": coarse.peak_vram_bytes,
                "coarse_tracker_settings": coarse.configuration,
                "region_scores": placement.region_scores.tolist(),
                "refinement_counts": placement.refinement_counts.tolist(),
                "visibility_gate_active": placement.visibility_gate_active,
            }
        tracks = _track(tracker, data.frames, queries)
        field = np.stack([k_nearest_strain(queries, tracks.positions[frame], tracks.visibility[frame], grid, k)
                          for frame in scored])
        secondary = np.stack([fixed_radius_strain(queries, tracks.positions[frame], tracks.visibility[frame], grid, radius)
                              for frame in scored])
        elongation = _elongation(queries, tracks, specimen.gauge_section & specimen.eroded_mask)
        metrics: dict[str, float | None] = {
            "tracking_failure_fraction": float(1 - (tracks.visibility[scored]
                & np.isfinite(tracks.positions[scored]).all(axis=-1)).mean()),
            "grip_extension_trend_correlation": _trend_correlation(elongation, extension),
        }
        for estimator, tensors in (("k_nearest", field), ("fixed_radius", secondary)):
            defined = np.isfinite(tensors).all(axis=(-2, -1))
            prefix = "" if estimator == "k_nearest" else "fixed_radius_"
            metrics[prefix + "coverage_loss"] = float(1 - defined.mean())
            if method != "dense_reference":
                reference = runs["dense_reference"]
                target = reference.strain if estimator == "k_nearest" else reference.fixed_radius_strain
                common = defined & np.isfinite(target).all(axis=(-2, -1))
                errors = np.abs(tensors[..., 0, 0] - target[..., 0, 0])
                metrics[prefix + "dense_reference_loading_axis_mae"] = float(errors[common].mean()) if common.any() else None
                metrics[prefix + "dense_comparison_coverage_loss"] = float(1 - common.mean())
        configuration = {
            **shared, "method": method, "tracker_settings": tracks.configuration,
            "dense_reference_run_id": runs["dense_reference"].record.run_id if runs else None,
            "global_elongation": [float(value) if np.isfinite(value) else None for value in elongation],
        }
        if method == "adaptive":
            configuration["adaptive"] = {
                **adaptive_configuration, "measurement_runtime_seconds": tracks.runtime_seconds,
                "measurement_peak_vram_bytes": tracks.peak_vram_bytes,
            }
        samples = {
            estimator: [StrainSample(int(frame), float(x), float(y),
                                     float(tensor[0, 0]) if np.isfinite(tensor).all() else None,
                                     float(tensor[1, 1]) if np.isfinite(tensor).all() else None,
                                     float(tensor[0, 1]) if np.isfinite(tensor).all() else None)
                        for frame, tensors in zip(scored, values) for (x, y), tensor in zip(grid, tensors)]
            for estimator, values in (("k_nearest", field), ("fixed_radius", secondary))
        }
        runtime = tracks.runtime_seconds
        peak = tracks.peak_vram_bytes
        if coarse is not None:
            runtime = (runtime + coarse.runtime_seconds
                       if runtime is not None and coarse.runtime_seconds is not None else None)
            peak = max(peak, coarse.peak_vram_bytes)
        record = RunRecord(uuid4().hex, configuration, metrics, len(queries), perf_counter() - started,
                           runtime, peak, samples)
        store.write(record)
        runs[method] = RealDataRun(record, queries, tracks, data.roi.to_full_frame(queries),
                                   replace(tracks, positions=data.roi.to_full_frame(tracks.positions)),
                                   field, secondary, elongation)
    return Sample6Result(runs["lattice_baseline"], runs["adaptive"], runs["dense_reference"], grid, scored)
