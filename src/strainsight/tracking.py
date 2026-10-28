"""Tracker boundary: images and the requested query points are the only inputs."""
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from .synthetic import UniformExtension


@dataclass(frozen=True)
class Tracks:
    positions: np.ndarray
    visibility: np.ndarray
    runtime_seconds: float | None = None
    peak_vram_bytes: int = 0
    configuration: dict[str, Any] = field(default_factory=dict)


class Tracker(Protocol):
    name: str
    support_grid_count: int

    def track(self, frames: np.ndarray, query_points: np.ndarray) -> Tracks: ...


class OracleTracker:
    name = "oracle"
    support_grid_count = 0

    def __init__(self, motion: UniformExtension | None = None) -> None:
        self.motion = motion or UniformExtension()

    def track(self, frames: np.ndarray, query_points: np.ndarray) -> Tracks:
        started = perf_counter()
        positions = np.stack([self.motion.forward(query_points, float(t))
                              for t in np.linspace(0, 1, len(frames))])
        return Tracks(positions, np.ones(positions.shape[:2], dtype=bool),
                      runtime_seconds=perf_counter() - started)
