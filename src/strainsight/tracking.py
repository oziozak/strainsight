"""Tracker boundary: images and the requested query points are the only inputs."""
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .synthetic import TensileMotion, UniformExtension


@dataclass(frozen=True)
class Tracks:
    positions: np.ndarray
    visibility: np.ndarray


class Tracker(Protocol):
    name: str
    support_grid_count: int

    def track(self, frames: np.ndarray, query_points: np.ndarray) -> Tracks: ...


class OracleTracker:
    name = "oracle"
    support_grid_count = 0

    def __init__(self, motion: UniformExtension | TensileMotion | None = None) -> None:
        self.motion = motion or UniformExtension()

    def track(self, frames: np.ndarray, query_points: np.ndarray) -> Tracks:
        positions = np.stack([self.motion.forward(query_points, float(t))
                              for t in np.linspace(0, 1, len(frames))])
        return Tracks(positions, np.ones(positions.shape[:2], dtype=bool))
