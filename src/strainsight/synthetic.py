"""Analytically textured reference specimen and inverse-mapped extension frames."""
from dataclasses import dataclass

import numpy as np

from .specimen import Specimen, points_in_mask

GENERATOR_VERSION = "uniform-extension-v1"


@dataclass(frozen=True)
class UniformExtension:
    final_stretch: float = 1.1

    def __post_init__(self) -> None:
        if not np.isfinite(self.final_stretch) or self.final_stretch <= 0:
            raise ValueError("final stretch must be finite and positive")

    def deformation_gradient(self, time: float) -> np.ndarray:
        stretch = 1 + time * (self.final_stretch - 1)
        # In-plane lateral contraction for an incompressible 3D uniaxial solid.
        return np.diag([stretch, stretch**-0.5])

    def forward(self, points: np.ndarray, time: float) -> np.ndarray:
        return (points - [256, 192]) @ self.deformation_gradient(time).T + [256, 192]

    def inverse(self, points: np.ndarray, time: float) -> np.ndarray:
        return (points - [256, 192]) @ np.linalg.inv(self.deformation_gradient(time)).T + [256, 192]

    def strain(self, time: float) -> np.ndarray:
        gradient = self.deformation_gradient(time)
        return (gradient.T @ gradient - np.eye(2)) / 2


@dataclass(frozen=True)
class SyntheticSequence:
    frames: np.ndarray
    specimen: Specimen
    motion: UniformExtension
    max_inversion_residual: float


def render_uniform_extension(specimen: Specimen, motion: UniformExtension,
                             frame_count: int, seed: int, speckle_size: float) -> SyntheticSequence:
    """Evaluate analytic Gaussian speckles at inverse-mapped pixel centres.

    Speckle size is Gaussian FWHM in reference tracker-input pixels. The
    continuous texture is shared by all frames, with no per-frame randomness.
    """
    rng = np.random.default_rng(seed)
    # Random Gaussian dots on a jittered 6 px lattice, evaluated continuously.
    dot_x, dot_y = np.meshgrid(np.arange(-48, 561, 6), np.arange(-48, 433, 6))
    centers = np.stack((dot_x, dot_y), axis=-1).astype(float)
    centers += rng.uniform(-2, 2, centers.shape)
    y, x = np.indices(specimen.specimen_mask.shape)
    pixels = np.column_stack((x.ravel(), y.ravel())).astype(float)
    frames = np.empty((frame_count, *x.shape), dtype=np.uint8)
    max_residual = 0.0
    sigma = speckle_size / np.sqrt(8 * np.log(2))
    for frame, time in enumerate(np.linspace(0, 1, frame_count)):
        reference = motion.inverse(pixels, float(time))
        residual = np.linalg.norm(motion.forward(reference, float(time)) - pixels, axis=1)
        if not np.isfinite(reference).all() or not np.isfinite(residual).all() or residual.max() > 0.01:
            raise ValueError("inverse mapping residual exceeds 0.01 px")
        max_residual = max(max_residual, float(residual.max()))
        foreground = points_in_mask(specimen.specimen_mask, reference)
        intensity = np.zeros(len(pixels))
        # Only nearby analytic dots contribute (tails beyond neighbouring cells
        # are negligible for the largest supported 5 px FWHM).
        cell = np.floor((reference + 48) / 6).astype(int)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ix = np.clip(cell[:, 0] + dx, 0, centers.shape[1] - 1)
                iy = np.clip(cell[:, 1] + dy, 0, centers.shape[0] - 1)
                distance2 = np.sum((reference - centers[iy, ix])**2, axis=1)
                intensity += np.exp(-distance2 / (2 * sigma**2))
        grey = np.where(foreground, 230 - 200 * np.clip(intensity, 0, 1), 24)
        frames[frame] = grey.reshape(x.shape).astype(np.uint8)
    return SyntheticSequence(frames, specimen, motion, max_residual)
