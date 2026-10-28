"""Analytically textured reference specimen and inverse-mapped extension frames."""
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Protocol

import numpy as np

from .specimen import Specimen, points_in_mask

GENERATOR_VERSION = "tensile-v2"


class Motion(Protocol):
    def forward(self, points: np.ndarray, time: float) -> np.ndarray: ...

    def inverse(self, points: np.ndarray, time: float) -> np.ndarray: ...

    def strain_at(self, points: np.ndarray, time: float) -> np.ndarray: ...


@dataclass(frozen=True)
class TensileMotion:
    """Reference-configuration tensile map with exact spatial strain.

    Green–Lagrange E_XX is prescribed on the centreline. Integrating
    sqrt(1 + 2 E_XX) gives the axial map; equal lateral stretches in y and z
    preserve 3D volume. Off the centreline, the gradient includes the shear
    induced by spatially varying lateral contraction.
    """
    kind: Literal["uniform", "necking", "smooth"]
    uniform_strain: float = 0.1
    band_fwhm: float = 24
    band_center: float = 256
    peak_extra_strain: float = 0.4

    def __post_init__(self) -> None:
        if self.kind not in ("uniform", "necking", "smooth"):
            raise ValueError("unknown tensile deformation case")
        if not np.isfinite([self.uniform_strain, self.band_fwhm, self.band_center,
                            self.peak_extra_strain]).all():
            raise ValueError("motion parameters must be finite")
        if self.uniform_strain <= 0 or self.band_fwhm <= 0 or self.peak_extra_strain < 0:
            raise ValueError("strain and band width must be positive")

    def _profile(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.kind == "necking":
            relative = x - self.band_center
            factor = 4 * np.log(2) / self.band_fwhm**2
            extra = self.peak_extra_strain * np.exp(-factor * relative**2)
            return self.uniform_strain + extra, -2 * factor * relative * extra
        if self.kind == "smooth":
            # A broad 512 px wavelength control; no localized band.
            phase = 2 * np.pi * (x - self.band_center) / 512
            amplitude = 0.05 * self.peak_extra_strain
            return (self.uniform_strain + amplitude * np.cos(phase),
                    -amplitude * (2 * np.pi / 512) * np.sin(phase))
        return np.full_like(x, self.uniform_strain), np.zeros_like(x)

    def _stretch(self, x: np.ndarray, time: float) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(time) or not 0 <= time <= 1:
            raise ValueError("time must lie between zero and one")
        strain, derivative = self._profile(x)
        stretch = np.sqrt(1 + 2 * time * strain)
        return stretch, time * derivative / stretch

    @lru_cache(maxsize=64)
    def _axial_map(self, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Simpson integration on 1/64 px cells and cubic Hermite interpolation
        # keep the motion and its analytic derivative consistent to roundoff.
        x = np.linspace(-128, 640, 49153)
        stretch, _ = self._stretch(x, time)
        midpoint_stretch, _ = self._stretch((x[:-1] + x[1:]) / 2, time)
        integral = (stretch[:-1] + 4 * midpoint_stretch + stretch[1:] - 6) / 384
        displacement = np.concatenate(([0.], np.cumsum(integral)))
        # Equal and opposite motion of the reference grip ends (66, 446).
        displacement -= (np.interp(66, x, displacement) + np.interp(446, x, displacement)) / 2
        return x, x + displacement, stretch

    def _forward_x(self, positions: np.ndarray, time: float) -> np.ndarray:
        x, deformed_x, stretch = self._axial_map(time)
        if np.any((positions < x[0]) | (positions > x[-1])):
            raise ValueError("material positions lie outside the motion map")
        cell = np.clip(np.floor((positions - x[0]) * 64).astype(int), 0, len(x) - 2)
        fraction = (positions - x[cell]) * 64
        # Hermite polynomial in local displacement to avoid cancellation.
        step = deformed_x[cell + 1] - deformed_x[cell]
        left = stretch[cell] / 64
        right = stretch[cell + 1] / 64
        return (deformed_x[cell] + fraction * (left + fraction *
                (3 * step - 2 * left - right + fraction * (left + right - 2 * step))))

    def forward(self, points: np.ndarray, time: float) -> np.ndarray:
        stretch, _ = self._stretch(points[:, 0], time)
        return np.column_stack((self._forward_x(points[:, 0], time),
                                192 + (points[:, 1] - 192) * stretch**-0.5))

    def inverse(self, points: np.ndarray, time: float) -> np.ndarray:
        x, deformed_x, _ = self._axial_map(time)
        if np.any((points[:, 0] < deformed_x[0]) | (points[:, 0] > deformed_x[-1])):
            raise ValueError("image positions lie outside the motion map")
        reference_x = np.interp(points[:, 0], deformed_x, x)
        # Refine the linear lookup against the same smooth forward map.
        stretch, _ = self._stretch(reference_x, time)
        reference_x -= (self._forward_x(reference_x, time) - points[:, 0]) / stretch
        stretch, _ = self._stretch(reference_x, time)
        return np.column_stack((reference_x, 192 + (points[:, 1] - 192) * np.sqrt(stretch)))

    def deformation_gradient(self, points: np.ndarray, time: float) -> np.ndarray:
        stretch, derivative = self._stretch(points[:, 0], time)
        gradient = np.zeros((len(points), 2, 2))
        gradient[:, 0, 0] = stretch
        gradient[:, 1, 1] = stretch**-0.5
        gradient[:, 1, 0] = (points[:, 1] - 192) * (-0.5 * stretch**-1.5 * derivative)
        return gradient

    def strain_at(self, points: np.ndarray, time: float) -> np.ndarray:
        gradient = self.deformation_gradient(points, time)
        return (gradient.swapaxes(-1, -2) @ gradient - np.eye(2)) / 2

    @property
    def grip_displacement(self) -> float:
        return float(self.forward(np.array([[446., 192.]]), 1)[0, 0] - 446)


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

    def strain_at(self, points: np.ndarray, time: float) -> np.ndarray:
        return np.broadcast_to(self.strain(time), (len(points), 2, 2))


@dataclass(frozen=True)
class GroundTruth:
    displacement: np.ndarray
    strain: np.ndarray


@dataclass(frozen=True)
class SyntheticSequence:
    frames: np.ndarray
    specimen: Specimen
    motion: Motion
    max_inversion_residual: float

    def ground_truth(self, frame: int) -> GroundTruth:
        """Dense truth at reference pixel centres; outside the specimen is NaN."""
        if not 0 <= frame < len(self.frames):
            raise ValueError("ground-truth frame is outside the sequence")
        y, x = np.indices(self.specimen.specimen_mask.shape)
        points = np.column_stack((x.ravel(), y.ravel())).astype(float)
        time = frame / (len(self.frames) - 1)
        displacement = (self.motion.forward(points, time) - points).reshape(*x.shape, 2)
        strain = self.motion.strain_at(points, time).reshape(*x.shape, 2, 2).copy()
        displacement[~self.specimen.specimen_mask] = np.nan
        strain[~self.specimen.specimen_mask] = np.nan
        return GroundTruth(displacement, strain)


def render_uniform_extension(specimen: Specimen, motion: UniformExtension,
                             frame_count: int, seed: int, speckle_size: float) -> SyntheticSequence:
    """Compatibility entry point for the uniform-extension tracer bullet."""
    return render_sequence(specimen, motion, frame_count, seed, speckle_size)


def render_sequence(specimen: Specimen, motion: Motion, frame_count: int,
                    seed: int, speckle_size: float, noise_sigma: float = 0,
                    blur_sigma: float = 0) -> SyntheticSequence:
    """Inverse-map analytic Gaussian speckles with 2×2 pixel supersampling.

    Speckle size is reference Gaussian FWHM in tracker-input pixels. Texture
    is attached to the material; independent sensor noise is added per frame
    after Gaussian optical blur. Separate RNG streams keep the texture fixed
    when noise/blur change. Every inverse-mapped subpixel is residual-checked.
    """
    if frame_count < 2:
        raise ValueError("a synthetic sequence needs at least two frames")
    if not np.isfinite(speckle_size) or not 1.5 <= speckle_size <= 5:
        raise ValueError("speckle size must be between 1.5 and 5 tracker-input px")
    if not np.isfinite([noise_sigma, blur_sigma]).all() or min(noise_sigma, blur_sigma) < 0:
        raise ValueError("noise and blur must be finite and nonnegative")
    texture_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(texture_seed)
    noise_rng = np.random.default_rng(noise_seed)
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
        grey = np.zeros(len(pixels))
        for offset in ((-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)):
            subpixels = pixels + offset
            reference = motion.inverse(subpixels, float(time))
            residual = np.linalg.norm(motion.forward(reference, float(time)) - subpixels, axis=1)
            if not np.isfinite(reference).all() or not np.isfinite(residual).all() or residual.max() > 0.01:
                raise ValueError("inverse mapping residual exceeds 0.01 px")
            max_residual = max(max_residual, float(residual.max()))
            foreground = points_in_mask(specimen.specimen_mask, reference)
            reference = reference[foreground]
            intensity = np.zeros(len(reference))
            # Gaussian tails beyond these neighbouring cells are negligible.
            cell = np.floor((reference + 48) / 6).astype(int)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ix = np.clip(cell[:, 0] + dx, 0, centers.shape[1] - 1)
                    iy = np.clip(cell[:, 1] + dy, 0, centers.shape[0] - 1)
                    distance2 = np.sum((reference - centers[iy, ix])**2, axis=1)
                    intensity += np.exp(-distance2 / (2 * sigma**2))
            subpixel_grey = np.full(len(pixels), 24.)
            subpixel_grey[foreground] = 230 - 200 * np.clip(intensity, 0, 1)
            grey += subpixel_grey / 4
        image = _blur(grey.reshape(x.shape), blur_sigma)
        if noise_sigma:
            image += noise_rng.normal(0, noise_sigma, image.shape)
        frames[frame] = np.rint(np.clip(image, 0, 255)).astype(np.uint8)
    return SyntheticSequence(frames, specimen, motion, max_residual)


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma == 0:
        return image
    radius = int(np.ceil(4 * sigma))
    kernel = np.exp(-np.arange(-radius, radius + 1, dtype=float)**2 / (2 * sigma**2))
    kernel /= kernel.sum()
    for axis in (0, 1):
        padded = np.pad(image, [(radius, radius) if i == axis else (0, 0) for i in (0, 1)],
                        mode="edge")
        image = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), axis, padded)
    return image
