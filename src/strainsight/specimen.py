"""Reference geometry in tracker-input pixels, shared by placement and scoring."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Specimen:
    specimen_mask: np.ndarray
    eroded_mask: np.ndarray
    sampling_region: np.ndarray
    gauge_section: np.ndarray


def dog_bone() -> Specimen:
    """512×384 dog-bone: W=64, gauge=256, radius=64, grips=102, length=380.

    Circular fillets meet the grips 45.508 px beyond each gauge end.
    The remaining 16.492 px of each grip is visible (approximately 0.25W).
    Masks classify pixel centres; erosion uses a Euclidean disk of radius 4.
    """
    y, x = np.indices((384, 512))
    axial = np.abs(x - 256)
    fillet_length = np.sqrt(2 * 64 * 19 - 19**2)
    beyond_gauge = np.clip(axial - 128, 0, fillet_length)
    half_width = 32 + 64 - np.sqrt(64**2 - beyond_gauge**2)
    half_width = np.where(axial >= 128 + fillet_length, 51, half_width)
    mask = (axial <= 190) & (np.abs(y - 192) <= half_width)
    padded = np.pad(mask, 4, constant_values=False)
    eroded = mask.copy()
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            if dx * dx + dy * dy <= 16:
                eroded &= padded[4 + dy:388 + dy, 4 + dx:516 + dx]
    return Specimen(mask, eroded, eroded & (axial < 128 + fillet_length),
                    mask & (axial <= 128))


def points_in_mask(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Membership uses the nearest reference pixel centre, including subpixels."""
    indices = np.floor(points + 0.5).astype(int)
    inside = ((indices[:, 0] >= 0) & (indices[:, 0] < mask.shape[1]) &
              (indices[:, 1] >= 0) & (indices[:, 1] < mask.shape[0]))
    valid = np.flatnonzero(inside)
    inside[valid] = mask[indices[valid, 1], indices[valid, 0]]
    return inside
