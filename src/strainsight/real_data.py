"""External Sample6 images, reference masks, and tracker-input coordinates."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .specimen import Specimen


@dataclass(frozen=True)
class ReferenceMasks:
    """Human-defined ROI pixel masks, before resizing or erosion."""

    specimen_mask: np.ndarray
    sampling_region: np.ndarray
    gauge_section: np.ndarray


class Segmenter(Protocol):
    def segment(self, reference_frame: np.ndarray) -> ReferenceMasks: ...


class HandMaskSegmenter:
    """Read an external NPZ annotation; no automatic mask or inferred grips."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def segment(self, reference_frame: np.ndarray) -> ReferenceMasks:
        with np.load(self.path, allow_pickle=False) as annotation:
            if not {"specimen_mask", "sampling_region", "gauge_section"} <= set(annotation.files):
                raise ValueError("hand annotation must contain specimen_mask, sampling_region and gauge_section")
            masks = ReferenceMasks(*(annotation[name].copy() for name in
                                     ("specimen_mask", "sampling_region", "gauge_section")))
        _validate_masks(masks, reference_frame.shape[:2])
        return masks


@dataclass(frozen=True)
class Sample6ROI:
    """Fixed horizontal loading axis; all coordinates refer to pixel centres."""

    x: int
    y: int

    width = 1365
    height = 1024
    tracker_width = 512
    tracker_height = 384

    def __post_init__(self) -> None:
        if any(not isinstance(v, int) or v < 0 for v in (self.x, self.y)):
            raise ValueError("ROI origin must contain nonnegative integers")

    @property
    def downscale_xy(self) -> np.ndarray:
        return np.array([self.width / self.tracker_width, self.height / self.tracker_height])

    def to_full_frame(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points) + .5) * self.downscale_xy - .5 + [self.x, self.y]

    def to_tracker_input(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points) - [self.x, self.y] + .5) / self.downscale_xy - .5

    def crop(self, frames: np.ndarray) -> np.ndarray:
        height, width = frames.shape[1:3]
        if self.x + self.width > width or self.y + self.height > height:
            raise ValueError("1365x1024 specimen ROI must fit inside every full frame")
        if (width, height) == (self.width, self.height):
            raise ValueError("supply full frames and a specimen ROI smaller than the full frame")
        return frames[:, self.y:self.y + self.height, self.x:self.x + self.width]


@dataclass(frozen=True)
class PreparedSample6:
    frames: np.ndarray
    specimen: Specimen
    roi: Sample6ROI
    full_frame_shape: tuple[int, int]
    frames_sha256: str
    masks_sha256: str


def _validate_masks(masks: ReferenceMasks, shape: tuple[int, ...]) -> None:
    for mask in (masks.specimen_mask, masks.sampling_region, masks.gauge_section):
        if mask.shape != shape or mask.dtype != np.bool_ or not mask.any():
            raise ValueError("reference masks must be nonempty Boolean arrays matching the specimen ROI")
    if (np.any(masks.sampling_region & ~masks.specimen_mask)
            or np.any(masks.gauge_section & ~masks.sampling_region)):
        raise ValueError("gauge section must be inside sampling region, inside specimen mask")
    if masks.specimen_mask[[0, -1]].any():
        raise ValueError("both lateral specimen edges must lie inside the specimen ROI")


def prepare_sample6(frames: np.ndarray, roi: Sample6ROI, segmenter: Segmenter,
                    *, edges_verified: bool) -> PreparedSample6:
    """Segment only the reference ROI, resize all crops, then erode by 4 px.

    A human must check lateral-edge containment over the entire sequence;
    the single reference mask cannot verify that property in later frames.
    """
    if not edges_verified:
        raise ValueError("human verification of both specimen edges across the sequence is required")
    frames = np.asarray(frames)
    if (frames.ndim not in (3, 4) or (frames.ndim == 4 and frames.shape[-1] != 3)
            or len(frames) < 2 or not np.isfinite(frames).all()
            or frames.min() < 0 or frames.max() > 255):
        raise ValueError("full frames must be finite grayscale or RGB images in 0..255, with T >= 2")
    crops = roi.crop(frames)
    masks = segmenter.segment(crops[0].copy())
    _validate_masks(masks, crops.shape[1:3])
    # Pixel-centre bilinear resize, shared with the inverse coordinate map.
    x = (np.arange(roi.tracker_width) + .5) * roi.downscale_xy[0] - .5
    y = (np.arange(roi.tracker_height) + .5) * roi.downscale_xy[1] - .5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    wx, wy = x - x0, y - y0
    if frames.ndim == 4:
        wx, wy = wx[:, None], wy[:, None, None]
    else:
        wy = wy[:, None]
    top = crops[:, y0[:, None], x0] * (1 - wx) + crops[:, y0[:, None], x0 + 1] * wx
    bottom = crops[:, y0[:, None] + 1, x0] * (1 - wx) + crops[:, y0[:, None] + 1, x0 + 1] * wx
    resized = np.rint(top * (1 - wy) + bottom * wy).astype(np.uint8)
    resized_masks = [mask[np.floor(y + .5).astype(int)[:, None], np.floor(x + .5).astype(int)]
                     for mask in (masks.specimen_mask, masks.sampling_region, masks.gauge_section)]
    specimen_mask, sampling_region, gauge_section = resized_masks
    eroded = specimen_mask.copy()
    padded = np.pad(specimen_mask, 4, constant_values=False)
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            if dx * dx + dy * dy <= 16:
                eroded &= padded[4 + dy:388 + dy, 4 + dx:516 + dx]
    specimen = Specimen(specimen_mask, eroded, eroded & sampling_region, gauge_section)
    if not specimen.sampling_region.any() or not (gauge_section & eroded).any():
        raise ValueError("reference annotation leaves no sampling or scoring area after erosion")
    return PreparedSample6(resized, specimen, roi, (frames.shape[1], frames.shape[2]),
                           hashlib.sha256(resized.tobytes()).hexdigest(),
                           hashlib.sha256(b"".join(mask.tobytes() for mask in
                               (masks.specimen_mask, masks.sampling_region, masks.gauge_section))).hexdigest())


def load_sample6_frames(paths: list[Path]) -> np.ndarray:
    """Read an explicitly ordered list; no filename or timestamp inference."""
    from PIL import Image  # type: ignore[import-not-found]

    if len(paths) < 2:
        raise ValueError("supply at least two ordered Sample6 image paths")
    images = []
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image)
            if array.ndim != 2 and (array.ndim != 3 or array.shape[-1] != 3):
                raise ValueError("Sample6 frames must be grayscale or RGB images")
            if array.dtype != np.uint8:
                raise ValueError("Sample6 images must be 8-bit; choose and document any intensity conversion")
            images.append(array.copy())
    return np.stack(images)


@dataclass(frozen=True)
class Sample6Source:
    frames: np.ndarray
    frame_ids: list[str]
    grip_extension: np.ndarray
    grip_extension_units: str


def load_sample6_manifest(path: str | Path) -> Sample6Source:
    """Load explicitly ordered frames and human-aligned machine measurements.

    Relative image paths resolve against the external manifest's directory.
    The manifest must state the machine units; no timestamp interpolation or
    assumption about the machine's CSV column layout is made here.
    """
    path = Path(path)
    manifest = json.loads(path.read_text())
    frame_ids = manifest.get("frames")
    if (not isinstance(frame_ids, list) or len(frame_ids) < 2
            or any(not isinstance(name, str) or not name for name in frame_ids)):
        raise ValueError("manifest frames must be an ordered list of at least two image paths")
    paths = [(path.parent / name).resolve() for name in frame_ids]
    if len(set(paths)) != len(paths):
        raise ValueError("manifest must not repeat a frame path")
    extension = np.asarray(manifest.get("grip_extension"), dtype=float)
    if extension.shape != (len(paths),) or not np.isfinite(extension).all():
        raise ValueError("grip extension must be finite and aligned one-to-one with frames")
    units = manifest.get("grip_extension_units")
    if not isinstance(units, str) or not units.strip():
        raise ValueError("manifest must state grip_extension_units")
    return Sample6Source(load_sample6_frames(paths), frame_ids, extension, units)
