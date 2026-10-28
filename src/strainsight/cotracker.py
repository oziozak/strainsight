"""Optional CoTracker3 offline adapter; importing it does not load the GPU stack."""
from pathlib import Path
from time import perf_counter

import numpy as np

from .tracking import Tracks


class CoTracker3Tracker:
    """Track reference-frame queries using the predictor's fixed default rules.

    Inputs are grayscale (T,H,W) or RGB (T,H,W,3), in the 0..255 range.
    Positions use input-image (x,y) pixels; resizing and the 6x6 support grid
    belong to the upstream predictor. No confidence gate or replacement queries
    are added. Model construction/download is excluded from per-run timing.
    """

    name = "cotracker3_offline"
    support_grid_count = 36
    checkpoint_revision = "bf55ea50d4390e1820a267f131cd6587240fb2c5"

    def __init__(self, checkpoint: str | Path | None = None, *, device: str = "cuda") -> None:
        import torch  # type: ignore[import-not-found]
        from cotracker.predictor import CoTrackerPredictor  # type: ignore[import-untyped,import-not-found]
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CoTracker3 requires CUDA; install the gpu extra on a CUDA host")
        if self.device.type not in {"cuda", "cpu"}:
            raise ValueError("CoTracker3 supports cuda or cpu devices")
        self.checkpoint = str(checkpoint) if checkpoint is not None else hf_hub_download(
            "facebook/cotracker3", "scaled_offline.pth", revision=self.checkpoint_revision)
        self._predictor = CoTrackerPredictor(checkpoint=self.checkpoint, offline=True,
                                             window_len=60).to(self.device).eval()

    def track(self, frames: np.ndarray, query_points: np.ndarray) -> Tracks:
        import torch  # type: ignore[import-not-found]

        frames = np.asarray(frames)
        query_points = np.asarray(query_points)
        if frames.ndim == 3:
            frames = np.repeat(frames[..., None], 3, axis=-1)
        if (frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] < 2
                or min(frames.shape[1:3]) < 2):
            raise ValueError("frames must have shape (T,H,W) or (T,H,W,3), with T,H,W >= 2")
        if not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 255:
            raise ValueError("frames must contain finite intensities in 0..255")
        if (query_points.ndim != 2 or query_points.shape[1] != 2 or not len(query_points)
                or not np.isfinite(query_points).all()):
            raise ValueError("query points must be a nonempty finite (N,2) array")
        if ((query_points < 0).any()
                or (query_points > [frames.shape[2] - 1, frames.shape[1] - 1]).any()):
            raise ValueError("query points must lie inside the input image")

        # Pin context independently of the method, query budget, and previous run.
        self._predictor.support_grid_size = 6
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = perf_counter()
        with torch.inference_mode():
            video = torch.from_numpy(np.ascontiguousarray(frames.transpose(0, 3, 1, 2))).to(
                device=self.device, dtype=torch.float32)[None]
            queries = np.column_stack((np.zeros(len(query_points)), query_points)).astype(np.float32)
            query_tensor = torch.from_numpy(queries).to(self.device)[None]
            positions, visibility = self._predictor(video, queries=query_tensor,
                                                   backward_tracking=False)
            # The offline predictor already applies visibility > 0.9, ignoring
            # confidence, and strips auxiliary support points. Keep its mask.
            positions_array = positions[0].cpu().numpy()
            visibility_array = visibility[0].cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_vram_bytes = torch.cuda.max_memory_allocated(self.device)
        else:
            peak_vram_bytes = 0
        runtime_seconds = perf_counter() - started
        expected = (len(frames), len(query_points))
        if (positions_array.shape != (*expected, 2) or visibility_array.shape != expected
                or visibility_array.dtype != np.bool_):
            raise ValueError("CoTracker3 must return positions and Boolean visibility for requested queries only")
        return Tracks(positions_array, visibility_array, runtime_seconds, peak_vram_bytes,
                      configuration={"checkpoint": self.checkpoint, "offline": True,
                                     "window_len": 60, "support_grid_size": 6,
                                     "visibility_threshold": 0.9, "confidence_used": False,
                                     "device": str(self.device)})
