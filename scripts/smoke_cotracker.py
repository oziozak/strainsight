"""Opt-in GPU smoke check: time CoTracker3 offline on the target GPU (milestone 1).

Renders a synthetic speckle sequence (32 frames, the spec's sequence length) with a
known uniform translation, tracks query sets at the budget-curve sizes, and records
per-run wall time, peak VRAM, and the frame size the model actually receives.

    python scripts/smoke_cotracker.py [--out outputs/smoke_cotracker.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from cotracker.predictor import CoTrackerPredictor

FRAMES = 32
TRACKER_HW = (384, 512)
BUDGETS = [315, 520, 745, 1290]  # N along the budget curve (docs/experiment_spec.md)
SHIFT_PX_PER_FRAME = (0.5, 0.25)  # (dx, dy) in px at the rendered resolution


def speckle_video(h: int, w: int, rng: np.random.Generator) -> torch.Tensor:
    """Random dot speckle translated by a known sub-pixel shift each frame."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    n_dots = int(h * w / 40)
    cx = rng.uniform(-40, w + 40, n_dots).astype(np.float32)
    cy = rng.uniform(-40, h + 40, n_dots).astype(np.float32)
    radius = rng.uniform(1.5, 3.0, n_dots).astype(np.float32) * (h / TRACKER_HW[0])
    frames = []
    for t in range(FRAMES):
        dx, dy = SHIFT_PX_PER_FRAME[0] * t, SHIFT_PX_PER_FRAME[1] * t
        img = np.zeros((h, w), np.float32)
        for x0, y0, r in zip(cx + dx, cy + dy, radius):
            x_lo, x_hi = max(int(x0 - 3 * r), 0), min(int(x0 + 3 * r) + 1, w)
            y_lo, y_hi = max(int(y0 - 3 * r), 0), min(int(y0 + 3 * r) + 1, h)
            if x_lo >= x_hi or y_lo >= y_hi:
                continue
            patch = (xx[y_lo:y_hi, x_lo:x_hi] - x0) ** 2 + (yy[y_lo:y_hi, x_lo:x_hi] - y0) ** 2
            img[y_lo:y_hi, x_lo:x_hi] = np.maximum(
                img[y_lo:y_hi, x_lo:x_hi], np.exp(-patch / (2 * (r / 2) ** 2))
            )
        frames.append(np.repeat((40 + 200 * img)[None], 3, axis=0))
    return torch.from_numpy(np.stack(frames))[None].float()  # B T C H W, 0..255


def queries_for(n: int, h: int, w: int, rng: np.random.Generator) -> torch.Tensor:
    margin = 0.15
    xs = rng.uniform(margin * w, (1 - margin) * w, n)
    ys = rng.uniform(margin * h, (1 - margin) * h, n)
    q = np.stack([np.zeros(n), xs, ys], axis=1).astype(np.float32)
    return torch.from_numpy(q)[None]  # B N 3 (t, x, y)


def timed_run(model, video, queries):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tracks, vis = model(video, queries=queries)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 2**20, tracks, vis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/smoke_cotracker.json")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    ckpt = hf_hub_download("facebook/cotracker3", "scaled_offline.pth")
    model = CoTrackerPredictor(checkpoint=ckpt, offline=True, window_len=60).cuda().eval()

    # The predictor calls model.forward() directly (bypassing hooks), so wrap it to
    # record the frame size the network actually receives.
    seen_shapes: list[tuple[int, ...]] = []
    inner_forward = model.model.forward

    def recording_forward(*a, **kw):
        seen_shapes.append(tuple(kw["video"].shape))
        return inner_forward(*a, **kw)

    model.model.forward = recording_forward

    rng = np.random.default_rng(0)
    results = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "model_resolution": list(model.interp_shape),
        "frames": FRAMES,
        "runs": [],
    }

    # Native tracker-input size (synthetic pipeline) and a 4:3 real-data ROI size
    # (Sample6 crop 1365x1024) to confirm the resize path end to end.
    for h, w in [TRACKER_HW, (1024, 1365)]:
        video = speckle_video(h, w, rng).cuda()
        for n in BUDGETS:
            queries = queries_for(n, h, w, rng).cuda()
            with torch.no_grad():
                seen_shapes.clear()
                timed_run(model, video, queries)  # warm-up
                times, peaks = [], []
                for _ in range(args.repeats):
                    dt, peak, tracks, vis = timed_run(model, video, queries)
                    times.append(dt)
                    peaks.append(peak)
            expected = queries[0, :, 1:].cpu() + torch.tensor(SHIFT_PX_PER_FRAME) * (FRAMES - 1)
            err = (tracks[0, -1].cpu() - expected).norm(dim=-1)
            row = {
                "input_hw": [h, w],
                "n_queries": n,
                "model_input_shape": list(seen_shapes[-1]),
                "output_tracks_shape": list(tracks.shape),
                "time_s_median": statistics.median(times),
                "time_s_all": times,
                "peak_vram_mib": max(peaks),
                "last_frame_err_px_median": err.median().item(),
                "visible_frac": vis.float().mean().item(),
            }
            results["runs"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "time_s_all"}))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
