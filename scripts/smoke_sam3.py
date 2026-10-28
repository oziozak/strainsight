"""Opt-in GPU smoke check: SAM 3.1 local install on the target GPU (milestone 1).

Segments a synthetic speckled dog-bone specimen on a single reference frame with a
text prompt and a box prompt, and reports IoU against the known mask, wall time,
and peak VRAM. Needs Hugging Face access to the gated facebook/sam3.1 repo.

    python scripts/smoke_sam3.py [--out outputs/smoke_sam3.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_predictor

H, W = 1024, 1365  # Sample6-sized 4:3 ROI


def dogbone(rng: np.random.Generator) -> tuple[Image.Image, np.ndarray]:
    """Horizontal dog-bone with speckle on a dark background, plus its mask."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, gauge_half, grip_half = H / 2, 120, 200
    x_gauge0, x_gauge1 = 380, W - 380
    # Gauge narrows to gauge_half between the fillets, grips at full width.
    half = np.full(W, grip_half, np.float32)
    ramp = 120
    x = np.arange(W, dtype=np.float32)
    left = np.clip((x - (x_gauge0 - ramp)) / ramp, 0, 1)
    right = np.clip(((x_gauge1 + ramp) - x) / ramp, 0, 1)
    t = np.minimum(left, right)
    half = grip_half - (grip_half - gauge_half) * (0.5 - 0.5 * np.cos(np.pi * t))
    mask = (np.abs(yy - cy) <= half[None, :]) & (xx >= 60) & (xx <= W - 60)

    img = np.full((H, W), 25, np.float32) + rng.normal(0, 3, (H, W))
    speckle = np.zeros((H, W), np.float32)
    for x0, y0, r in zip(rng.uniform(0, W, 9000), rng.uniform(0, H, 9000), rng.uniform(2, 6, 9000)):
        x_lo, x_hi, y_lo, y_hi = int(max(x0 - 2 * r, 0)), int(min(x0 + 2 * r + 1, W)), int(max(y0 - 2 * r, 0)), int(min(y0 + 2 * r + 1, H))
        d = (xx[y_lo:y_hi, x_lo:x_hi] - x0) ** 2 + (yy[y_lo:y_hi, x_lo:x_hi] - y0) ** 2
        speckle[y_lo:y_hi, x_lo:x_hi] = np.maximum(speckle[y_lo:y_hi, x_lo:x_hi], d <= r * r)
    img = np.where(mask, 210 - 170 * speckle, img)
    rgb = np.repeat(np.clip(img, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    return Image.fromarray(rgb), mask


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/smoke_sam3.json")
    args = ap.parse_args()

    image, truth = dogbone(np.random.default_rng(0))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    image.save(Path(args.out).with_suffix(".png"))

    t0 = time.perf_counter()
    predictor = build_sam3_predictor(version="sam3.1", use_fa3=False)  # FA3 is Hopper-only; no sm_120 build
    load_s = time.perf_counter() - t0

    # Upstream bug at the pinned sam3 commit: Sam3BasePredictor.start_session passes
    # offload_state_to_cpu, which the SAM 3.1 multiplex init_state does not accept.
    init_state = predictor.model.init_state
    predictor.model.init_state = lambda offload_state_to_cpu=False, **kw: init_state(**kw)

    ys, xs = np.nonzero(truth)
    box_xywh = [xs.min() / W, ys.min() / H, (xs.max() - xs.min()) / W, (ys.max() - ys.min()) / H]
    prompts = {
        "text:specimen": dict(text="specimen"),
        "text:metal strip": dict(text="metal strip"),
        "box": dict(bounding_boxes=[box_xywh], bounding_box_labels=[1]),
    }

    results = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "model_load_s": load_s, "image_hw": [H, W], "prompts": []}
    for name, prompt in prompts.items():
        session = predictor.handle_request({"type": "start_session", "resource_path": [image]})
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        resp = predictor.handle_request(
            {"type": "add_prompt", "session_id": session["session_id"], "frame_index": 0, **prompt}
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        out = resp["outputs"]
        masks = out["out_binary_masks"]
        masks = masks.cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
        union = masks.any(axis=0) if len(masks) else np.zeros_like(truth)
        row = {
            "prompt": name,
            "n_objects": int(len(masks)),
            "scores": np.asarray(out["out_probs"]).round(3).tolist(),
            "iou_vs_truth": iou(union, truth),
            "time_s": dt,
            "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
        }
        results["prompts"].append(row)
        print(json.dumps(row))
        predictor.handle_request({"type": "close_session", "session_id": session["session_id"]})

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
