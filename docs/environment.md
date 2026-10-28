# Environment and measured GPU facts (milestone 1)

Measured 2026-10-28 on the RTX 5070 Ti PC (WSL2, driver 610.88, 16 GB). Reproduce with the opt-in smoke checks in `scripts/`; they write JSON to `outputs/` (git-ignored).

## Environment

One environment for everything; no separate SAM environment was needed.

| Item | Value |
|---|---|
| Python | 3.12 (`requires-python = ">=3.12"`) |
| PyTorch | 2.11.0+cu128, torchvision 0.26.0+cu128 (sm_120 supported) |
| CoTracker | `facebookresearch/co-tracker` @ `82e02e8` |
| SAM 3 / 3.1 | `facebookresearch/sam3` @ `f1647e8` |

```bash
uv sync --extra dev --extra gpu     # Linux + CUDA only; on the Mac use --extra dev
uv run python scripts/smoke_cotracker.py
uv run python scripts/smoke_sam3.py # needs HF access to facebook/sam3.1
```

The `gpu` extra pulls torch from the PyTorch cu128 index and the two model repos from pinned GitHub commits (see `[tool.uv.sources]` in `pyproject.toml`).

## CoTracker3 offline

Checkpoint `facebook/cotracker3` / `scaled_offline.pth`, `CoTrackerPredictor(offline=True, window_len=60)`, 32-frame synthetic speckle sequence, queries on frame 0, median of 5 timed runs after a warm-up. Per-run time includes the predictor's resize and its 6×6 support grid.

| Input frame | N queries | Time / run | Peak VRAM |
|---|---|---|---|
| 384×512 | 315 | 0.34 s | 4.6 GB |
| 384×512 | 520 | 0.45 s | 4.6 GB |
| 384×512 | 745 | 0.59 s | 4.6 GB |
| 384×512 | 1290 | 0.90 s | 4.6 GB |
| 1024×1365 | 315–1290 | 0.34–0.92 s | 5.1 GB |

Resize confirmed end to end: both input sizes reach the network as `[1, 32, 3, 384, 512]` (`model.model_resolution`). `CoTrackerPredictor` bilinearly resizes the frames (`align_corners=True`), scales the queries in, and scales the tracks back to input px. On a known translation the median last-frame error was 0.23 px at 384×512 and 0.74 full-res px (≈ 0.28 tracker px) at 1024×1365. Visibility is thresholded at > 0.9 in the offline predictor, which matches the spec's lost-track rule.

### Tracker workload

≈ 1,150 runs × 0.34–0.92 s ≈ **7–18 min of tracker time** (budget-curve N from 315 to 1290). This replaces the assumed 1–5 s/run (20 min – 1.6 h). Not included: the one-off model load and frame generation, and real-data sequences longer than 32 frames (time scales with frame count).

## SAM 3.1

Hugging Face access to the gated `facebook/sam3.1` repo is **granted**. The checkpoint (`sam3.1_multiplex.pt`) downloads and runs locally. SAM 3.1 ships only as the multiplex video predictor (`build_sam3_predictor(version="sam3.1")`); a single reference frame is passed as a one-image list. Model load takes ≈ 14 s.

Single 1024×1365 synthetic speckled dog-bone frame, prompts on frame 0:

| Prompt | Objects | IoU vs true mask | Time | Peak VRAM |
|---|---|---|---|---|
| Box (specimen bbox) | 1 (score 0.99) | 0.986 | 0.33 s | 4.5 GB |
| Text "specimen" | 0 | – | 0.93 s | 4.4 GB |
| Text "metal strip" | 0 | – | 0.18 s | 4.4 GB |

Text prompts found nothing on the synthetic frame. Whether they work on real Sample6 frames is unchecked, so plan on box prompts for the specimen mask. Point prompts exist in the API (`points`, `point_labels`) but were not tested.

Required workarounds at the pinned commit (both in `scripts/smoke_sam3.py`):

- `use_fa3=False`: the builder defaults to Flash Attention 3, which is Hopper-only and has no sm_120 build.
- `Sam3BasePredictor.start_session` passes `offload_state_to_cpu`, which the multiplex `init_state` does not accept. Drop the kwarg until this is fixed upstream.
