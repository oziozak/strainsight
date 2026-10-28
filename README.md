# StrainSight

Measuring deformation of a stretched specimen from video, and testing whether measurement-guided query placement beats a lattice baseline at the same query budget.

The first complete experiment runs a synthetic uniform-extension sequence through lattice placement, oracle or CoTracker3 offline tracking, a k-nearest strain estimator, full-field evaluation, and an immutable results store. Oracle tracking runs on CPU with NumPy; CoTracker3 uses the optional GPU environment. Adaptive placement, the SAM 3.1 pipeline adapter, and real-data evaluation are future work.

## Run the experiment

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev                 # add --extra gpu on the CUDA PC (CoTracker3, SAM 3.1)
uv run pytest
uv run python -m strainsight.demo
```

The demo renders 32 frames at 512×384 px and scores every fourth frame after the reference frame. It prints the run ID, query count, full-field E_XX mean absolute error, coverage loss, runtime, and results-store path. Oracle tracking should give MAE near floating-point precision and zero coverage loss.

The `gpu` extra keeps torch and torchvision on the PyTorch cu128 index and CoTracker3 / SAM 3.1 at pinned Git commits. GPU setup, smoke checks, and measured runtime / VRAM are documented in [docs/environment.md](docs/environment.md).

Choose a baseline spacing, a sub-spacing lattice offset, and a results directory:

```bash
uv run python -m strainsight.demo --spacing 8 --offset 1.5 2.5 --results-dir outputs/results
```

On a CUDA host with the `gpu` extra installed, select CoTracker3 offline:

```bash
uv run python -m strainsight.demo --tracker cotracker
# Optionally use an existing scaled_offline.pth instead of the pinned HF download:
uv run python -m strainsight.demo --tracker cotracker --checkpoint checkpoints/scaled_offline.pth
uv run python scripts/smoke_cotracker_adapter.py
```

The adapter uses the offline predictor's visibility rule (`visibility > 0.9`, confidence unused) and its fixed 6×6 support grid. These 36 auxiliary points are identical for every query set and excluded from the query budget and strain estimates. Lost queries stay in the budget, are excluded only in frames where invisible, and are never replaced. The opt-in smoke check uses the real CUDA model at two budgets and checks RGB resizing as well as persisted resources; it is outside the default test suite.

Watch the query points move over the actual frames:

```bash
uv run python -m strainsight.demo --tracker cotracker --visualize
# Or choose a filename; oracle tracking also supports --visualize:
uv run python -m strainsight.demo --visualize outputs/oracle-tracks.html
```

Open the printed HTML path in your browser (`outputs/tracking.html` by default).
The viewer includes play/pause, frame stepping, a scrubber, speed control, colored
query points, short trajectory trails, and optional reference positions. Red
crosses mark invisible predictions; per-frame visible/lost counts include
non-finite positions, which have no marker. The HTML embeds the frames and tracks
and works offline without extra packages or a server. The file is written from
that invocation's tracking output; existing SQLite records contain summary
metrics rather than saved video/tracks, so rerun with `--visualize` to inspect them.

Records live in `outputs/results/runs.sqlite3` by default. Each run gets a distinct ID; inserting another record with that ID raises `FileExistsError`. SQLite transactions make writes atomic. Read a completed record through the store:

```python
from strainsight.results import ResultsStore

record = ResultsStore("outputs/results").read("<run ID printed by the demo>")
print(record.configuration, record.metrics, record.query_count, record.runtime_seconds)
print(record.tracker_runtime_seconds, record.peak_vram_bytes)
```

## What the tracer bullet measures

- The reference specimen is a horizontal dog-bone with a 256×64 px gauge section, circular 64 px fillets, 102 px grips, and a 380 px visible reference length. These dimensions give 16.492 px of each grip in view (approximately 0.25 gauge widths).
- A 4 px Euclidean disk erodes the reference specimen mask. Only the gauge section and fillets in that eroded mask receive query points; grips receive none. Masks classify pixel centres, and subpixel membership uses the nearest pixel centre.
- The lattice baseline's count defines the query budget. At zero offset, spacings 4.0, 5.3, 6.4, and 8.0 px give 1357, 759, 523, and 317 query points, respectively. Counts vary with lattice offset.
- A seeded analytic Gaussian speckle texture is evaluated through the inverse reference-configuration motion map at each output pixel. Nonfinite inversions or forward–inverse residuals above 0.01 px reject the sequence before tracking.
- Uniform extension ramps from stretch 1 to 1.1 with incompressible lateral contraction. Strain is 2D Green–Lagrange strain: the final E_XX is 0.105, and E_YY is approximately −0.04545.
- The tracker boundary takes only frames and query points and returns their positions, Boolean visibility, and resource measurements. The oracle returns true positions only at those queries and uses no support grid. For a custom stretch, supply `OracleTracker(UniformExtension(config.final_stretch))`; mismatched oracle configurations are rejected. Injected trackers declare their name and support-grid count. `strainsight.cotracker.CoTracker3Tracker` accepts grayscale `(T,H,W)` or RGB `(T,H,W,3)` arrays in 0–255 and queries as `(N,2)` input-image `(x,y)` coordinates, all placed in the reference frame. Its GPU imports and checkpoint loading happen only when constructed.
- The unweighted affine least-squares estimator uses k = round(π × 2.5²) = 20 nearest visible query points in the reference configuration. Fewer than six neighbours, deficient rank, or a design condition number above 10⁶ give undefined strain.
- Evaluation uses a fixed 4 px grid over the eroded gauge section, independent of placement. Full-field E_XX MAE averages defined positions across scored frames; coverage loss reports the undefined fraction separately. If every estimate is undefined, MAE is `None`, stored as JSON `null`. Lost tracks retain their place in the query budget and are never replaced.
- Records contain configuration, the generator version and rendered-frame SHA-256, speckle stratum, query count, metrics, total pipeline wall time (rendering through evaluation, excluding persistence), tracker wall time, and peak allocated VRAM in bytes (zero for CPU). CoTracker timing synchronizes CUDA and includes input conversion, device transfers, predictor resizing, and output transfer; model loading/download is excluded. Peak VRAM is reset per tracking call and includes the resident model, inputs, and inference allocations, rather than CUDA's reserved cache. The run configuration records checkpoint path, device, support-grid size/count, and visibility settings. Older records without resource fields remain readable (`tracker_runtime_seconds=None`, `peak_vram_bytes=0`). This tracer bullet uses synthetic development/sanity sequences; held-out benchmark runs and their pre-registration are later work.

## Development

```bash
uv run python -m mypy src/strainsight tests
uv run python -m pytest tests/test_pipeline.py
uv run python -m pytest
```

Pipeline tests inject trackers at the agreed experiment boundary and check the physics, geometry, budget, independent evaluation, loss handling, inverse-warp rejection, and results-store immutability. The old farthest-point sampler and two-point engineering strain estimator remain only as legacy sanity checks in `tests/test_core.py`.

`src/strainsight/` contains the specimen geometry, synthetic generator, placement, tracker interface, strain estimator, experiment runner, results store, and demo. The full protocol and future milestones are in [docs/experiment_spec.md](docs/experiment_spec.md), with vocabulary in [GLOSSARY.md](GLOSSARY.md) and the budget/estimator decision in [ADR 0001](docs/adr/0001-budget-and-primary-estimator.md).

Do not commit DIC Challenge images or derived artifacts before licensing is verified. This experiment generates its own texture and requires no external datasets or model weights.

## License

MIT; see [LICENSE](LICENSE).
