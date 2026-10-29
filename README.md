# StrainSight

Measuring deformation of a stretched specimen from video, and testing whether measurement-guided query placement beats a lattice baseline at the same query budget.

The experiment runs synthetic uniform-extension, localized-necking, and smooth control sequences through lattice placement, oracle or CoTracker3 offline tracking, k-nearest and fixed-radius strain estimators, full-field evaluation, and an immutable results store. A frozen generator defines 8 dev and 24 test sequences from one distribution. Oracle tracking runs on CPU with NumPy; CoTracker3 uses the optional GPU environment. Adaptive placement, the SAM 3.1 pipeline adapter, and real-data evaluation are future work.

## Run the experiment

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev                 # add --extra gpu on the CUDA PC (CoTracker3, SAM 3.1)
uv run pytest
uv run python -m strainsight.demo
```

The demo renders 32 frames at 512×384 px and scores every fourth frame after the reference frame. It prints the run ID, query count, full-field E_XX mean absolute error and coverage loss for both estimator variants, runtime, and results-store path. Oracle tracking should give MAE near floating-point precision and zero coverage loss.

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
- A seeded analytic Gaussian speckle texture is evaluated through the inverse reference-configuration motion map with 2×2 pixel supersampling. Nonfinite inversions or forward–inverse residuals above 0.01 px reject the sequence before tracking.
- Uniform extension ramps from stretch 1 to 1.1 with incompressible lateral contraction. Strain is 2D Green–Lagrange strain: the final E_XX is 0.105, and E_YY is approximately −0.04545.
- The tracker boundary takes only frames and query points and returns their positions, Boolean visibility, and resource measurements. The oracle returns true positions only at those queries and uses no support grid. For a custom stretch, supply `OracleTracker(UniformExtension(config.final_stretch))`; mismatched oracle configurations are rejected. Injected trackers declare their name and support-grid count. `strainsight.cotracker.CoTracker3Tracker` accepts grayscale `(T,H,W)` or RGB `(T,H,W,3)` arrays in 0–255 and queries as `(N,2)` input-image `(x,y)` coordinates, all placed in the reference frame. Its GPU imports and checkpoint loading happen only when constructed.
- Both strain estimator variants share an unweighted affine least-squares displacement fit in the reference configuration. The secondary uses all finite visible query points within R = 2.5 × baseline spacing. The primary uses k nearest points, with k calibrated separately at each evaluation position as the number of intact lattice-baseline points in that radius. The counts are frozen across placement methods and frames, before visibility filtering. Both stencils therefore coincide on the intact baseline, including for non-affine fields, arbitrary lattice offsets, and mask edges. They can differ after placement changes or track loss.
- An interior square-lattice site has 21 points in its 2.5-spacing disk, including its centre; other evaluation positions can have different counts. This replaces the old area approximation k = round(π × 2.5²) = 20. Direct calls to `k_nearest_strain` default to 21; pass counts from `baseline_neighbour_counts` to follow the experiment protocol on an independent evaluation grid.
- Fewer than six neighbours, deficient rank, or a design condition number above 10⁶ give undefined strain in both variants. `strainsight.strain.MINIMUM_NEIGHBOURS` exposes the common minimum for placement's coverage floor; `CONDITION_LIMIT` exposes the common conditioning threshold. Invisible query points and nonfinite reference or tracked positions never enter either fit.
- Evaluation uses a fixed 4 px grid over the eroded gauge section, independent of placement. Full-field E_XX MAE averages defined positions across scored frames; coverage loss reports the undefined fraction separately. If every estimate is undefined, MAE is `None`, stored as JSON `null`. Lost tracks retain their place in the query budget and are never replaced.
- `ExperimentResult.strain` holds the primary field and `fixed_radius_strain` the secondary field, computed from the same tracks. Records retain `full_field_exx_mae` and `coverage_loss` for the primary and add `fixed_radius_exx_mae` and `fixed_radius_coverage_loss` for the secondary. Configuration records the calibrated k counts in evaluation-grid order, their calibration source, the physical fixed radius in tracker-input pixels, and both shared fit thresholds. A denser placement at a given budget keeps those counts and the physical radius fixed: only k-nearest's support shrinks. Along the budget curve, R scales with each budget's baseline spacing.
- Records contain configuration, the generator version and content SHA-256, rendered-frame SHA-256, speckle stratum, query count, metrics, total pipeline wall time (rendering through evaluation, excluding persistence), tracker wall time, and peak allocated VRAM in bytes (zero for CPU). CoTracker timing synchronizes CUDA and includes input conversion, device transfers, predictor resizing, and output transfer; model loading/download is excluded. Peak VRAM is reset per tracking call and includes the resident model, inputs, and inference allocations, rather than CUDA's reserved cache. The run configuration records checkpoint path, device, support-grid size/count, and visibility settings. Older records without resource fields remain readable (`tracker_runtime_seconds=None`, `peak_vram_bytes=0`). Held-out benchmark runs require the later pre-registration step.

## Frozen synthetic sequences

The committed [manifest](src/strainsight/synthetic_manifest.json) fixes the distribution, baseline spacings, dev/test seeds, and generator content hash. Both splits contain all three deformation cases and both speckle strata. Changed generator sources or manifest content are rejected before an experiment runs.

```python
from strainsight.dataset import frozen_dataset
from strainsight.experiment import ExperimentConfig, run_experiment
from strainsight.results import ResultsStore
from strainsight.tracking import OracleTracker

dataset = frozen_dataset()
case = dataset.dev[0]  # Localized necking; tuning uses dev sequences only.
sequence = case.render()  # 32 grayscale frames, specimen masks, and motion.
truth = sequence.ground_truth(28)  # Dense reference displacement and strain.
result = run_experiment(ExperimentConfig.from_case(case), OracleTracker(case.motion),
                        ResultsStore("outputs/results"))
```

The [generator protocol](docs/synthetic_generator.md) defines the strain convention, motion map, texture, noise/blur distributions, hash inputs, and intentional refreezing procedure. Synthetic frames are generated on demand and are not committed.

## Development

Decision-rule analysis reads an explicitly selected set of completed records:

```python
from strainsight.analysis import analyze_results

report = analyze_results(store, selected_run_ids, seed=42)
print(report.primary.passes, report.primary.high_strain_ci)
```

The typed report includes paired effects, a 10,000-resample sequence-cluster
bootstrap, the 5% full-field non-inferiority check, and separate descriptive
summaries by spacing, tracker, estimator, and speckle stratum. See the
[analysis contract](docs/decision_rule.md) for required metrics, pairing rules,
and minimum inputs. Fixture tests exercise this without running held-out
experiments; the experiment-grid milestone still supplies adaptive runs and
high-strain metrics.

```bash
uv run python -m mypy src/strainsight tests
uv run python -m pytest tests/test_analysis.py
uv run python -m pytest tests/test_strain.py
uv run python -m pytest tests/test_pipeline.py
uv run python -m pytest
```

Pipeline tests inject trackers at the agreed experiment boundary and check the physics, geometry, budget, independent evaluation, loss handling, inverse-warp rejection, and results-store immutability. The old farthest-point sampler and two-point engineering strain estimator remain only as legacy sanity checks in `tests/test_core.py`.

Estimator tests use the public k-nearest and fixed-radius APIs to check affine recovery, baseline agreement for a nonlinear displacement on independent evaluation grids and lattice offsets, shared observability rules, and a Gaussian displacement band with analytic Green–Lagrange truth. The band's error is small but nonzero and decreases as query density increases: k-nearest retains its coarse-baseline calibration, and fixed-radius is checked along the budget curve.

`src/strainsight/` contains the specimen geometry, synthetic generator, placement, tracker interface, strain estimator, experiment runner, results store, and demo. The full protocol and future milestones are in [docs/experiment_spec.md](docs/experiment_spec.md), with vocabulary in [GLOSSARY.md](GLOSSARY.md) and the budget/estimator decision in [ADR 0001](docs/adr/0001-budget-and-primary-estimator.md).

Do not commit DIC Challenge images or derived artifacts before licensing is verified. This experiment generates its own texture and requires no external datasets or model weights.

## License

MIT; see [LICENSE](LICENSE).
