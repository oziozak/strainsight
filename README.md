# StrainSight

Measuring deformation of a stretched specimen from video, and testing whether measurement-guided query placement beats a lattice baseline at the same query budget.

The first complete experiment runs a synthetic uniform-extension sequence through lattice placement, oracle tracking, a k-nearest strain estimator, full-field evaluation, and an immutable results store. It runs on CPU with NumPy. CoTracker3 and SAM 3.1 have been checked in the GPU environment; their pipeline adapters, adaptive placement, and real-data evaluation are future work.

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

Records live in `outputs/results/runs.sqlite3` by default. Each run gets a distinct ID; inserting another record with that ID raises `FileExistsError`. SQLite transactions make writes atomic. Read a completed record through the store:

```python
from strainsight.results import ResultsStore

record = ResultsStore("outputs/results").read("<run ID printed by the demo>")
print(record.configuration, record.metrics, record.query_count, record.runtime_seconds)
```

## What the tracer bullet measures

- The reference specimen is a horizontal dog-bone with a 256×64 px gauge section, circular 64 px fillets, 102 px grips, and a 380 px visible reference length. These dimensions give 16.492 px of each grip in view (approximately 0.25 gauge widths).
- A 4 px Euclidean disk erodes the reference specimen mask. Only the gauge section and fillets in that eroded mask receive query points; grips receive none. Masks classify pixel centres, and subpixel membership uses the nearest pixel centre.
- The lattice baseline's count defines the query budget. At zero offset, spacings 4.0, 5.3, 6.4, and 8.0 px give 1357, 759, 523, and 317 query points, respectively. Counts vary with lattice offset.
- A seeded analytic Gaussian speckle texture is evaluated through the inverse reference-configuration motion map at each output pixel. Nonfinite inversions or forward–inverse residuals above 0.01 px reject the sequence before tracking.
- Uniform extension ramps from stretch 1 to 1.1 with incompressible lateral contraction. Strain is 2D Green–Lagrange strain: the final E_XX is 0.105, and E_YY is approximately −0.04545.
- The tracker boundary takes only frames and query points and returns their positions and Boolean visibility. The oracle returns true positions only at those queries and uses no support grid. For a custom stretch, supply `OracleTracker(UniformExtension(config.final_stretch))`; mismatched oracle configurations are rejected. Injected trackers declare their name and support-grid count.
- Both strain estimator variants share an unweighted affine least-squares displacement fit in the reference configuration. The secondary uses all finite visible query points within R = 2.5 × baseline spacing. The primary uses k nearest points, with k calibrated separately at each evaluation position as the number of intact lattice-baseline points in that radius. The counts are frozen across placement methods and frames, before visibility filtering. Both stencils therefore coincide on the intact baseline, including for non-affine fields, arbitrary lattice offsets, and mask edges. They can differ after placement changes or track loss.
- An interior square-lattice site has 21 points in its 2.5-spacing disk, including its centre; other evaluation positions can have different counts. This replaces the old area approximation k = round(π × 2.5²) = 20. Direct calls to `k_nearest_strain` default to 21; pass counts from `baseline_neighbour_counts` to follow the experiment protocol on an independent evaluation grid.
- Fewer than six neighbours, deficient rank, or a design condition number above 10⁶ give undefined strain in both variants. `strainsight.strain.MINIMUM_NEIGHBOURS` exposes the common minimum for placement's coverage floor; `CONDITION_LIMIT` exposes the common conditioning threshold. Invisible query points and nonfinite reference or tracked positions never enter either fit.
- Evaluation uses a fixed 4 px grid over the eroded gauge section, independent of placement. Full-field E_XX MAE averages defined positions across scored frames; coverage loss reports the undefined fraction separately. If every estimate is undefined, MAE is `None`, stored as JSON `null`. Lost tracks retain their place in the query budget and are never replaced.
- `ExperimentResult.strain` holds the primary field and `fixed_radius_strain` the secondary field, computed from the same tracks. Records retain `full_field_exx_mae` and `coverage_loss` for the primary and add `fixed_radius_exx_mae` and `fixed_radius_coverage_loss` for the secondary. Configuration records the calibrated k counts in evaluation-grid order, their calibration source, the physical fixed radius in tracker-input pixels, and both shared fit thresholds. A denser placement at a given budget keeps those counts and the physical radius fixed: only k-nearest's support shrinks. Along the budget curve, R scales with each budget's baseline spacing.
- Records contain configuration, the generator version and rendered-frame SHA-256, speckle stratum, query count, metrics, and total CPU pipeline runtime (rendering through evaluation, excluding persistence). This tracer bullet uses synthetic development/sanity sequences; held-out benchmark runs and their pre-registration are later work.

## Development

```bash
uv run python -m mypy src/strainsight tests
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
