# StrainSight

Measuring deformation of a stretched specimen from video, and testing whether measurement-guided query placement beats a lattice baseline at the same query budget.

The experiment runs synthetic uniform-extension, localized-necking, and smooth control sequences through lattice placement, oracle tracking, a k-nearest strain estimator, full-field evaluation, and an immutable results store. A frozen generator defines 8 dev and 24 test sequences from one distribution. It runs on CPU with NumPy. CoTracker3 and SAM 3.1 have been checked in the GPU environment; their pipeline adapters, adaptive placement, and real-data evaluation are future work.

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
- A seeded analytic Gaussian speckle texture is evaluated through the inverse reference-configuration motion map with 2×2 pixel supersampling. Nonfinite inversions or forward–inverse residuals above 0.01 px reject the sequence before tracking.
- Uniform extension ramps from stretch 1 to 1.1 with incompressible lateral contraction. Strain is 2D Green–Lagrange strain: the final E_XX is 0.105, and E_YY is approximately −0.04545.
- The tracker boundary takes only frames and query points and returns their positions and Boolean visibility. The oracle returns true positions only at those queries and uses no support grid. For a custom stretch, supply `OracleTracker(UniformExtension(config.final_stretch))`; mismatched oracle configurations are rejected. Injected trackers declare their name and support-grid count.
- The unweighted affine least-squares estimator uses k = round(π × 2.5²) = 20 nearest visible query points in the reference configuration. Fewer than six neighbours, deficient rank, or a design condition number above 10⁶ give undefined strain.
- Evaluation uses a fixed 4 px grid over the eroded gauge section, independent of placement. Full-field E_XX MAE averages defined positions across scored frames; coverage loss reports the undefined fraction separately. If every estimate is undefined, MAE is `None`, stored as JSON `null`. Lost tracks retain their place in the query budget and are never replaced.
- Records contain configuration, the generator version and content SHA-256, rendered-frame SHA-256, speckle stratum, query count, metrics, and total CPU pipeline runtime (rendering through evaluation, excluding persistence). Held-out benchmark runs require the later pre-registration step.

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
