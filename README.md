# StrainSight

**Mask-guided adaptive point tracking for deformation measurement** — Meta Hackathon prototype.

StrainSight segments a deforming specimen, samples tracking queries **inside its mask**, tracks them across a video, estimates displacement and strain, and allocates additional queries to informative regions. The goal is to evaluate whether adaptive query allocation improves strain estimation at a fixed tracking budget.

## Architecture

```text
Video frames → specimen mask → mask-constrained baseline points
            → point trajectories → displacement / local strain
            → refinement score → additional points → strain visualization
```

Initially the user supplies a binary specimen mask for frame 0; SAM integration is a later module. Tracking is initially model-agnostic; CoTracker integration is planned. Neither segmentation nor learned tracking is falsely presented as implemented here.

## First runnable baseline

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate           # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e '.[dev]'
pytest
python -m strainsight.demo
```

The demo creates a synthetic rectangular specimen, samples masked points, applies known uniaxial extension, and estimates axial engineering strain. This is **not** a benchmark on real DIC images.

## Next milestones

1. **Dataset loader:** select a small DIC Challenge sequence with accessible reference displacement / strain data and document evaluation region/units.
2. **Segmentation:** manual/reference mask baseline first; experiment with Meta SAM when feasible.
3. **Tracking:** integrate CoTracker / equivalent, monitor visibility and mask drift.
4. **Evaluation:** compare mask-constrained uniform vs mask-constrained adaptive under equal tracking-query budgets.
5. **Policy:** rank cells by estimated strain gradients, spatial coverage, and trajectory confidence, and retrack additional points in informative regions.
6. **Demo:** displacement and strain heatmaps with a Meta-model assistant querying computed results (no invented physics measurements).

## Evaluation guardrails

- Compare **both policies inside the same specimen mask**; do not compare adaptive masked against whole-scene uniform.
- Keep query count, initialization frames, and model settings matched.
- Hold out independent ground truth; do not grade against a reference used for sampling decisions.
- Report strain/displacement errors and wall time / GPU memory.
- Keep point correspondences in the material reference frame; a changing mask must not silently replace lost points.

## Hardware

Main vision development: RTX 5070 Ti PC (CUDA). Mac M2 Pro: frontend, documentation and light tests.

## Repository layout

```text
src/strainsight/         Core algorithms and demo
  sampling.py            Mask-constrained uniform/grid samples
  strain.py              Engineering axial strain estimator
  demo.py                Tiny synthetic validation
 tests/                  Unit tests
 docs/idea.md            Hackathon project scope
```

## Status

- [x] Project scaffold
- [x] Mask-constrained fixed-budget sampler
- [x] Synthetic axial engineering strain demonstration
- [ ] Real dataset ingestion
- [ ] Segmentation model
- [ ] Pretrained point-tracker integration
- [ ] Adaptive query policy
- [ ] End-to-end Meta AI assistant

## License

MIT; see `LICENSE`.
