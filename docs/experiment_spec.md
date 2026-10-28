# Experiment spec: measurement-guided query placement vs. lattice baseline

Published as GitHub issue #1; this file and the issue body are kept identical below this header.

## Problem Statement

A researcher measuring deformation of a stretched specimen from video wants to know whether placing tracker query points where measurements say they matter (strain gradients, coverage, tracking reliability) recovers the strain field — especially a narrow necking band — more accurately than a regular lattice spending the same query budget inside the same specimen mask. Today the repo only has a farthest-point sampler and a two-point axial strain check; there is no way to run, score, or demonstrate that comparison honestly. The comparison is easy to rig by accident (mismatched budgets, grading on points a method chose, ground truth leaking into placement, tracker errors masquerading as placement effects), so the protocol must make those failure modes structurally impossible. The project is also a Meta hackathon entry, so it needs a reliable demo, a 3-minute video, real use of Meta tooling, and a public repository that never redistributes unlicensed data.

## Solution

A reproducible experiment pipeline that renders frozen synthetic tensile sequences with exact Green–Lagrange ground truth, places query points with either the lattice baseline or the adaptive method at an equal query budget, tracks them under oracle tracking and under CoTracker, reconstructs the strain field with one shared strain estimator, and scores every method on the same evaluation grid with pre-registered metrics, stratified by speckle size. The same pipeline runs on real data (TelecentricSample6 with a SAM 3.1 specimen mask; Sample14 as a sparse-frame condition) without redistributing it. A results assistant (Muse Spark via the Meta Model API) answers questions about completed runs only through read-only tools, never computing numbers itself. A local demo app over precomputed runs and a 3-minute video present the result — or, if the adaptive method does not win, an honest alternate storyline.

All terms below are as defined in the project glossary; the budget and estimator decisions are recorded in ADR 0001.

## User Stories

### Experiment design and fairness
1. As a researcher, I want the query budget fixed by the lattice baseline's in-mask count at a chosen spacing, so that every budget corresponds to a real, reproducible lattice.
2. As a researcher, I want the adaptive method to spend exactly the same number of distinct query points as the lattice baseline, so that any difference is attributable to placement.
3. As a researcher, I want re-tracked points to count once against the budget, so that the extra coarse pass is reported as compute rather than hidden as points.
4. As a researcher, I want both methods to report strain only from a single measurement pass of all N points tracked together, so that tracker interactions between queries affect both methods equally.
5. As a researcher, I want the support grid identical across methods, excluded from N, and reported, so that tracker context cannot favour either method.
6. As a researcher, I want both methods confined to the same eroded mask and sampling region, so that the comparison never pits masked against unmasked sampling.
7. As a researcher, I want results averaged over several random lattice offsets shared between the baseline and the coarse pass, so that results do not depend on one lattice alignment and comparisons are paired.
8. As a researcher, I want a budget curve over several baseline spacings, so that I can see where adaptive placement helps and where it does not.
9. As a researcher, I want the narrowest necking band to be at least twice the largest baseline spacing, so that the baseline could in principle resolve every band.

### Placement policy
10. As a researcher, I want the coarse pass to be a 1.5×-spacing lattice of the same family as the baseline, so that it samples the whole sampling region before refinement.
11. As a researcher, I want a coverage floor placed from geometry alone before any measurement, so that no region is left without enough neighbours for the fixed-radius estimator.
12. As a researcher, I want the refinement score to rise with coarse strain-gradient magnitude and lack of coverage and to be suppressed — never raised — by tracking uncertainty, so that points are not wasted where they would become lost tracks.
13. As a researcher, I want the refinement score computed from the maximum of a temporally smoothed per-frame score over the whole sequence, so that a late-forming neck is found without chasing single-frame noise.
14. As a researcher, I want refinement points allocated to regions in proportion to score on a finer sub-lattice of the same family, so that the budget is not spent on a single hotspot.
15. As a researcher, I want the uncertainty gate to use the tracker's own default visibility rule with a threshold fixed a priori, so that no gate parameter is tuned.
16. As a researcher, I want the refinement-score weights tuned only on dev sequences and frozen before any test sequence is run, so that test results are not fitted.
17. As a researcher, I want the adaptive policy to see only coarse-pass tracked trajectories, tracker visibility, and quantities reconstructed from them, so that ground truth never influences placement.

### Tracking conditions
18. As a researcher, I want every synthetic comparison run first under oracle tracking, so that placement effects are measured without tracker error.
19. As a researcher, I want the same comparison run separately under CoTracker, so that I can see how tracker error changes the placement effect.
20. As a researcher, I want oracle tracking to give each method true positions only at its own query points, so that the oracle stands in for a perfect tracker without leaking the dense field.
21. As a researcher, I want lost tracks excluded from that frame's estimate, counted against the budget, and never replaced, so that track loss shows up honestly as coverage loss.

### Strain reconstruction
22. As a researcher, I want one unweighted affine least-squares strain estimator shared by both methods, so that reconstruction cannot favour either.
23. As a researcher, I want the k-nearest variant (k tied to R = 2.5 baseline spacings at baseline density) as the pre-registered primary, so that added density can sharpen resolution of narrow bands.
24. As a researcher, I want the fixed-radius variant (R = 2.5 baseline spacings) reported as the conservative secondary, so that a variance-only gain is also visible.
25. As a researcher, I want identical conditioning thresholds and minimum neighbour counts for both methods, with undefined outputs reported as coverage loss, so that failed regions are never silently excluded.
26. As a researcher, I want strain expressed as 2D Green–Lagrange strain at reference-frame material positions, so that results match the ground truth and stay valid at large local strain.

### Synthetic data
27. As a researcher, I want synthetic sequences rendered from a reference-configuration motion map with an explicit inverse mapping per pixel, so that ground truth is exact at tens-of-pixel displacements.
28. As a researcher, I want frames with inversion residual above 0.01 px rejected, so that a wrong-direction warp fails loudly.
29. As a researcher, I want the base speckle rendered supersampled or analytically, so that rendering does not alias the texture the tracker relies on.
30. As a researcher, I want uniform-extension, localized-necking, and smoothly varying (control) deformation cases, so that I can check the adaptive method gains nothing where it should gain nothing.
31. As a researcher, I want band width, position, strength, speckle size, noise, and blur drawn from one frozen distribution with disjoint dev and test seeds, so that the split is honest.
32. As a researcher, I want the generator frozen before any method runs, so that the data cannot be tuned to the method.

### Evaluation and pre-registration
33. As a researcher, I want all errors measured on a fixed evaluation grid over the eroded gauge section, independent of any method's query points, so that no method is graded where it chose to look.
34. As a researcher, I want co-primary metrics of full-field E_XX MAE and high-strain-region E_XX MAE (top 10% by ground truth, per scored frame) under the k-nearest estimator at equal budgets, so that both overall accuracy and accuracy in the band are judged.
35. As a researcher, I want secondary metrics (E_YY and E_XY errors, mean and 95th-percentile displacement error, coverage loss, tracking failures, runtime, VRAM, fixed-radius results), so that trade-offs are visible.
36. As a researcher, I want every metric reported separately in the 1.5–3 px and 3–5 px speckle strata, so that results are not an average over incomparable texture conditions.
37. As a researcher, I want the metrics, strata, estimator, decision rule, and parameters committed as a pre-registration before any test run, so that the analysis cannot drift after seeing results.

### Real data
38. As a researcher, I want real frames cropped to a 4:3 specimen ROI before the tracker, never the full frame, so that the tracker's fixed resize does not distort the strain components.
39. As a researcher, I want the Sample6 specimen ROI fixed at 1365×1024 px with the long side along the loading axis and both specimen edges inside over the whole sequence, so that the mask, erosion and floor are valid.
40. As a researcher, I want the per-dataset downscale factor recorded and reported as a limitation, so that readers know the tracker's effective resolution.
41. As a researcher, I want SAM 3.1 to segment the specimen once on the reference frame of the ROI, with the mask approved visually and logged, so that a bad mask cannot silently break the run.
42. As a researcher, I want a hand-annotated reference mask and the SAM mask's IoU against it, with the comparison re-run on the hand mask, so that mask error is measured, not assumed.
43. As a researcher, I want Sample6 scored by global elongation trend against the machine grip-extension data (sanity only) and by agreement with a dense reference run (consistency, not ground truth), so that real-data claims stay honest.
44. As a researcher, I want Sample14 tracked as a 4-frame sparse-frame condition against its Lagrangian ground truth, so that there is an external check on data we did not make.

### Results assistant
45. As a demo viewer, I want to ask questions about completed runs in plain language, so that I can explore results without reading tables.
46. As a demo viewer, I want every number in an answer to come from a tool output with a cited run, so that the assistant cannot invent measurements.
47. As a researcher, I want comparison tools to return differences and percentages computed in code, and the model to write placeholders that code fills, so that the model never does arithmetic.
48. As a researcher, I want the assistant unable to launch runs or change parameters, so that it cannot become a third placement method.
49. As a researcher, I want the assistant behind a provider-neutral interface with Muse Spark via the Meta Model API as the configured provider, so that the model can be swapped without touching tools.
50. As a researcher, I want a scripted grounding check where every number in each answer must match a tool output exactly, so that the assistant's trustworthiness is demonstrated.

### Demo, video, and repository
51. As a judge, I want a local demo app over precomputed runs showing baseline vs. adaptive vs. ground-truth heatmaps, the budget curve, Sample6 with mask and points, and the assistant, so that the demo is reliable.
52. As a judge, I want a 3-minute video opening on the neck-forming strain heatmap, so that the problem is visible in the first seconds.
53. As a presenter, I want the assistant answering live with a clearly labelled prerecorded fallback, so that a network failure does not sink the demo.
54. As a presenter, I want an alternate, scientifically honest storyline ready if the adaptive method does not outperform the baseline, so that the submission does not overclaim.
55. As a maintainer, I want no DIC Challenge images or derived artifacts (masks, tracks, hand annotation) in the public repo until licensing is verified, so that we do not redistribute unlicensed data.
56. As a maintainer, I want the full Git history audited for dataset files and secrets before submission, so that the public repo is clean.

## Implementation Decisions

### Fixed parameters (synthetic)
| Item | Value |
|---|---|
| Tracker input frame | 512×384 px (CoTracker3 internal resolution, confirmed in code) |
| Specimen | Horizontal dog-bone, loading along x; gauge width W = 64 px, gauge length 256 px, fillet radius 64 px, grip width 102 px, grips 0.25W in view, reference length 380 px |
| Sampling region / scoring | Gauge + fillets get points (grips none); only the eroded gauge section is scored |
| Eroded mask | Specimen mask eroded by a fixed 4 tracker-input px on every dataset; N counted in it |
| Budget curve | s ∈ {4.0, 5.3, 6.4, 8.0} px; N ≈ 1290 / 745 / 520 / 315 after erosion (to be recomputed by the implementation) |
| Coarse pass | 1.5× baseline spacing, shared lattice offset (≈ 44% of N) |
| Coverage floor | Geometry-only; simulated cost ≤ 8.5% of N along the curve |
| Estimator | Unweighted affine LS; primary k-nearest with k tied to R = 2.5s at baseline density (≈ 20); secondary fixed-radius R = 2.5s, min 6 neighbours + conditioning check |
| Deformation | Uniform gauge strain 8–12%; Gaussian band FWHM 16–40 px, peak extra strain 20–60%; incompressible lateral contraction; grip-end displacement ±12 to ±28 px; plus uniform-extension and smooth controls |
| Speckle | 1.5–5 px dots in tracker-input px; strata 1.5–3 and 3–5 px |
| Frames | 32 per sequence, scored every 4th frame |
| Split | One frozen generator distribution; ~8 dev and ~24 test sequences with disjoint seeds, controls in both |
| Repeats | 3 lattice offsets per sequence |

### Modules
- **Specimen geometry and masks**: builds the specimen mask, eroded mask, sampling region and gauge section for synthetic specimens; accepts an externally supplied mask for real data. Erosion is always in tracker-input px.
- **Synthetic sequence generator**: renders frames from a reference-configuration motion map with per-pixel inverse mapping and residual rejection; supersampled/analytic speckle; emits frames, specimen mask, and dense Green–Lagrange and displacement ground truth. Frozen (versioned and hashed) before any method runs.
- **Placement**: produces query points for the lattice baseline, the coarse pass, the coverage floor, and refinement points from a refinement score. Its only inputs from measurement are coarse-pass tracks, visibility, and reconstructed quantities; it has no access to ground truth.
- **Tracker interface**: one interface — frames, query points → tracks and visibility — with two implementations: oracle tracking (true positions at the given query points only) and a CoTracker3 offline adapter (uses the predictor's own visibility rule and built-in support grid; records runtime and VRAM).
- **Strain estimator**: k-nearest and fixed-radius variants of the same affine fit, shared conditioning rules, returns undefined where the fit fails.
- **Evaluation**: evaluation grid, co-primary and secondary metrics, high-strain region per scored frame, coverage loss, speckle strata, paired comparison across offsets and seeds.
- **Experiment runner and results store**: runs (sequence, method, tracking condition, spacing, offset) cells, writes immutable run records with configuration, metrics, runtime, VRAM, and query counts. The store is the only thing the demo and assistant read.
- **Real-data pipeline**: 4:3 specimen ROI crop (Sample6: 1365×1024, long side along loading axis, both edges inside across the sequence), downscale factor recorded, mapping tracks back to full-frame px; segmenter interface with a SAM 3.1 local adapter (fallback: Meta Model API) and a hand-mask path; Sample14 loader as a sparse-frame condition. Data lives outside the repo.
- **Results assistant**: read-only tools over the results store (list runs, metrics, strain at a point or region, method comparison returning differences and percentages); provider-neutral model client with a Muse Spark adapter; answer rendering that fills model-written placeholders from tool outputs; scripted grounding check.
- **Demo app**: single local web app over the results store; heatmaps, budget curve, Sample6 view, assistant panel with live mode and labelled prerecorded fallback.

### Protocol decisions
- Adaptive runs: coarse pass (placement only) → refinement points → measurement pass over all N points; the coarse pass is reported as compute.
- Lost track = CoTracker offline default (visibility > 0.9, confidence not used). Under oracle tracking visibility is 1 everywhere, so the gate is inactive; this is stated in results.
- Refinement score = (weighted strain-gradient term + coverage term) × visibility gate, using the max over frames of a temporally smoothed score; weights tuned once on dev sequences under CoTracker, frozen, and reused unchanged under oracle tracking.
- Ground truth is read only by the evaluation module (and by the oracle tracker, at query points only).
- Pre-registration document committed to the repo before the first test run; its commit is the freeze.
- Python 3.12 single environment (SAM 3.1 requires ≥ 3.12; CUDA ≥ 12.6, cu128 builds for the RTX 5070 Ti); separate SAM environment only as fallback.
- Estimated tracker workload ≈ 1,150 runs, roughly 20 min – 1.6 h GPU time at an assumed 1–5 s per run; to be replaced by a measured per-run time in milestone 1.

### Milestones
1. **Environment and facts**: Python 3.12 env; CoTracker install and one timed run on the RTX 5070 Ti; confirm 384×512 resize behaviour end to end; SAM 3.1 local install check (HF access pending); record per-run time and update the GPU estimate.
2. **Synthetic core**: geometry and masks, frozen generator, oracle tracker, strain estimators, evaluation; oracle-tracking results on dev sequences.
3. **Placement and tuning**: lattice baseline, coarse pass, coverage floor, refinement score and allocation; tune refinement-score weights once on dev sequences under CoTracker and freeze them.
4. **Pre-registration and test runs**: commit the pre-registration; run test sequences under oracle tracking, then CoTracker; produce stratified tables and the budget curve.
5. **Real data**: Sample6 width check, ROI, SAM 3.1 mask approval, hand mask and IoU, baseline/adaptive and dense reference runs, grip-extension sanity check; Sample14 sparse-frame runs.
6. **Results assistant**: tools, placeholder filling, Muse Spark adapter, grounding check.
7. **Demo and video**: demo app over precomputed runs; 3-minute video opening on the neck-forming heatmap, live assistant with labelled fallback; alternate storyline prepared; README updated (research question now three-tier, not DIC Challenge 2.0).
8. **Repository hygiene**: re-run the Git history audit for dataset files and secrets; confirm no DIC-derived artifacts; credit datasets in rendered figures.

## Testing Decisions

- Good tests exercise external behaviour at the highest seam: given a sequence, a method, a tracking condition and a spacing, does the run record satisfy the protocol and get the physics right? They do not assert on internal helpers.
- **Seam 1 — experiment pipeline with an injected tracker.** Tiny synthetic sequences with oracle tracking, asserting: both methods use exactly N distinct query points; all points lie in the sampling region; re-tracked points count once; strain from a uniform-extension sequence matches the analytic Green–Lagrange value; an affine field is recovered exactly by both estimator variants; undefined estimates appear as coverage loss; the placement module cannot reach ground truth (it is given none); inversion residual rejection triggers on a deliberately wrong warp.
- **Seam 2 — results-assistant tool layer.** A scripted fake model client against a fixture results store, asserting tools are read-only, comparison tools return code-computed differences and percentages, placeholders are filled only from tool outputs, and the grounding check fails when an answer contains an unsourced number.
- CoTracker, SAM 3.1 and Muse Spark adapters are behind interfaces and covered only by opt-in smoke checks (GPU/network), not the default test suite.
- Prior art: the existing pytest unit tests of the masked sampler and two-point strain (inside-mask assertion, analytic strain value, empty-mask error) — the same style, lifted to the pipeline seam. The current farthest-point sampler and two-point strain become sanity checks or are retired.

## Out of Scope

- Online (causal) allocation and multiple refinement rounds (possible later ablations).
- Stereo or 3D data, including the Stereo-DIC 2.0 "Tensile Experience" set (no access request for the hackathon).
- Tiled ROIs on real data.
- The assistant launching runs, changing parameters, seeing images, or doing arithmetic.
- Live tracking during the demo.
- Evaluating the assistant beyond the grounding check.
- Redistributing any DIC Challenge data or derivatives.

## Further Notes

### Unverified (must be confirmed before relying on them)
- Official hackathon rules, required deliverables, and judging weights (reported: Impact 30, Meta tooling 25, Demo 25, Technical depth 20).
- Muse Spark's tool-calling format on the Meta Model API, and whether API uploads are retained or used for training.
- SAM 3.1 weights access (gated on Hugging Face, SAM License), compatibility with the RTX 5070 Ti, and image prompt types beyond text and boxes.
- CoTracker per-run time and VRAM on the target GPU.
- Licensing of DIC Challenge datasets (no licence found; downloads are open).

### Known facts behind the decisions
- DIC Challenge 2.0 sets are single image pairs with sub-pixel smooth sinusoidal fields and no localization, so they are not used as the benchmark.
- Sample6: horizontal specimen, ~858–890 px wide (~70 px/mm), ~35–40 px rigid vertical drift; extension column is machine grip extension, not an extensometer; speckle ≈ 3.4 px (blob) / 6.9 px (autocorrelation) at full resolution, ≈ 1.3–2.6 px in tracker input at the 2.67× downscale.

### Alternate storyline (if the adaptive method does not outperform)
Report the null result on the pre-registered metrics, then show what the pipeline does establish: the oracle-vs-CoTracker gap separating placement-limited from tracker-limited regimes; the budget at which the lattice baseline saturates; where (by speckle stratum and band width) placement matters or does not; and the honest measurement stack — mask-constrained, coverage-reporting, grounded assistant — as the contribution. The video still opens on the neck-forming heatmap.

### Decision rule (pre-registered)
The adaptive method **outperforms** the lattice baseline if, at the nominal budget (s = 6.4 px), under CoTracker, with the k-nearest strain estimator, both hold:
- (a) **High-strain-region E_XX MAE is lower**, with the paired 95% bootstrap confidence interval of the difference (adaptive − baseline) excluding zero; pairs are test sequence × lattice offset.
- (b) **Full-field E_XX MAE is not worse by more than 5% relative** (non-inferiority margin).

The budget curve, the fixed-radius estimator, and the oracle-tracking condition are reported descriptively, not as pass/fail.

### Refinement-score weight tuning
Weights are tuned once on dev sequences under CoTracker, frozen before any test run, and the same weights are used under oracle tracking.
