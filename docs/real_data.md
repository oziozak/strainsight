# Sample6 input and evaluation contract

The pipeline runs external TelecentricSample6 frames through the shared lattice
baseline, adaptive method, and strain estimators. The automated tests use generated
arrays only. A real run requires the human annotation and a CUDA CoTracker3 setup;
there are no reported real-data results in the repository.

## External inputs and the human step

Keep images, machine data, annotations, and manifests in an external directory.
Install image decoding with `uv sync --extra dev --extra real`; add `--extra gpu`
for CoTracker3. The manifest explicitly orders frames and aligns one machine
grip-extension measurement to each frame. Paths resolve relative to the manifest.
The loader preserves this order, rejects repeated paths and mismatched lengths,
and does no sorting, timestamp interpolation, or machine-column guessing.

```json
{
  "frames": ["images/frame000.tif", "images/frame004.tif", "images/frame008.tif"],
  "grip_extension": [0.0, 0.1, 0.2],
  "grip_extension_units": "mm"
}
```

These are illustrative values, not measurements. Frames must be 8-bit grayscale
or RGB, identically sized, with at least two frames. Any source intensity
conversion must be chosen and documented separately. The first listed frame is
the reference frame. Select `--score-every 1` for a short sequence; the default
scores every fourth supplied frame after the reference frame.

Choose a **1365×1024** specimen ROI in full-frame pixels. Sample6 loads along
horizontal x, the long ROI side. Keep the same origin for all frames and visually
check that both lateral specimen edges remain inside it across the entire
sequence, including drift. The `--edges-verified` flag logs this human check;
one reference mask cannot certify edge containment in later frames.

Export the reference ROI outside the repository for annotation, for example:

```python
from PIL import Image

x, y = 300, 200  # Illustrative; choose from the real images.
with Image.open("/external/images/frame000.tif") as image:
    image.crop((x, y, x + 1365, y + 1024)).save("/external/reference-roi.png")
```

Annotate three binary images of this ROI, each **1024 rows × 1365 columns**:

- `specimen_mask`: the specimen in the reference frame.
- `sampling_region`: gauge section and fillets, excluding grips, inside the
  specimen mask.
- `gauge_section`: the central gauge section used for scoring, inside the sampling
  region.

Convert these human annotations to an external NPZ. White/nonzero means inside;
store Boolean arrays rather than 0–255 integers:

```python
import numpy as np
from PIL import Image

masks = {}
for name in ("specimen_mask", "sampling_region", "gauge_section"):
    with Image.open(f"/external/{name}.png") as image:
        masks[name] = np.asarray(image.convert("L")) > 0
np.savez_compressed("/external/hand-mask.npz", **masks)
```

`Segmenter.segment(reference_frame)` takes only the reference ROI and returns
`ReferenceMasks`. `HandMaskSegmenter` implements this interface by loading the
NPZ. The pipeline checks Boolean masks, matching dimensions, nonempty regions,
nested regions, and reference lateral-edge containment. It segments once and
never redraws a mask on a deformed frame. A SAM adapter and SAM/hand-mask IoU are
separate work.

## Run and inspect

```bash
uv run python -m strainsight.real_demo \
  --manifest /external/sequence.json --mask /external/hand-mask.npz \
  --roi 300 200 --edges-verified --score-every 1 \
  --spacing 8 --dense-spacing 2 --results-dir outputs/sample6
```

The origin is illustrative. `--checkpoint` can select an existing checkpoint.
The command prints each run ID, query count, metrics, track archive path, and the
limitations. Results use the existing immutable `ResultsStore`. Archives under
the results directory's `tracks/` contain query points and tracked positions in
both tracker-input and full-frame coordinates, plus Boolean visibility. Only
measurement tracks are exported for the adaptive method.

The same public interfaces support injected trackers without the GPU stack:

```python
from strainsight.real_data import HandMaskSegmenter, Sample6ROI, load_sample6_manifest, prepare_sample6
from strainsight.real_experiment import RealDataConfig, run_sample6
from strainsight.results import ResultsStore

source = load_sample6_manifest("/external/sequence.json")
data = prepare_sample6(source.frames, Sample6ROI(300, 200),
                      HandMaskSegmenter("/external/hand-mask.npz"), edges_verified=True)
# Supply a Tracker implementation as tracker.
result = run_sample6(data, tracker, ResultsStore("outputs/sample6"), source.grip_extension,
                     RealDataConfig(score_every=1), frame_ids=source.frame_ids,
                     grip_extension_units=source.grip_extension_units)
```

## Coordinates and resolution

Every tracker call, including the coarse pass and dense reference, receives only
the 512×384 bilinear resize of the specimen ROI. Masks use nearest-pixel resizing.
The specimen mask is then eroded by a Euclidean disk of **4 tracker-input px**;
query placement uses its intersection with the sampling region, and scoring uses
its intersection with the gauge section. No grips receive queries.

The pixel-centre transform is reversible for arbitrary subpixel track positions:

```text
downscale_xy = [1365 / 512, 1024 / 384]
full_xy = (tracker_xy + 0.5) * downscale_xy - 0.5 + [roi_x, roi_y]
```

Records retain both factors, the source and input sizes, the ROI, mask and input
frame SHA-256 hashes, supplied frame IDs, and the containment check. Strain sample
positions and estimator radii remain in tracker-input pixels. The approximately
2.67× downscale may leave speckles or narrow bands unresolved. The specified
1365×1024 ROI approximates 4:3, so its x/y scale factors differ slightly. These
limitations are saved and printed on every command invocation.

## Consistency and sanity metrics

The baseline lattice defines the query budget. The adaptive method spends exactly
that budget using the shared geometry-only coverage floor, coarse measurements,
and one measurement pass. Only measurement tracks produce its reported strain.
Coarse and measurement timings and VRAM are saved separately; tracker runtime
totals both, and peak VRAM is their maximum. Per-run pipeline time covers tracking
through scoring, excluding loading, preprocessing, model construction, and saving.

The dense reference uses an independent lattice on the same sampling region.
Its spacing is at most half the baseline spacing, with at least four times as
many queries; default spacings 8 and 2 usually give about sixteen times as many.
It receives the same images, support grid, visibility rule, evaluation grid, and
strain estimators. No dense measurements enter adaptive placement.

The evaluation grid is fixed independently of method queries. All runs use k
counts calibrated on the intact baseline within 2.5 baseline spacings, including
the dense reference; its larger density can therefore shrink primary support.
The secondary fixed-radius estimator retains that same physical radius for all
three runs. Both reconstruct Green–Lagrange strain with the existing shared
observability rules.

For baseline and adaptive records:

- `dense_reference_loading_axis_mae` is the mean absolute difference in the x
  loading-axis component against the saved dense-reference run over jointly
  defined scored positions and frames.
- `coverage_loss` reports each method's undefined fraction independently.
- `dense_comparison_coverage_loss` reports the fraction without a jointly defined
  method/reference pair. An entirely missing comparison has MAE `None`, not zero.
- The `fixed_radius_` metrics repeat these definitions for the secondary estimator.

The dense run has its own coverage and tracking-failure metrics, without scoring
it against itself. Records identify the dense run explicitly and label the
comparison **consistency only, not ground truth**. Shared tracker errors can
produce misleading agreement; these records use no synthetic accuracy metric.

`global_elongation` is the engineering extension slope of a global affine
loading-axis displacement fit over finite visible gauge queries for each frame.
The fit removes rigid translation and shear terms. Fewer than six usable points,
bad conditioning, or rank deficiency yields a missing value. The full per-frame
series, aligned machine extension relative to its reference value, and declared
machine units are retained. `grip_extension_trend_correlation` is Pearson
correlation of the two series where defined, requiring at least three paired
frames and nonconstant signals. Otherwise it is `None`. There is no pixel-to-mm
conversion or amplitude accuracy claim: machine grip extension includes grip and
system compliance and is **a global trend sanity check only**.

## Git guard

Never commit DIC Challenge images or derivatives. The existing image, array,
archive, database, data-directory, and output-directory ignores are extended with
Sample6/Sample14/Telecentric/DIC-Challenge names, `masks/`, `tracks/`,
`annotations/`, text track exports, pickles, and additional image formats. Store
generic manifests and other text derivatives outside the repo too; Git cannot
recognize data provenance from arbitrary filenames. Do not force-add ignored
artifacts. Documentation and generated test fixture code are the reviewable
artifacts committed for this pipeline.
