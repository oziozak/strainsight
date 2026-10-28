# Frozen synthetic generator

Issue #6 is implemented by `strainsight.dataset` and `strainsight.synthetic`.
The generator version is `tensile-v2`; the authoritative content SHA-256 is
`generator_sha256` in `src/strainsight/synthetic_manifest.json`.

## Frozen distribution and split

The manifest fixes 8 dev and 24 test seeds. Dev has 4 localized-necking,
3 uniform-extension, and 1 smooth control sequence; test has 10 localized-necking,
6 uniform-extension, and 8 smooth control sequences. Both splits include both
speckle strata. The manifest and seeds are frozen before comparison runs;
test sequences must not be used for weight tuning. Benchmark test runs remain
subject to the pre-registration step in the experiment spec.

`sample_case(seed)` uses PCG64 with the seed alone, with no split-dependent
parameters. Each proposal draws a case with probabilities 0.25 uniform,
0.50 necking, and 0.25 smooth, followed by independent uniform draws:

| Parameter | Range | Units/convention |
| --- | --- | --- |
| Uniform gauge strain | 0.08–0.12 | Centreline Green–Lagrange E_XX at final load |
| Band FWHM | 16–40 | Reference tracker-input px |
| Band centre | 176–336 | Reference x coordinate, inside the gauge section |
| Peak extra strain | 0.20–0.60 | Added centreline E_XX for necking |
| Speckle size | 1.5–5 | Reference Gaussian FWHM in tracker-input px |
| Sensor noise sigma | 0–4 | 8-bit grey levels |
| Optical blur sigma | 0–0.8 | Output tracker-input px |

All proposals, including controls, draw all parameters in the same order.
Uniform controls ignore the band parameters. Smooth controls use the band centre
as a phase and 0.05 times the drawn extra strain as a broad modulation amplitude;
their band FWHM parameter is unused. A proposal is accepted only when the final
grip-end displacements lie within ±12 to ±28 px; otherwise the entire proposal
is redrawn from the same stream. This is one joint rejection distribution for
both splits, without clipping the strain or band-width ranges.

The baseline spacings are 4.0, 5.3, 6.4, and 8.0 px. The narrowest 16 px band
is twice the largest spacing. Each sequence has 32 frames with normalized times
0 through 1; frames 4, 8, 12, 16, 20, 24, and 28 are scored. Frame 0 is the
reference frame.

## Motion and ground truth

Coordinates and specimen geometry follow the experiment spec. At material
position X on the centreline, final loading-axis strain is:

- Uniform: E(X) = e.
- Necking: E(X) = e + a exp(−4 log(2) ((X − c) / w)²).
- Smooth: E(X) = e + 0.05 a cos(2π (X − c) / 512).

At time t, the loading stretch is λ(X,t) = sqrt(1 + 2tE(X)). The axial map
integrates λ with Simpson quadrature on 1/64 px cells and cubic Hermite
interpolation. Its translation gives equal and opposite displacements at the
reference grip ends X = 66 and 446. The grips are part of the prescribed smooth
field; this generator does not model rigid grip boundary conditions.

The lateral map is y = 192 + (Y − 192) λ(X,t)^(−1/2), with the same stretch in
the unrendered z direction. This preserves 3D volume. The spatial gradient
includes the shear induced by the X-dependent lateral contraction; off the
centreline E_XX can therefore exceed the prescribed centreline value. The
ground truth is the full 2D Green–Lagrange tensor (FᵀF − I)/2, including that
shear, in the reference configuration.

The inverse starts with an axial lookup and a Newton correction against the
same forward map, then undoes the lateral stretch. Rendering checks every
subpixel's forward–inverse residual and rejects nonfinite results or a maximum
residual above 0.01 px before tracking.

`SyntheticSequence.ground_truth(frame)` returns dense `(384, 512, 2)`
displacements and `(384, 512, 2, 2)` strain tensors at reference pixel centres.
Positions outside the specimen mask have NaN truth. The oracle accesses the
motion only at its requested query points; evaluation uses its independent
evaluation grid.

## Rendering and provenance

The continuous material texture consists of Gaussian dots on a jittered 6 px
lattice. Dots are evaluated analytically at inverse-mapped positions with
2×2 output-pixel supersampling, including specimen edges. A separable Gaussian
blur is applied before independent, per-frame Gaussian sensor noise. Values
are rounded and clipped to uint8. Texture and sensor noise use separate child
RNG streams, so changing noise or blur does not change the underlying dots.

The content hash covers the exact bytes of `dataset.py`, `specimen.py`, and
`synthetic.py`, with names and NUL separators, followed by canonical sorted JSON
of the manifest excluding its own `generator_sha256` field. The manifest
therefore freezes the distribution, seed split, and spacing schedule as well as
the code. `generator_identity()` checks that hash and version before any
experiment or frozen-case render. The JSON manifest ships in installed wheels.

Every experiment record includes the generator version/content hash, NumPy
version, rendered-frame SHA-256, full case parameters, actual speckle size and
stratum, scored-frame schedule, inversion residual, and split membership.
Manually constructed or modified cases are labelled `custom` unless they
exactly match a frozen case. The existing uniform-extension demo is also
labelled `custom` and records the same generator identity.

An intentional generator change requires updating `GENERATOR_VERSION` and the
manifest version, recomputing the manifest's hash with the maintenance helper
`strainsight.dataset._generator_sha256(manifest)`, rerunning the checks, and
committing the new freeze before comparison runs. Refreezing invalidates earlier
benchmark comparability. Reproduce runs with the committed `uv.lock`; the frame
hash identifies the actual rendered content separately from the source freeze.
