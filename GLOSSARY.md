# StrainSight

Measuring deformation of a stretched specimen from video by tracking material points, and testing whether measurement-guided placement of those points beats a fixed lattice at the same budget.

## Points and budget

**Query point**:
A material point, chosen in the reference frame inside the specimen mask, that the tracker is asked to follow. A query point counts once, however many times it is tracked.
_Avoid_: Track, sample, seed

**Query budget**:
The number of distinct query points a method may use in one run, fixed per spacing as the lattice-in-mask count of the baseline.
_Avoid_: Point budget, N points

**Support grid**:
Auxiliary points the tracker uses for context, identical across methods and not counted against the query budget.
_Avoid_: Context points, background points

**Budget curve**:
Results across a series of baseline spacings, each spacing fixing its own query budget; the specimen width in spacings changes along it.

**Lattice offset**:
A random sub-spacing shift of a lattice relative to the image, varied across repeats so results do not depend on one alignment.

## Methods

**Lattice baseline**:
The comparison method: a regular lattice at a chosen spacing, clipped to the specimen mask, whose in-mask count defines the query budget.
_Avoid_: Uniform grid, fixed grid, uniform sampling

**Adaptive method**:
The method under test: a coarse pass whose measurements guide where the rest of the query budget is placed, followed by a measurement pass over all its query points.
_Avoid_: Adaptive policy, smart sampling

**Coarse pass**:
The adaptive method's first tracking run, on a coarser lattice of the same family as the lattice baseline; it only guides placement and contributes no reported strain.
_Avoid_: Pass 1 results, preliminary measurement

**Refinement points**:
The adaptive method's query points added once, after the coarse pass, chosen from its measurements and the coverage floor.
_Avoid_: Refinement round, second round, refinement pass

**Measurement pass**:
The single tracking run of a method's full query budget whose tracks alone produce its reported strain; for the adaptive method it re-tracks coarse and refinement points together.
_Avoid_: Final pass, second pass

## Specimen

**Specimen mask**:
The binary region occupied by the specimen in the reference frame; query points may only be placed inside it, and it is never redrawn in later frames.
_Avoid_: Segmentation, foreground, ROI

**Eroded mask**:
The specimen mask shrunk inward by a fixed margin measured in tracker-input pixels, the same on every dataset, so that no query point or scored position sits on a specimen edge.
_Avoid_: Inner mask, safe mask

**Sampling region**:
The part of the eroded mask that may receive query points: the gauge section and fillets, never the grips.

**Gauge section**:
The uniform-width central part of the specimen; only its part inside the eroded mask is scored.
_Avoid_: Gauge length (that is a distance), test section

**Region**:
The cell of the specimen belonging to one coarse-pass query point; the unit the refinement score ranks.
_Avoid_: Patch, tile, block

**Coverage floor**:
The refinement points placed from mask geometry alone, before any measurement, so that no region is left without enough neighbors for the fixed-radius strain estimator.
_Avoid_: Minimum refinement, edge padding

**Refinement score**:
A region's priority for refinement points, rising with coarse strain-gradient magnitude and lack of coverage, and suppressed (never raised) by tracking uncertainty.
_Avoid_: Saliency, informativeness

## Measurement

**Reference frame**:
The frame in which the specimen mask is given and query points are placed; all strains are expressed at material positions in this frame.
_Avoid_: First frame (when meaning the configuration)

**Strain field**:
The 2D Green–Lagrange strain (ε_xx, ε_yy, ε_xy) over the specimen, in the reference configuration.
_Avoid_: Strain map (except for its visualisation), engineering strain

**Loading-axis component**:
The normal component of the strain field along a dataset's loading direction; the headline quantity scored.
_Avoid_: Axial strain, ε_xx (unless the loading axis is x)

**Strain estimator**:
The procedure turning scattered tracked displacements into the strain field by a local fit over neighboring query points, identical across methods; it comes in a fixed-radius and a k-nearest variant.
_Avoid_: Strain fit, interpolator

**Lost track**:
A query point the tracker reports as invisible or unreliable in a frame; it is left out of that frame's estimate, still counts against the query budget, and is never replaced.
_Avoid_: Dropped point, outlier

**Oracle tracking**:
A tracking condition on synthetic sequences in which each method receives the true positions of its own query points only, standing in for a perfect tracker.
_Avoid_: Perfect tracker, ground-truth tracking

**Tracker input**:
The frame exactly as the tracker sees it after cropping to the specimen ROI and resizing; pixel quantities such as erosion and speckle size are measured in it.
_Avoid_: Model resolution (when meaning the image), resized frame

## Evaluation

**Speckle stratum**:
A pre-registered speckle-size bin in tracker-input pixels (1.5–3 px or 3–5 px) by which every metric is reported.
_Avoid_: Speckle bucket, texture class

**Evaluation grid**:
A fixed set of reference-frame positions, independent of every method's query points, where all errors are measured.
_Avoid_: Test points, probe points

**Coverage loss**:
The share of the evaluation grid where a method's strain estimator gives no answer; reported alongside error, never folded into it.
_Avoid_: Missing values, NaN rate

**High-strain region**:
The evaluation-grid points in the top 10% of ground-truth loading-axis component magnitude within the mask, used only for scoring.
_Avoid_: Hotspot, necking zone (unless physically a neck)

**Synthetic sequence**:
A rendered multi-frame video of a speckled specimen deformed by a known field, from a generator frozen before any method is run.
_Avoid_: Simulated data, fake video

**Control sequence**:
A synthetic sequence with smooth strain and no localized band, where adaptive placement should bring no advantage.

**Dev sequence**:
A sequence used to tune the refinement-score weights; never used for reported results.
_Avoid_: Validation set

**Test sequence**:
A held-out sequence scored only after the weights are frozen.
_Avoid_: Eval set

**Dense reference**:
A tracking run with far more query points than the query budget, used on real data as a consistency target, not ground truth.
_Avoid_: Ground truth, gold standard

**Specimen ROI**:
The crop of a real frame around the specimen that is passed to the tracker in place of the full frame.
_Avoid_: Crop, window

## Assistant

**Results assistant**:
A hosted language model that answers questions about completed runs only through read-only tools over computed outputs; it never computes, estimates, or launches runs.
_Avoid_: Meta assistant, chatbot, AI analyst
