# Decision-rule analysis

Issue #12 implements the decision rule in the [experiment spec](experiment_spec.md).
The analysis reads completed records; it does not run experiments or tune parameters.
The experiment grid and evaluation metrics are supplied by issue #8.

## Input contract

Select the completed run IDs from one experiment campaign explicitly. Each record
uses the experiment runner's configuration fields:

- `dataset_split`: only `test` records enter the analysis; dev/custom runs are ignored.
- `synthetic_case`: `seed` identifies the sequence and `kind` distinguishes
  `necking`, `uniform`, and `smooth` sequences.
- `speckle_stratum`, `spacing`, `lattice_offset`, `method`, `tracker`, `estimator`.
- Methods are `lattice_baseline` and `adaptive`; nominal tracking is
  `cotracker3_offline` and the primary estimator is `k_nearest`.

Primary metric keys are `full_field_exx_mae` and `high_strain_exx_mae`.
The latter is the per-run high-strain-region error computed by evaluation, never
inferred from full-field error. The existing bundled secondary keys are
`fixed_radius_exx_mae` and `fixed_radius_coverage_loss`; evaluation can also supply
`fixed_radius_high_strain_exx_mae`.

Every sequence × spacing × tracker × lattice offset must have exactly one record
for each method and estimator, with equal query budgets and matching sequence,
tracking, and evaluation configuration. Sequence identity and speckle stratum
must remain consistent across offsets and conditions. Optional provenance and
protocol fields, when recorded, must agree between partners; a field present
only on one partner also fails that comparison.
Incomplete pairs and duplicate cells are errors rather than silently omitted runs.
Nominal co-primary MAEs must be finite and nonnegative; undefined nominal MAEs
cannot establish the decision rule. A nominal analysis requires at least two
independent necking sequences for the bootstrap and at least one control sequence
for the full-field comparison. It need not have equal numbers of offsets per
sequence: whole-cluster resampling retains each selected sequence's offsets.

Standalone `fixed_radius` records can use the canonical metric keys. Bundled
fixed-radius metrics are normalized into a separate comparison automatically.
Selecting both representations for the same cell is an ambiguous duplicate.

## Computation

Pair adaptive and lattice-baseline runs by sequence and lattice offset within
each spacing and tracking condition. Give each paired run equal weight when
averaging its MAE. At the nominal spacing of 6.4 px under CoTracker, use k-nearest:

1. On necking test sequences, draw whole sequences with replacement, preserving
   all paired lattice offsets in each draw. Use 10,000 draws and the 2.5th and
   97.5th percentiles of the pooled mean adaptive-minus-baseline difference.
   The high-strain criterion passes only when the upper endpoint is below zero.
2. Across all test sequences, including uniform-extension and smooth controls,
   require mean adaptive full-field MAE ≤ 1.05 × mean baseline full-field MAE.
   This is the specified mean-error threshold; the bootstrap CI applies to the
   high-strain criterion. If baseline MAE is zero, adaptive MAE must also be zero.
3. Overall pass requires both criteria, pooled over the two speckle strata.

Retain the bootstrap seed and selected run IDs with the report for reproducibility.
Report each speckle stratum separately as descriptive summaries. Other spacings,
oracle tracking, and fixed-radius estimates are descriptive and carry no pass/fail
decision. Undefined descriptive metrics remain undefined rather than being
averaged over only the successful runs.

## Python interface

```python
from strainsight.analysis import analyze_results
from strainsight.results import ResultsStore

store = ResultsStore("outputs/results")
selected_run_ids = ["<baseline run ID>", "<adaptive run ID>"]  # Supply the full selection.
report = analyze_results(store, selected_run_ids, seed=42)
# The selection must include the nominal necking and control pairs described above.
print(report.primary.passes)
print(report.primary.high_strain_ci)
print(report.primary.full_field.relative_change)
for comparison in report.descriptive:
    print(comparison.condition, comparison.pooled, comparison.per_stratum)
```

`AnalysisReport` retains the input run IDs, seed, 10,000-resample count, 95%
confidence level, `PrimaryDecision`, and a tuple of `DescriptiveComparison` values.
Each metric comparison includes pair/sequence counts, both method means, the
adaptive-minus-baseline difference, and the relative change as a fraction (0.05
means 5%). The relative change is `None` when baseline error is zero and adaptive
error is positive; the non-inferiority check still fails in that case. Zero error
for both methods gives relative change zero and passes non-inferiority.
Effects are computed as the stable mean of paired differences, matching the
bootstrap statistic. At floating-point precision, this can retain a tiny effect
even when the separately reported method means round to the same value. The
high-strain criterion requires both a negative observed effect and an entirely
negative confidence interval.

The primary result exposes both criterion booleans and the combined `passes`
property. Descriptive comparisons include the nominal condition as well as the
other selected conditions, so the complete budget curve is available. Each has a
pooled summary and a map of per-speckle-stratum summaries, without verdicts. In
both primary and descriptive summaries, high-strain-region E_XX MAE is summarized
over necking sequences only; other metrics include controls. Pair counts refer
to all selected pairs for that metric, including pairs with undefined descriptive
values. A metric's method mean is `None` if any selected value for that method is
undefined, and its effect is `None` if either mean is undefined.

Invalid analysis inputs raise `ValueError`; an unknown selected run ID raises
the results store's `KeyError`. No other stored runs are discovered or selected.
Input order does not affect statistics. The bootstrap uses a local PCG64 random
generator initialized by the supplied nonnegative integer seed, independent of
NumPy's global random state.

Selecting run IDs defines the campaign under analysis; it does not certify that
all pre-registered sequences, offsets, or experiment cells have been collected.
This analysis is separate from the pre-registration freeze and does not replace
the requirement to commit that freeze before any held-out experiment runs.
