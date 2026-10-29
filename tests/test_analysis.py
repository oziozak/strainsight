from dataclasses import replace

import pytest

from strainsight.analysis import analyze_results
from strainsight.results import ResultsStore, RunRecord


def write_pair(store, sequence, *, kind="necking", stratum="1.5–3 px",
               offset=(0., 0.), spacing=6.4, tracker="cotracker3_offline",
               estimator="k_nearest", baseline_full=.02, adaptive_full=.0208,
               baseline_high=.1, adaptive_high=.08, secondary=False):
    ids = []
    for method, full, high in (("lattice_baseline", baseline_full, baseline_high),
                               ("adaptive", adaptive_full, adaptive_high)):
        run_id = f"{sequence}-{offset}-{spacing}-{tracker}-{estimator}-{method}"
        configuration = {
            "dataset_split": "test", "synthetic_case": {"seed": sequence, "kind": kind},
            "speckle_stratum": stratum, "spacing": spacing, "lattice_offset": list(offset),
            "method": method, "tracker": tracker, "estimator": estimator,
        }
        metrics = {"full_field_exx_mae": full, "high_strain_exx_mae": high,
                   "coverage_loss": 0.}
        if secondary:
            metrics.update({"fixed_radius_exx_mae": .5 if method == "adaptive" else .01,
                            "fixed_radius_high_strain_exx_mae": .9,
                            "fixed_radius_coverage_loss": .2})
        store.write(RunRecord(run_id, configuration, metrics, 100, .1))
        ids.append(run_id)
    return ids


def campaign(store, **kwargs):
    ids = []
    for sequence, kind, stratum in (
        (1, "necking", "1.5–3 px"), (2, "necking", "3–5 px"),
        (3, "uniform", "1.5–3 px"), (4, "smooth", "3–5 px"),
    ):
        for offset in ((0., 0.), (1., 2.), (3., 4.)):
            ids.extend(write_pair(store, sequence, kind=kind, stratum=stratum,
                                  offset=offset, **kwargs))
    return ids


def test_passing_decision_is_structured_and_pooled_over_test_pairs(tmp_path):
    store = ResultsStore(tmp_path)
    ids = campaign(store)

    report = analyze_results(store, ids, seed=42)

    assert report.run_ids == tuple(ids)
    assert report.seed == 42
    assert report.bootstrap_resamples == 10_000
    assert report.confidence_level == .95
    assert report.primary.pair_count == 12
    assert report.primary.sequence_count == 4
    assert report.primary.high_strain.pair_count == 6
    assert report.primary.high_strain.sequence_count == 2
    assert report.primary.high_strain.baseline_mean == pytest.approx(.1)
    assert report.primary.high_strain.adaptive_mean == pytest.approx(.08)
    assert report.primary.high_strain.difference == pytest.approx(-.02)
    assert report.primary.high_strain_ci == pytest.approx((-.02, -.02))
    assert report.primary.high_strain_improves
    assert report.primary.full_field.relative_change == pytest.approx(.04)
    assert report.primary.noninferiority_margin == .05
    assert report.primary.full_field_noninferior
    assert report.primary.passes


def replace_selected(store, ids, index, **changes):
    record = store.read(ids[index])
    replacement = replace(record, run_id=record.run_id + "-modified", **changes)
    store.write(replacement)
    ids[index] = replacement.run_id


@pytest.mark.parametrize("problem", [
    "missing", "duplicate-cell", "duplicate-id", "sequence", "offset", "tracking",
    "spacing", "estimator", "budget", "kind", "stratum", "sequence-configuration",
])
def test_incomplete_ambiguous_or_mismatched_pairs_are_rejected(tmp_path, problem):
    store = ResultsStore(tmp_path)
    ids = campaign(store)
    record = store.read(ids[1])
    config = record.configuration
    if problem == "missing":
        ids.pop(1)
    elif problem == "duplicate-id":
        ids.append(ids[1])
    elif problem == "duplicate-cell":
        duplicate = replace(record, run_id="another-run-for-same-cell")
        store.write(duplicate)
        ids.append(duplicate.run_id)
    elif problem == "budget":
        replace_selected(store, ids, 1, query_count=101)
    else:
        if problem == "sequence":
            config["synthetic_case"]["seed"] = 99
        elif problem == "kind":
            config["synthetic_case"]["kind"] = "smooth"
        elif problem == "sequence-configuration":
            config["synthetic_case"]["band_fwhm"] = 30.
        else:
            field, value = {
                "offset": ("lattice_offset", [1., 1.]),
                "tracking": ("tracker", "oracle"), "spacing": ("spacing", 8.),
                "estimator": ("estimator", "fixed_radius"),
                "stratum": ("speckle_stratum", "3–5 px"),
            }[problem]
            config[field] = value
        replace_selected(store, ids, 1, configuration=config)

    with pytest.raises(ValueError):
        analyze_results(store, ids, seed=42)


def test_cluster_bootstrap_retains_offsets_and_is_reproducible(tmp_path):
    store = ResultsStore(tmp_path)
    ids = []
    # Every cluster has mean effect -0.1, although offset effects cross zero.
    # A bootstrap of individual offsets would give a non-degenerate interval.
    for sequence, effects in ((1, (-.3, 0., 0.)), (2, (.1, -.2, -.2))):
        for offset, effect in enumerate(effects):
            ids.extend(write_pair(store, sequence, offset=(float(offset), 0.),
                                  baseline_high=1., adaptive_high=1. + effect))
    ids.extend(write_pair(store, 3, kind="smooth"))

    report = analyze_results(store, ids, seed=123)

    assert report.primary.high_strain_ci == pytest.approx((-.1, -.1))
    assert report == analyze_results(store, ids, seed=123)
    assert report.primary == analyze_results(store, list(reversed(ids)), seed=123).primary


def test_cluster_resampling_pools_all_offsets_even_with_unequal_cluster_sizes(tmp_path):
    store = ResultsStore(tmp_path)
    ids = []
    for offset in (0., 1., 2.):
        ids.extend(write_pair(store, 1, offset=(offset, 0.), adaptive_high=.01))
    ids.extend(write_pair(store, 2, adaptive_high=.13))
    ids.extend(write_pair(store, 3, kind="uniform"))

    primary = analyze_results(store, ids, seed=10).primary

    # Three pairs at -0.09 and one at +0.03: pooled effect is -0.06.
    # Two cluster draws have endpoints at -0.09 and +0.03.
    assert primary.high_strain.difference == pytest.approx(-.06)
    assert primary.high_strain_ci == pytest.approx((-.09, .03))
    assert not primary.high_strain_improves
    assert not primary.passes


@pytest.mark.parametrize("baseline,adaptive,noninferior,relative", [
    (.02, .021, True, .05), (.02, .021002, False, .0501),
    (0., 0., True, 0.), (0., .001, False, None),
])
def test_relative_full_field_margin_and_zero_baseline(tmp_path, baseline, adaptive,
                                                    noninferior, relative):
    store = ResultsStore(tmp_path)
    ids = campaign(store, baseline_full=baseline, adaptive_full=adaptive)

    primary = analyze_results(store, ids).primary

    assert primary.full_field_noninferior is noninferior
    if relative is None:
        assert primary.full_field.relative_change is None
    else:
        assert primary.full_field.relative_change == pytest.approx(relative)
    assert primary.passes is noninferior


@pytest.mark.parametrize("high", [.1, .11])
def test_zero_or_positive_high_strain_difference_fails(tmp_path, high):
    store = ResultsStore(tmp_path)
    ids = campaign(store, adaptive_high=high)

    primary = analyze_results(store, ids).primary

    assert primary.full_field_noninferior
    assert not primary.high_strain_improves
    assert not primary.passes


def test_controls_enter_full_field_but_not_necking_high_strain_decision(tmp_path):
    store = ResultsStore(tmp_path)
    ids = campaign(store, adaptive_full=.015)
    for index, run_id in enumerate(ids):
        record = store.read(run_id)
        if (record.configuration["method"] == "adaptive"
                and record.configuration["synthetic_case"]["kind"] != "necking"):
            metrics = dict(record.metrics, full_field_exx_mae=.03, high_strain_exx_mae=100.)
            replace_selected(store, ids, index, metrics=metrics)

    primary = analyze_results(store, ids).primary

    assert primary.high_strain_improves
    assert primary.full_field.adaptive_mean == pytest.approx(.0225)
    assert not primary.full_field_noninferior
    assert not primary.passes


def test_only_explicitly_selected_test_runs_affect_analysis(tmp_path):
    store = ResultsStore(tmp_path)
    ids = campaign(store)
    expected = analyze_results(store, ids, seed=42)
    # Unselected records include an incomplete pair and an extreme competing run.
    write_pair(store, 99, adaptive_high=100., adaptive_full=100.)
    record = replace(store.read(ids[0]), run_id="unselected-duplicate")
    store.write(record)
    # Selected dev/custom records need not be paired and must not affect inference.
    ignored_ids = []
    for split in ("dev", "custom"):
        ignored = replace(record, run_id=split,
                          configuration={"dataset_split": split})
        store.write(ignored)
        ignored_ids.append(ignored.run_id)

    assert analyze_results(store, ids, seed=42) == expected
    assert analyze_results(store, ids + ignored_ids, seed=42).primary == expected.primary


@pytest.mark.parametrize("nominal_high,passes", [(.08, True), (.12, False)])
def test_descriptive_conditions_and_strata_do_not_change_verdict(tmp_path, nominal_high, passes):
    store = ResultsStore(tmp_path)
    ids = campaign(store, adaptive_high=nominal_high, secondary=True)
    expected = analyze_results(store, ids, seed=42).primary
    for spacing in (4., 5.3, 8.):
        ids.extend(campaign(store, spacing=spacing, adaptive_high=.001,
                            adaptive_full=.001, secondary=True))
    ids.extend(campaign(store, tracker="oracle", adaptive_high=.9,
                        adaptive_full=.9, secondary=True))
    # Standalone secondary records at another spacing also pair by estimator.
    ids.extend(campaign(store, spacing=10., estimator="fixed_radius"))

    report = analyze_results(store, ids, seed=42)

    assert report.primary == expected
    assert report.primary.passes is passes
    by_condition = {(row.condition.spacing, row.condition.tracker, row.condition.estimator): row
                    for row in report.descriptive}
    assert len(by_condition) == 11
    nominal = by_condition[(6.4, "cotracker3_offline", "k_nearest")]
    assert nominal.pooled.pair_count == 12
    assert nominal.pooled.full_field.relative_change == pytest.approx(.04)
    assert set(nominal.per_stratum) == {"1.5–3 px", "3–5 px"}
    for summary in nominal.per_stratum.values():
        assert summary.pair_count == 6
        assert summary.high_strain.pair_count == 3
        assert not hasattr(summary, "passes")
    secondary = by_condition[(6.4, "cotracker3_offline", "fixed_radius")]
    assert secondary.pooled.full_field.adaptive_mean == .5
    assert secondary.pooled.metrics["coverage_loss"].baseline_mean == pytest.approx(.2)
    assert not hasattr(secondary, "passes")
    assert by_condition[(6.4, "oracle", "k_nearest")].pooled.full_field.adaptive_mean == .9


def test_per_stratum_failure_does_not_override_pooled_primary_decision(tmp_path):
    store = ResultsStore(tmp_path)
    ids = campaign(store)
    for index, run_id in enumerate(ids):
        record = store.read(run_id)
        if record.configuration["method"] == "adaptive":
            full = .018 if record.configuration["speckle_stratum"] == "1.5–3 px" else .022
            replace_selected(store, ids, index,
                             metrics=dict(record.metrics, full_field_exx_mae=full))

    report = analyze_results(store, ids, seed=42)

    nominal = report.descriptive[0]
    assert nominal.per_stratum["1.5–3 px"].full_field.relative_change == pytest.approx(-.1)
    assert nominal.per_stratum["3–5 px"].full_field.relative_change == pytest.approx(.1)
    assert report.primary.full_field.relative_change == pytest.approx(0.)
    assert report.primary.passes


@pytest.mark.parametrize("metric", ["full_field_exx_mae", "high_strain_exx_mae"])
@pytest.mark.parametrize("value", [None, -.001])
def test_undefined_or_negative_nominal_maes_cannot_establish_a_verdict(tmp_path, metric, value):
    store = ResultsStore(tmp_path)
    ids = campaign(store)
    record = store.read(ids[1])
    replace_selected(store, ids, 1, metrics=dict(record.metrics, **{metric: value}))

    with pytest.raises(ValueError, match=metric):
        analyze_results(store, ids)


def test_missing_nominal_metric_is_rejected(tmp_path):
    store = ResultsStore(tmp_path)
    ids = campaign(store)
    metrics = store.read(ids[1]).metrics
    del metrics["high_strain_exx_mae"]
    replace_selected(store, ids, 1, metrics=metrics)

    with pytest.raises(ValueError, match="high_strain_exx_mae"):
        analyze_results(store, ids)


@pytest.mark.parametrize("selection", ["empty", "oracle-only", "one-necking", "no-controls"])
def test_primary_requires_nominal_pairs_independent_necking_sequences_and_controls(tmp_path, selection):
    store = ResultsStore(tmp_path)
    ids = campaign(store, tracker="oracle" if selection == "oracle-only" else "cotracker3_offline")
    if selection == "empty":
        ids = []
    elif selection == "one-necking":
        ids = [run_id for run_id in ids
               if store.read(run_id).configuration["synthetic_case"]["seed"] != 2]
    elif selection == "no-controls":
        ids = [run_id for run_id in ids
               if store.read(run_id).configuration["synthetic_case"]["kind"] == "necking"]

    with pytest.raises(ValueError, match="nominal|necking|control"):
        analyze_results(store, ids)


def test_undefined_descriptive_mae_is_not_averaged_over_only_successful_runs(tmp_path):
    store = ResultsStore(tmp_path)
    ids = campaign(store)
    other = campaign(store, tracker="oracle")
    record = store.read(other[1])
    replace_selected(store, other, 1, metrics=dict(record.metrics, full_field_exx_mae=None))

    report = analyze_results(store, ids + other)

    oracle = next(row for row in report.descriptive if row.condition.tracker == "oracle")
    assert oracle.pooled.full_field.pair_count == 12
    assert oracle.pooled.full_field.baseline_mean == pytest.approx(.02)
    assert oracle.pooled.full_field.adaptive_mean is None
    assert oracle.pooled.full_field.difference is None
    assert report.primary.passes


@pytest.mark.parametrize("field,value", [
    ("tracker_settings", {"checkpoint": "another-model", "visibility_threshold": .5}),
    ("support_grid_count", 49), ("frames_sha256", "another-video"),
    ("generator_sha256", "another-generator"), ("scored_frames", [4, 8]),
    ("k", [6, 7, 8]), ("fixed_radius", 20.),
])
def test_paired_tracking_evaluation_and_generator_settings_must_match(tmp_path, field, value):
    store = ResultsStore(tmp_path)
    ids = campaign(store)
    record = store.read(ids[1])
    replace_selected(store, ids, 1, configuration=dict(record.configuration, **{field: value}))

    with pytest.raises(ValueError, match=field):
        analyze_results(store, ids)


def test_sequence_identity_and_stratum_are_consistent_across_offsets(tmp_path):
    store = ResultsStore(tmp_path)
    ids = campaign(store)
    # Change both partners at one offset, so each pair is individually consistent.
    for index in (2, 3):
        record = store.read(ids[index])
        replace_selected(store, ids, index,
                         configuration=dict(record.configuration, speckle_stratum="3–5 px"))

    with pytest.raises(ValueError, match="sequence"):
        analyze_results(store, ids)


@pytest.mark.parametrize("field,value", [
    ("estimator", "unknown"), ("method", "unknown"),
    ("spacing", -6.4), ("lattice_offset", [1.]),
    ("synthetic_case", {"seed": 1, "kind": "unknown"}),
    ("speckle_stratum", "unknown"),
])
def test_malformed_test_record_metadata_is_rejected(tmp_path, field, value):
    store = ResultsStore(tmp_path)
    ids = campaign(store)
    record = store.read(ids[1])
    replace_selected(store, ids, 1, configuration=dict(record.configuration, **{field: value}))

    with pytest.raises(ValueError):
        analyze_results(store, ids)


def test_missing_metadata_is_reported_as_invalid_analysis_input(tmp_path):
    store = ResultsStore(tmp_path)
    ids = campaign(store)
    replace_selected(store, ids, 1, configuration={"dataset_split": "test"})

    with pytest.raises(ValueError, match="metadata"):
        analyze_results(store, ids)


def test_non_degenerate_bootstrap_is_seeded_without_using_global_random_state(tmp_path):
    import numpy as np

    store = ResultsStore(tmp_path)
    ids = []
    for sequence, effect in enumerate((-.18, -.13, -.09, -.04, -.015, .02, .08, .17), start=1):
        ids.extend(write_pair(store, sequence, baseline_high=1., adaptive_high=1. + effect))
    ids.extend(write_pair(store, 9, kind="smooth"))
    np.random.seed(1)
    first = analyze_results(store, ids, seed=42)
    np.random.seed(99)
    repeat = analyze_results(store, ids, seed=42)
    another = analyze_results(store, ids, seed=43)

    assert first == repeat
    assert first.primary.high_strain_ci != another.primary.high_strain_ci
    assert first.primary.high_strain_ci[0] < 0 < first.primary.high_strain_ci[1]


def test_unpaired_selected_descriptive_runs_are_also_rejected(tmp_path):
    store = ResultsStore(tmp_path)
    ids = campaign(store)
    other = write_pair(store, 1, tracker="oracle")

    with pytest.raises(ValueError, match="incomplete pair"):
        analyze_results(store, ids + other[:1])


def test_observed_paired_effect_stays_consistent_with_bootstrap_at_float_precision(tmp_path):
    import numpy as np

    store = ResultsStore(tmp_path)
    ids = []
    for sequence in (1, 2):
        for offset in (0., 1., 2.):
            adaptive = float(np.nextafter(.01, 0.)) if offset == 0. else .01
            ids.extend(write_pair(store, sequence, offset=(offset, 0.),
                                  baseline_high=.01, adaptive_high=adaptive))
    ids.extend(write_pair(store, 3, kind="uniform"))

    primary = analyze_results(store, ids).primary

    expected = -float(np.spacing(.01)) / 3
    # Separate rounded method means coincide, but the actual paired effect is negative.
    assert primary.high_strain.baseline_mean == primary.high_strain.adaptive_mean
    assert primary.high_strain.difference == expected
    assert primary.high_strain.relative_change == expected / .01
    assert primary.high_strain_ci == (expected, expected)
    assert primary.high_strain_improves
