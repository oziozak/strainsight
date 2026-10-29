from pathlib import Path

import pytest

from strainsight.assistant import (GroundingError, ModelReply, ResultsAssistant, ResultsTools,
                                 ToolCall, check_grounding)
from strainsight.results import ResultsStore, RunRecord


class ScriptedModel:
    def __init__(self, *replies: ModelReply) -> None:
        self.replies = iter(replies)

    def respond(self, messages, tools) -> ModelReply:
        return next(self.replies)


def test_model_answers_with_metrics_from_a_completed_run(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path)
    record = RunRecord("baseline-run", {"method": "lattice_baseline"},
                       {"full_field_exx_mae": 0.04}, 317, 1.5)
    store.write(record)
    model = ScriptedModel(
        ModelReply(tool_calls=(ToolCall("lookup", "metrics", {"run_id": "baseline-run"}),)),
        ModelReply(answer="The full-field error is {{lookup:full_field_exx_mae}}."),
    )

    answer = ResultsAssistant(tmp_path, model).answer("What is the full-field error?")

    assert answer == "The full-field error is 0.04 [run baseline-run]."
    assert store.read("baseline-run") == record


def test_model_lists_runs_and_uses_code_computed_method_comparison(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path)
    common = {"tracker": "oracle", "spacing": 8, "seed": 3, "speckle_stratum": "1.5–3 px"}
    store.write(RunRecord("baseline-run", dict(common, method="lattice_baseline"),
                          {"full_field_exx_mae": 0.04}, 317, 1.5))
    store.write(RunRecord("adaptive-run", dict(common, method="adaptive"),
                          {"full_field_exx_mae": 0.02}, 317, 2.0))

    class ListingModel(ScriptedModel):
        def respond(self, messages, tools) -> ModelReply:
            assert {tool.name for tool in tools} == {
                "list_runs", "metrics", "strain_at_point", "strain_in_region", "compare_methods"}
            if messages[-1].role == "tool" and messages[-1].tool_call_id == "runs":
                assert '"adaptive-run"' in messages[-1].content
                assert '"baseline-run"' in messages[-1].content
            return super().respond(messages, tools)

    model = ListingModel(
        ModelReply(tool_calls=(ToolCall("runs", "list_runs", {}),)),
        ModelReply(tool_calls=(ToolCall("compare", "compare_methods", {
            "baseline_run_id": "baseline-run", "candidate_run_id": "adaptive-run",
            "metric": "full_field_exx_mae"}),)),
        ModelReply(answer="Error changes by {{compare:difference}}, or {{compare:percent_change}}%."),
    )

    answer = ResultsAssistant(tmp_path, model).answer("Compare the completed methods.")

    assert answer == ("Error changes by -0.02 [run baseline-run] [run adaptive-run], "
                      "or -50.0 [run baseline-run] [run adaptive-run]%.")


def test_model_queries_saved_point_and_region_strain_with_coverage(tmp_path: Path) -> None:
    from strainsight.results import StrainSample

    store = ResultsStore(tmp_path)
    store.write(RunRecord("strain-run", {}, {}, 317, 1.5, strain_samples={
        "k_nearest": [StrainSample(4, 0., 0., 0.125, -0.05, 0.),
                      StrainSample(4, 4., 0., 0.375, -0.1, 0.),
                      StrainSample(4, 8., 0., None, None, None),
                      StrainSample(8, 0., 0., 0.5, -0.2, 0.)],
        "fixed_radius": [StrainSample(4, 0., 0., 0.25, -0.05, 0.)],
    }))
    model = ScriptedModel(
        ModelReply(tool_calls=(
            ToolCall("point", "strain_at_point", {"run_id": "strain-run", "frame": 4,
                     "x": 0., "y": 0., "component": "exx", "estimator": "fixed_radius"}),
            ToolCall("region", "strain_in_region", {"run_id": "strain-run", "frame": 4,
                     "x_min": 0., "y_min": 0., "x_max": 8., "y_max": 0.,
                     "component": "exx", "estimator": "k_nearest"}),
        )),
        ModelReply(answer="Point strain is {{point:value}}; region mean is {{region:mean}} "
                   "over {{region:defined_count}} of {{region:sample_count}} saved positions."),
    )

    answer = ResultsAssistant(tmp_path, model).answer("Report point and region strain.")

    assert answer == ("Point strain is 0.25 [run strain-run]; region mean is 0.25 [run strain-run] "
                      "over 2 [run strain-run] of 3 [run strain-run] saved positions.")


def test_model_reads_strain_persisted_by_the_experiment(tmp_path: Path) -> None:
    from strainsight.experiment import ExperimentConfig, run_experiment
    from strainsight.tracking import OracleTracker

    result = run_experiment(ExperimentConfig(frame_count=2, score_every=1),
                            OracleTracker(), ResultsStore(tmp_path))
    x, y = result.evaluation_grid[0]
    model = ScriptedModel(
        ModelReply(tool_calls=(ToolCall("point", "strain_at_point", {
            "run_id": result.record.run_id, "frame": 1, "x": float(x), "y": float(y),
            "component": "exx", "estimator": "k_nearest"}),)),
        ModelReply(answer="{{point:value}}"),
    )

    answer = ResultsAssistant(tmp_path, model).answer("What strain was measured here?")

    assert float(answer.split()[0]) == pytest.approx(0.105, abs=1e-12)
    assert f"[run {result.record.run_id}]" in answer


@pytest.mark.parametrize("template", [
    "Error is 0.04.",  # Reject even a literal that matches a tool value.
    "{{lookup:full_field_exx_mae}} plus 12.",
    "Error is ４.", "Error is ².", "Error is twenty percent.",
    "This is the sixth run.", "There are a dozen runs.",
    "Adaptive reduced error twofold.", "There are IV runs.", "There are a trillion runs.",
    "Error is 一.",
    "{{missing:full_field_exx_mae}}", "{{lookup:invented}}", "{{lookup}}",
])
def test_scripted_grounding_check_rejects_unsourced_numbers_and_references(
        tmp_path: Path, template: str) -> None:
    ResultsStore(tmp_path).write(RunRecord("run", {}, {"full_field_exx_mae": 0.04}, 317, 1.))
    model = ScriptedModel(
        ModelReply(tool_calls=(ToolCall("lookup", "metrics", {"run_id": "run"}),)),
        ModelReply(answer=template),
    )

    with pytest.raises(GroundingError):
        ResultsAssistant(tmp_path, model).answer("What is the error?")


def test_grounding_check_detects_a_number_added_after_rendering(tmp_path: Path) -> None:
    ResultsStore(tmp_path).write(RunRecord("run", {}, {"error": 0.04}, 317, 1.))
    result = ResultsTools(tmp_path).call("metrics", {"run_id": "run"})
    template = "Error is {{lookup:error}}."
    answer = "Error is 0.04 [run run]."
    check_grounding(answer, template, {"lookup": result})

    with pytest.raises(GroundingError):
        check_grounding(answer + " Improved by 25%.", template, {"lookup": result})


@pytest.mark.parametrize("baseline,candidate,difference,percentage", [
    (0., 0.02, 0.02, None), (None, 0.02, None, None), (0.04, None, None, None),
])
def test_comparisons_do_not_invent_missing_or_zero_denominator_percentages(
        tmp_path: Path, baseline, candidate, difference, percentage) -> None:
    store = ResultsStore(tmp_path)
    store.write(RunRecord("baseline", {"method": "lattice_baseline"}, {"error": baseline}, 317, 1.))
    store.write(RunRecord("adaptive", {"method": "adaptive"}, {"error": candidate}, 317, 1.))

    result = ResultsTools(tmp_path).call("compare_methods", {
        "baseline_run_id": "baseline", "candidate_run_id": "adaptive", "metric": "error"})

    assert result.values["difference"].value == difference
    assert result.values["percent_change"].value == percentage


@pytest.mark.parametrize("changed", [
    {"query_count": 100}, {"configuration": {"method": "adaptive", "tracker": "cotracker"}},
    {"configuration": {"method": "adaptive", "seed": 4}},
    {"configuration": {"method": "adaptive", "speckle_stratum": "3–5 px"}},
])
def test_comparison_rejects_unpaired_runs(tmp_path: Path, changed) -> None:
    from dataclasses import replace

    store = ResultsStore(tmp_path)
    baseline = RunRecord("baseline", {"method": "lattice_baseline"}, {"error": 0.04}, 317, 1.)
    candidate = replace(baseline, run_id="adaptive", configuration={"method": "adaptive"})
    store.write(baseline)
    store.write(replace(candidate, **changed))

    with pytest.raises(ValueError, match="paired runs"):
        ResultsTools(tmp_path).call("compare_methods", {
            "baseline_run_id": "baseline", "candidate_run_id": "adaptive", "metric": "error"})


def test_tools_cannot_write_or_create_a_results_store(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path)
    record = RunRecord("run", {}, {"error": 0.04}, 317, 1.)
    store.write(record)
    original = store.path.read_bytes()
    tools = ResultsTools(tmp_path)

    tools.call("list_runs", {})
    tools.call("metrics", {"run_id": "run"})
    with pytest.raises(ValueError, match="unknown read-only tool"):
        tools.call("run_experiment", {})
    with pytest.raises(ValueError, match="invalid arguments"):
        tools.call("metrics", {"run_id": "run", "spacing": 4})
    with pytest.raises(KeyError):
        tools.call("metrics", {"run_id": "'; DELETE FROM runs; --"})

    assert store.read("run") == record
    assert store.path.read_bytes() == original
    missing = tmp_path / "missing"
    import sqlite3
    with pytest.raises(sqlite3.OperationalError):
        ResultsTools(missing).call("list_runs", {})
    assert not missing.exists()


@pytest.mark.parametrize("point", [(0., 0.), (2., 2.)])
def test_model_reports_unavailable_strain_without_interpolation(tmp_path: Path, point) -> None:
    # This also exercises backwards-compatible records without saved strain.
    ResultsStore(tmp_path).write(RunRecord("old-run", {}, {}, 317, 1.))
    model = ScriptedModel(
        ModelReply(tool_calls=(ToolCall("point", "strain_at_point", {
            "run_id": "old-run", "frame": 4, "x": point[0], "y": point[1],
            "component": "exx", "estimator": "k_nearest"}),)),
        ModelReply(answer="Point strain is {{point:value}}."),
    )

    assert ResultsAssistant(tmp_path, model).answer("What strain is here?") == (
        "Point strain is unavailable [run old-run].")


def test_model_can_discover_saved_scored_frames_and_configuration(tmp_path: Path) -> None:
    ResultsStore(tmp_path).write(RunRecord("run", {
        "method": "lattice_baseline", "spacing": 6.4, "scored_frames": [4, 8],
    }, {}, 523, 1.))
    model = ScriptedModel(
        ModelReply(tool_calls=(ToolCall("lookup", "metrics", {"run_id": "run"}),)),
        ModelReply(answer="Spacing is {{lookup:configuration.spacing}}; last scored frame is "
                   "{{lookup:configuration.scored_frames.1}}."),
    )

    assert ResultsAssistant(tmp_path, model).answer("What configuration was used?") == (
        "Spacing is 6.4 [run run]; last scored frame is 8 [run run].")


@pytest.mark.parametrize("changed", [
    {"frame": True}, {"frame": 1.5}, {"frame": -1}, {"x": float("nan")},
    {"x": float("inf")}, {"component": "exz"}, {"estimator": "invented"},
])
def test_strain_tools_reject_invalid_queries(tmp_path: Path, changed) -> None:
    arguments = {"run_id": "run", "frame": 4, "x": 0., "y": 0.,
                 "component": "exx", "estimator": "k_nearest"}
    with pytest.raises(ValueError, match="invalid argument"):
        ResultsTools(tmp_path).call("strain_at_point", dict(arguments, **changed))


def test_tool_outputs_from_a_previous_question_cannot_ground_a_new_answer(tmp_path: Path) -> None:
    ResultsStore(tmp_path).write(RunRecord("run", {}, {"error": 0.04}, 317, 1.))
    model = ScriptedModel(
        ModelReply(tool_calls=(ToolCall("lookup", "metrics", {"run_id": "run"}),)),
        ModelReply(answer="{{lookup:error}}"),
        ModelReply(answer="{{lookup:error}}"),
    )
    assistant = ResultsAssistant(tmp_path, model)
    assert assistant.answer("What is the error?") == "0.04 [run run]"
    with pytest.raises(GroundingError, match="unknown tool output"):
        assistant.answer("What about another run?")


def test_reused_tool_call_ids_cannot_replace_an_answer_source(tmp_path: Path) -> None:
    ResultsStore(tmp_path).write(RunRecord("run", {}, {"error": 0.04}, 317, 1.))
    reply = ModelReply(tool_calls=(ToolCall("lookup", "metrics", {"run_id": "run"}),))
    with pytest.raises(ValueError, match="unique"):
        ResultsAssistant(tmp_path, ScriptedModel(reply, reply)).answer("What is the error?")


def test_listing_run_metadata_does_not_materialize_saved_strain(tmp_path: Path) -> None:
    import tracemalloc

    from strainsight.results import StrainSample

    store = ResultsStore(tmp_path)
    # Thousands of evaluation positions per run are normal; listing needs metadata only.
    samples = [StrainSample(4, float(x), 0., 0.1, -0.05, 0.) for x in range(5000)]
    for index in range(3):
        store.write(RunRecord(f"run-{index}", {"method": "lattice_baseline"}, {}, 317, 1.,
                              strain_samples={"k_nearest": samples}))
    tracemalloc.start()
    try:
        result = ResultsTools(tmp_path).call("list_runs", {})
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result.values["runs.2.run_id"].value == "run-2"
    assert peak_bytes < 2_000_000  # Metadata fits easily; decoded fields exceed this severalfold.
