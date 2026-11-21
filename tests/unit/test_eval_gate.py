"""The CI regression gate: metric and prompt comparison, verifier replay, and the report."""

from __future__ import annotations

from regintel.answering.schema import DraftAnswer, DraftCitation, DraftStatement
from regintel.evaluation import provenance
from regintel.evaluation.gate import (
    VerifierCase,
    compare_metrics,
    compare_prompts,
    passed,
    render_markdown,
    run_verifier_cases,
)
from regintel.ingestion.chunking import Chunk
from regintel.prompts import prompt_tags, prompt_versions, prompts


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="ema-bioequivalence",
        jurisdiction="EU",
        section="4.1.8",
        section_title="Evaluation",
        page_start=14,
        page_end=15,
        header="[EMA · test] Guideline — section 4.1.8 Evaluation",
        text=text,
    )


PASSAGE = _chunk(
    "ema-bioequivalence#0001",
    "The 90% confidence interval for the ratio should be contained within the acceptance interval of 80.00-125.00%.",
)


def _case(text: str, quote: str, expected: bool, passage: int = 1) -> VerifierCase:
    draft = DraftAnswer(
        answerable=True,
        statements=[DraftStatement(text=text, citations=[DraftCitation(passage=passage, quote=quote)])],
        decline_reason=None,
    )
    return VerifierCase(
        id=text[:10],
        kind="test",
        question="What is the acceptance interval?",
        passages=[PASSAGE.chunk_id],
        draft=draft,
        expected_verified=[expected],
        expected_label="grounded" if expected else "not_grounded",
    )


def test_metric_within_tolerance_passes_and_a_larger_drop_fails() -> None:
    rows = compare_metrics({"a": 0.80, "b": 0.80}, {"a": 0.79, "b": 0.70}, tolerance=0.02)
    status = {r.name: r.status for r in rows}
    assert status == {"a": "ok", "b": "regressed"}
    assert not passed(rows, [])


def test_improvement_new_and_missing_metrics() -> None:
    rows = compare_metrics({"a": 0.5, "gone": 1.0}, {"a": 0.6, "fresh": 0.3})
    status = {r.name: r.status for r in rows}
    assert status == {"a": "improved", "fresh": "new", "gone": "missing"}
    assert not passed(rows, [])  # a metric must not disappear silently
    assert next(r for r in rows if r.name == "a").delta == 0.1


def test_verifier_metrics_get_no_tolerance() -> None:
    rows = compare_metrics({"verifier.labels_match": 1.0}, {"verifier.labels_match": 0.99}, tolerance=0.05)
    assert rows[0].status == "regressed"


def test_prompt_changes_need_a_version_bump() -> None:
    base = {"writer": {"version": "v5", "sha256": "aaa"}, "router": {"version": "v1", "sha256": "bbb"}}
    now = {"writer": {"version": "v5", "sha256": "ccc"}, "router": {"version": "v2", "sha256": "ddd"}}
    rows = {r.name: r for r in compare_prompts(base, now)}
    assert rows["writer"].status == "unversioned_change"
    assert rows["router"].status == "bumped"
    assert rows["router"].current == "v2+ddd"
    assert not passed([], list(rows.values()))
    assert passed([], [rows["router"]])  # a bump is reported, not failed


def test_prompt_added_or_removed() -> None:
    rows = {r.name: r.status for r in compare_prompts({"old": {"version": "v1", "sha256": "x"}},
                                                      {"new": {"version": "v1", "sha256": "y"}})}  # fmt: skip
    assert rows == {"new": "new", "old": "missing"}


def test_verifier_replay_matches_and_reports_mismatches() -> None:
    chunks = {PASSAGE.chunk_id: PASSAGE}
    good = _case("The interval is 80.00-125.00%.", "within the acceptance interval of 80.00-125.00%", True)
    misquote = _case("The interval is 80.00-125.00%.", "within the acceptance range of 80.00-125.00%", False)
    outcome = run_verifier_cases([good, misquote], chunks)
    assert outcome.mismatches == []
    assert outcome.metrics() == {"verifier.statement_decisions_match": 1.0, "verifier.labels_match": 1.0}
    assert (outcome.kept, outcome.dropped) == (1, 1)

    wrong = _case("The interval is 70-130%.", "within the acceptance interval of 80.00-125.00%", True)
    outcome = run_verifier_cases([wrong], chunks)
    assert outcome.metrics()["verifier.statement_decisions_match"] == 0.0
    assert "expected [True]" in outcome.mismatches[0]


def test_empty_verifier_outcome_counts_as_matching() -> None:
    assert run_verifier_cases([], {}).metrics() == {
        "verifier.statement_decisions_match": 1.0,
        "verifier.labels_match": 1.0,
    }


def test_report_shows_failures_bumps_and_mismatches() -> None:
    metrics = compare_metrics({"retrieval.x": 0.9}, {"retrieval.x": 0.5})
    prompt_rows = compare_prompts(
        {"writer": {"version": "v5", "sha256": "a"}, "router": {"version": "v1", "sha256": "b"}},
        {"writer": {"version": "v5", "sha256": "c"}, "router": {"version": "v2", "sha256": "d"}},
    )
    report = render_markdown(metrics, prompt_rows, 0.02, notes=["a note"], mismatches=["q1 changed"])
    assert "❌ failed" in report
    assert "-0.400" in report
    assert "Bump the version constant" in report
    assert "make eval-answers" in report
    assert "q1 changed" in report and "> a note" in report


def test_report_passes_when_nothing_changed() -> None:
    rows = compare_metrics({"retrieval.x": 0.9}, {"retrieval.x": 0.9})
    assert "✅ passed" in render_markdown(rows, [], 0.02)


def test_prompt_registry_versions_and_hashes() -> None:
    names = {"writer", "router", "reviewer", "rewriter", "agent"}
    assert set(prompts()) == names
    versions = prompt_versions()
    assert versions["writer"]["version"].startswith("v")
    assert all(len(v["sha256"]) == 12 for v in versions.values())
    tags = prompt_tags()
    assert tags["writer"] == f"{versions['writer']['version']}+{versions['writer']['sha256']}"


def test_run_metadata_records_prompts_and_commit() -> None:
    meta = provenance.run_metadata()
    assert meta["prompts"] == prompt_versions()
    assert "recorded_at" in meta
    deployed = {"writer": "v5+abc"}
    assert provenance.run_metadata(deployed)["prompts"] == deployed


def test_git_commit_without_git(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(provenance.shutil, "which", lambda _: None)
    assert provenance.git_commit() is None
