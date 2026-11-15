import json
from pathlib import Path

import pytest

from regintel.evaluation.retrieval_metrics import (
    ExpectedSource,
    GoldenQuestion,
    load_golden_set,
    score_question,
    section_matches,
    summarise,
)
from regintel.ingestion.chunking import Chunk
from regintel.retrieval.base import Hit

GOLDEN = Path(__file__).parents[1] / "evaluation" / "datasets" / "golden_set.json"


def hit(doc: str, section: str) -> Hit:
    return Hit(Chunk(f"{doc}#{section}", doc, "EU", section, "", 1, 1, "text", "header"), 1.0)


@pytest.mark.parametrize(
    ("locator", "expected", "match"),
    [
        ("4.1.8", "4.1", True),
        ("4.1", "4.1", True),
        ("4.10", "4.1", False),
        ("III.A.10", "III.A", True),
        ("Q5", "Q", False),
    ],
)
def test_section_prefix_matching(locator: str, expected: str, match: bool) -> None:
    assert section_matches(locator, expected) is match


def question(*sources: tuple[str, list[str]], qtype: str = "single") -> GoldenQuestion:
    return GoldenQuestion(
        id="q", type=qtype, question="?", facts=[], expected=[ExpectedSource(doc_id=d, sections=s) for d, s in sources]
    )


def test_doc_and_section_ranks() -> None:
    result = score_question(
        question(("ema-be", ["4.1.8"])), [hit("fda", "I"), hit("ema-be", "4.1.1"), hit("ema-be", "4.1.8")]
    )
    assert (result.doc_rank, result.section_rank) == (2, 3)
    assert result.retrieved[0] == "fda §I"


def test_document_level_question_counts_any_section() -> None:
    result = score_question(question(("gvp", [])), [hit("gvp", "VI")])
    assert result.section_rank == 1


def test_comparison_needs_every_document() -> None:
    q = question(("ema-be", []), ("fda-be", []), qtype="comparison")
    assert not score_question(q, [hit("ema-be", "4")]).all_docs_found
    assert score_question(q, [hit("ema-be", "4"), hit("fda-be", "I")]).all_docs_found


def test_multihop_needs_every_source_at_section_level() -> None:
    q = question(("fih", ["7.2"]), ("fih", ["8.2.6"]), qtype="multihop")
    one_section = score_question(q, [hit("fih", "7.2"), hit("fih", "7.3")])
    assert one_section.all_docs_found and not one_section.all_sources_found
    both = [score_question(q, [hit("fih", "8.2.6"), hit("fih", "7.2")]), one_section]
    assert summarise(both, k=5)["multihop_all_sources@5"] == 0.5


def test_summary_metrics() -> None:
    q = question(("d", ["1"]))
    results = [
        score_question(q, [hit("d", "1")]),
        score_question(q, [hit("x", "1"), hit("d", "2")]),
        score_question(q, []),
    ]
    summary = summarise(results, k=5)
    assert summary["doc_hit@1"] == round(1 / 3, 3)
    assert summary["doc_hit@5"] == round(2 / 3, 3)
    assert summary["section_hit@5"] == round(1 / 3, 3)
    assert summary["doc_mrr"] == round((1 + 1 / 2) / 3, 3)


def test_golden_set_is_well_formed() -> None:
    golden = load_golden_set(GOLDEN)
    ids = [q.id for q in golden.questions]
    assert len(ids) == len(set(ids)) == 60
    manifest = json.loads((Path(__file__).parents[2] / "data" / "manifest.json").read_text(encoding="utf-8"))
    known_docs = {d["id"] for d in manifest["guidance"]}
    for q in golden.questions:
        if q.type == "unanswerable":
            assert not q.expected and not q.facts
        else:
            assert q.expected and q.facts
            assert {s.doc_id for s in q.expected} <= known_docs
        if q.type in ("comparison", "multihop"):
            assert len(q.expected) >= 2
