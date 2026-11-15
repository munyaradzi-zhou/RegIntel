from regintel.answering.schema import AnswerResult, CheckedStatement, Usage, VerifiedCitation
from regintel.evaluation.answer_metrics import fact_found, score_answer, summarise_answers
from regintel.evaluation.retrieval_metrics import ExpectedSource, GoldenQuestion


def citation(doc: str, section: str) -> VerifiedCitation:
    return VerifiedCitation(f"{doc}#1", doc, section, "1", doc, "quote")


def result(label: str, *statements: CheckedStatement) -> AnswerResult:
    return AnswerResult("q", label, list(statements), None, [], Usage(input_tokens=1000, calls=1), 0.1, 2.0)  # type: ignore[arg-type]


def question(qtype: str, facts: list[str], *expected: tuple[str, list[str]]) -> GoldenQuestion:
    sources = [ExpectedSource(doc_id=d, sections=s) for d, s in expected]
    return GoldenQuestion(id="q1", type=qtype, question="?", expected=sources, facts=facts)  # type: ignore[arg-type]


def test_fact_matching_ignores_case_dashes_and_spacing() -> None:
    assert fact_found("80.00", "within 80.00\u2013125.00%")
    assert fact_found("5 times the terminal half-life", "at least 5 times the Terminal  half life")
    assert not fact_found("12", "at least 24 subjects")
    assert not fact_found("12", "at least 120 subjects") and fact_found("12", "at least 12 subjects")


def test_answered_question_scores_facts_and_sources() -> None:
    shown = CheckedStatement("EU range 80.00-125.00%.", [citation("ema-be", "4.1.8")], True)
    dropped = CheckedStatement("FDA says 3 days.", [], False, ["quote not found in P2: 'x'"])
    q = question("comparison", ["80.00", "125.00", "90%"], ("ema-be", ["4.1"]), ("fda-be", []))
    score = score_answer(q, result("partially_grounded", shown, dropped))
    assert score.answered and abs((score.fact_recall or 0) - 2 / 3) < 1e-9
    assert score.facts_missing == ["90%"]
    assert score.cited_expected and not score.cited_all_docs
    assert (score.statements, score.verified) == (2, 1)


def test_unanswerable_rewards_abstention() -> None:
    q = question("unanswerable", [])
    declined = score_answer(q, result("declined"))
    answered = score_answer(q, result("grounded", CheckedStatement("x", [citation("d", "1")], True)))
    summary = summarise_answers([declined, answered])
    assert summary["abstention_rate"] == 0.5
    assert declined.fact_recall is None


def test_withheld_answer_counts_as_not_answered() -> None:
    q = question("single", ["7"], ("ich-e2a", []))
    score = score_answer(q, result("not_grounded", CheckedStatement("7 days", [], False, ["no citation"])))
    assert not score.answered and score.fact_recall == 0.0
    summary = summarise_answers([score])
    assert summary["dropped_by_reason"] == {"no citation": 1}
    assert summary["statements_verified"] == 0.0
