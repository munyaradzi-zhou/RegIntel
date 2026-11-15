from regintel.answering.review import Review, StatementVerdict, apply_review, review_input, subject_in_passages
from regintel.answering.schema import AnswerResult, CheckedStatement, Usage, VerifiedCitation


def statement(text: str, verified: bool = True) -> CheckedStatement:
    citation = VerifiedCitation("d#1", "d", "1", "1", "Source · §1, p.1", "should be validated")
    return CheckedStatement(text, [citation] if verified else [], verified, [] if verified else ["quote not found"])


def result(*statements: CheckedStatement) -> AnswerResult:
    return AnswerResult(
        "What does Annex 11 require for ML models?", "grounded", list(statements), None, [], Usage(), 0, 0
    )


def review(*verdicts: str, addresses: bool = True, subject: str | None = None) -> Review:
    items = [StatementVerdict(statement=i, verdict=v, reason="r") for i, v in enumerate(verdicts, start=1)]  # type: ignore[arg-type]
    return Review(verdicts=items, addresses_question=addresses, missing_subject=subject)


def test_reviewer_sees_only_shown_statements_numbered_in_order() -> None:
    text = review_input(result(statement("A."), statement("dropped", verified=False), statement("B.")))
    assert "S1: A." in text and "S2: B." in text and "dropped" not in text


def test_overreach_is_dropped_and_label_recomputed() -> None:
    reviewed = apply_review(
        result(statement("A."), statement("x", False), statement("B.")), review("supported", "overreach"), []
    )
    assert [s.verified for s in reviewed.statements] == [True, False, False]
    assert reviewed.statements[2].problems == ["reviewer: overreach: r"]
    assert reviewed.label == "partially_grounded"


def test_everything_rejected_is_withheld() -> None:
    reviewed = apply_review(result(statement("A.")), review("unsupported"), [])
    assert reviewed.label == "not_grounded" and "withheld" in reviewed.text


def test_missing_subject_declines_only_when_no_passage_names_it() -> None:
    passages = ["This annex applies to all forms of computerised systems. The application should be validated."]
    declined = apply_review(
        result(statement("A.")), review("supported", addresses=False, subject="machine learning models"), passages
    )
    assert declined.label == "declined" and "machine learning models" in declined.text
    kept = apply_review(
        result(statement("A.")), review("supported", addresses=False, subject="computerised systems"), passages
    )
    assert kept.label == "grounded"


def test_subject_matching_handles_plurals_and_hyphens() -> None:
    assert subject_in_passages("non-serious ICSR time frame", ["Non-serious valid ICSRs ... within the time frames"])
    assert not subject_in_passages("machine learning", ["computerised systems"])
    assert subject_in_passages("", ["anything"])  # nothing named: never decline on it
