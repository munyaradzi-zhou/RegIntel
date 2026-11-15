import pytest

from regintel.answering.pipeline import Answerer, label_for
from regintel.answering.schema import CheckedStatement, DraftAnswer, DraftCitation, DraftStatement, Usage
from regintel.answering.verify import check_statement, numbers_in, quote_in_passage
from regintel.ingestion.chunking import Chunk
from regintel.retrieval.base import Hit

E2A = Chunk(
    "ich-e2a#0010",
    "ich-e2a",
    "ICH",
    "III.B.1",
    "Fatal or Life-Threatening Unexpected ADRs",
    7,
    7,
    "Fatal or lifethreatening, unexpected ADRs occurring in clinical investigations qualify for very rapid "
    "reporting. Regulatory agencies should be notified as soon as possible but no later than 7 calendar days "
    "after first knowledge by the sponsor that a case qualifies, followed by as complete a report as possible "
    "within 8 additional calendar days.",
    "[ICH · E2A · 1994-10-27] Clinical Safety Data Management — section III.B.1",
)
EMA = Chunk(
    "ema-be#0040",
    "ema-be",
    "EU",
    "4.1.8",
    "Evaluation",
    15,
    16,
    "The 90% confidence interval for the ratio of the test and reference products should be contained "
    "within the acceptance interval of 80.00\u2013125.00%.",
    "[EMA · CPMP/EWP/QWP/1401/98 Rev. 1 · 2010-01-20] Guideline on the Investigation of Bioequivalence — section 4.1.8",
)


def statement(text: str, *citations: tuple[int, str]) -> DraftStatement:
    return DraftStatement(text=text, citations=[DraftCitation(passage=p, quote=q) for p, q in citations])


@pytest.mark.parametrize(
    ("quote", "found"),
    [
        ("no later than 7 calendar days after first knowledge", True),
        ("Fatal or life-threatening, unexpected ADRs", True),  # hyphen lost by PDF extraction
        ("\u201cno later than 7 calendar days\u201d", False),  # quote marks are not part of the text
        ("no later than 7 calendar days ... within 8 additional calendar days", True),  # in order
        ("within 8 additional calendar days ... no later than 7 calendar days", False),  # out of order
        ("no later than 5 calendar days", False),
        ("calendar days", False),  # too short to support anything
    ],
)
def test_quote_matching(quote: str, found: bool) -> None:
    assert quote_in_passage(quote, E2A.text) is found


def test_words_split_by_pdf_extraction_still_match() -> None:
    passage = "a conservative approach will be necessary e.g. Bonferroni's or a related proce dure."
    assert quote_in_passage(
        "a conservative approach will be necessary e.g. Bonferroni's or a related procedure", passage
    )
    assert not quote_in_passage("a conservative approach will be necessary e.g. Holm's procedure", passage)


def test_numbers_compare_by_value() -> None:
    assert numbers_in("AUC(0-t) at the .025 level, 1,000 subjects, section 4.1.8") == {".025", "1,000", "4.1.8"}
    checked = check_statement(
        statement("The CI must lie within 80-125%.", (1, "The 90% confidence interval ... 80.00-125.00%")), [EMA], None
    )
    assert checked.verified, checked.problems


def test_numbers_split_on_ranges() -> None:
    assert numbers_in("contained within 80.00-125.00%, i.e. 90%") == {"80.00", "125.00", "90"}
    assert numbers_in("section III.B.1, p.15, ICH E2A") == {"2"}


def test_verified_statement_carries_its_source() -> None:
    checked = check_statement(
        statement(
            "The 90% CI must lie within 80.00-125.00%.",
            (1, "The 90% confidence interval for the ratio ... within the acceptance interval of 80.00-125.00%"),
        ),
        [EMA],
        None,
    )
    assert checked.verified, checked.problems
    citation = checked.citations[0]
    assert (citation.chunk_id, citation.section, citation.pages) == ("ema-be#0040", "4.1.8", "15-16")
    assert citation.source_label == "ema-be · §4.1.8, pp.15-16"


def test_wrong_number_with_a_real_quote_is_rejected() -> None:
    checked = check_statement(
        statement(
            "Regulators must be told within 15 calendar days.",
            (1, "as soon as possible but no later than 7 calendar days"),
        ),
        [E2A],
        None,
    )
    assert not checked.verified
    assert "numbers not in the cited quotes: 15" in checked.problems[0]


def test_numbers_from_the_cited_header_are_allowed() -> None:
    checked = check_statement(
        statement(
            "Section III.B.1 of ICH E2A sets a 7-day limit.", (1, "no later than 7 calendar days after first knowledge")
        ),
        [E2A],
        None,
    )
    assert checked.verified, checked.problems


@pytest.mark.parametrize(
    ("draft", "problem"),
    [
        (statement("Seven days.", (3, "no later than 7 calendar days")), "cites P3, which was not retrieved"),
        (statement("Seven days."), "no citation"),
        (statement("Seven days.", (1, "within seven calendar days of notice")), "quote not found in P1"),
    ],
)
def test_unverifiable_citations(draft: DraftStatement, problem: str) -> None:
    checked = check_statement(draft, [E2A], None)
    assert not checked.verified and checked.problems[0].startswith(problem)


def _checked(*verified: bool) -> list[CheckedStatement]:
    return [CheckedStatement("s", [], v) for v in verified]


@pytest.mark.parametrize(
    ("answerable", "verified", "label"),
    [
        (True, (True, True), "grounded"),
        (True, (True, False), "partially_grounded"),
        (True, (False,), "not_grounded"),
        (False, (), "declined"),
        (True, (), "declined"),  # claims to answer but says nothing
        (False, (True,), "grounded"),  # partial answer with a stated gap still shows what is verified
    ],
)
def test_labels(answerable: bool, verified: tuple[bool, ...], label: str) -> None:
    assert label_for(answerable, _checked(*verified)) == label


class FakeModel:
    name = "fake"

    def __init__(self, draft: DraftAnswer) -> None:
        self.answer = draft
        self.seen: list[int] = []

    def draft(self, question: str, hits: list[Hit]) -> tuple[DraftAnswer, Usage]:
        self.seen.append(len(hits))
        return self.answer, Usage(input_tokens=100, calls=1)


class FakeRetriever:
    name = "fake"

    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        return self.hits[:top_k]


def test_pipeline_shows_only_verified_statements_and_numbers_sources() -> None:
    draft = DraftAnswer(
        answerable=True,
        statements=[
            statement(
                "EU: the 90% CI must lie within 80.00-125.00%.",
                (2, "The 90% confidence interval ... acceptance interval of 80.00-125.00%"),
            ),
            statement(
                "ICH E2A: 7 calendar days for fatal cases.", (1, "no later than 7 calendar days after first knowledge")
            ),
            statement("FDA: 3 days.", (1, "made-up words that are not there")),
        ],
        decline_reason=None,
    )
    web = Hit(Chunk("web@0", "web:https://fda.gov/x", "", "web", "", 0, 0, "summary", ""), 9.0)
    model = FakeModel(draft)
    result = Answerer(FakeRetriever([web, Hit(E2A, 2.0), Hit(EMA, 1.0)]), model).answer("q")
    assert model.seen == [2]  # the web pointer never reaches the answer model
    assert result.label == "partially_grounded"
    assert result.text.splitlines() == [
        "EU: the 90% CI must lie within 80.00-125.00%. [1]",
        "ICH E2A: 7 calendar days for fatal cases. [2]",
    ]
    assert [s.chunk_id for s in result.sources] == ["ema-be#0040", "ich-e2a#0010"]


def test_ungrounded_answer_is_withheld() -> None:
    draft = DraftAnswer(
        answerable=True,
        statements=[statement("Invented.", (1, "words not in the passage at all"))],
        decline_reason=None,
    )
    result = Answerer(FakeRetriever([Hit(E2A, 1.0)]), FakeModel(draft)).answer("q")
    assert result.label == "not_grounded" and result.sources == []
    assert "withheld" in result.text


def test_nothing_retrieved_declines_without_calling_the_model() -> None:
    model = FakeModel(DraftAnswer(answerable=True, statements=[], decline_reason=None))
    result = Answerer(FakeRetriever([]), model).answer("q")
    assert result.label == "declined" and model.seen == []
