import pytest

from regintel.corpus.manifest import GuidanceDoc
from regintel.ingestion.chunking import (
    MIN_WORDS,
    TARGET_WORDS,
    Heading,
    chunk_document,
    parse_heading,
    prose_share,
    section_locator,
)
from regintel.ingestion.pdf_text import Page

DOC = GuidanceDoc(
    id="fda-test",
    issuer="FDA",
    jurisdiction="US",
    title="Bioequivalence Studies",
    reference="Guidance for Industry",
    date="2026-05",
    status="final",
    topics=["bioequivalence"],
    url="https://example.org/doc.pdf",
    sha256="0" * 64,
    pages=2,
)


@pytest.mark.parametrize(
    ("line", "number", "level"),
    [
        ("III. ESTABLISHING BE", "III", 1),
        ("A. In Vivo BE Studies With PK Endpoints", "A", 2),
        ("1. General Considerations", "1", 3),
        ("4.1.1 Study design", "4.1.1", 5),
        ("A.2. A FRAMEWORK TO ALIGN PLANNING", "A.2", 3),
        ("VI.B.7. Submission of reports", "VI.B.7", 3),
        ("Q4. Is FDA recommending that electronic records be certified?", "Q4", 3),
    ],
)
def test_heading_styles(line: str, number: str, level: int) -> None:
    heading = parse_heading(line)
    assert heading is not None
    assert (heading.number, heading.level) == (number, level)


@pytest.mark.parametrize(
    "line",
    [
        "1 This guidance has been prepared by the Office of Generic Drugs in the Center for Drug Evaluation.",
        "0. Not a section",
        "4.1 lower-case continuation is not a heading",
        "Plain sentence without a number.",
    ],
)
def test_non_headings(line: str) -> None:
    assert parse_heading(line) is None


def test_long_numbered_provision_is_a_locator() -> None:
    line = (
        "9.1 The quality and amount of the information generated in a clinical trial "
        "should be fit for purpose and sufficient."
    )
    heading = parse_heading(line)
    assert heading is not None and heading.number == "9.1"


def test_locators() -> None:
    fda_path = [Heading("III", "ESTABLISHING BE", 1), Heading("A", "In Vivo", 2), Heading("1", "General", 3)]
    assert section_locator(fda_path) == "III.A.1"
    assert section_locator([Heading("4", "Requirements", 3), Heading("4.1.1", "Study design", 5)]) == "4.1.1"
    assert section_locator([Heading("III", "X", 1), Heading("Q4", "question", 3)]) == "Q4"
    assert section_locator([]) == "preamble"


def _words(n: int, word: str = "word") -> str:
    return " ".join([word] * n) + "."


def test_chunks_follow_sections_and_carry_context() -> None:
    pages = [
        Page(1, f"III. ESTABLISHING BE\n{_words(40, 'alpha')}\nA. In Vivo Studies\n{_words(40, 'beta')}"),
        Page(2, f"1. General Considerations\n{_words(40, 'gamma')}"),
    ]
    chunks = chunk_document(DOC, pages)
    assert [c.section for c in chunks] == ["III", "III.A", "III.A.1"]
    assert chunks[2].page_start == 2
    assert chunks[1].section_title == "In Vivo Studies"
    assert chunks[0].header.startswith("[FDA · Guidance for Industry · 2026-05] Bioequivalence Studies — section III")
    assert "alpha" in chunks[0].text and "beta" not in chunks[0].text
    assert chunks[0].search_text.startswith(chunks[0].header)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_long_sections_split_and_fragments_dropped() -> None:
    paragraphs = "\n".join(_words(120, f"p{chr(97 + i)}") for i in range(8))
    chunks = chunk_document(DOC, [Page(1, f"II. BACKGROUND\n{paragraphs}\nIII. NEXT\ntoo short")])
    assert all(len(c.text.split()) <= TARGET_WORDS + 121 for c in chunks)
    assert len(chunks) >= 3
    assert all(len(c.text.split()) >= MIN_WORDS for c in chunks)
    assert all(c.section == "II" for c in chunks)


def test_tables_and_formula_blocks_dropped() -> None:
    schedule = "Study Week -2 -1 0 1 2 3 4 5 6 8 Informed consent x x x History x Physical exam x x x"
    prose = "The sponsor should report suspected unexpected serious adverse reactions within fifteen days."
    assert prose_share(schedule) < 0.3 < prose_share(prose)
    text = "\n".join(["II. SCHEDULE", schedule, "III. REPORTING", prose])
    chunks = chunk_document(DOC, [Page(1, text)])
    assert [c.section for c in chunks] == ["III"]


def test_sentence_across_a_page_break_is_rejoined_and_footnotes_follow_it() -> None:
    pages = [
        Page(
            5,
            "II. BACKGROUND\nTo establish BE, the interval should fall within a BE limit, usually 80.00",
            ("Footnote 4: See 21 CFR 314.93.",),
        ),
        Page(6, "percent to 125.00 percent for the ratio of the product averages.\nA second paragraph follows here."),
    ]
    chunks = chunk_document(DOC, pages)
    text = chunks[0].text
    assert "usually 80.00 percent to 125.00 percent for the ratio" in text
    assert text.index("125.00 percent") < text.index("Footnote 4") < text.index("A second paragraph")


def test_footnotes_of_a_page_ending_a_sentence_stay_on_that_page() -> None:
    pages = [
        Page(
            1,
            "II. BACKGROUND\nThe sponsor must report every case within the stated period.",
            ("Footnote 1: See section 312.32 of the regulations.",),
        ),
        Page(2, "III. NEXT\nA new section starts here with plenty of words to keep."),
    ]
    sections = {c.section: c.text for c in chunk_document(DOC, pages)}
    assert "Footnote 1" in sections["II"] and "Footnote 1" not in sections["III"]
