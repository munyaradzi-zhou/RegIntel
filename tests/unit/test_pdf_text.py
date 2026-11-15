from regintel.ingestion.pdf_text import (
    FORMULA_PLACEHOLDER,
    clean_pages,
    group_footnotes,
    is_line_numbered,
    join_lines,
    repair_glyphs,
    split_footnotes,
)

HEADER = "Contains Nonbinding Recommendations"


def _pages(*bodies: str) -> list[str]:
    return [f"{HEADER}\n{i + 1}\n{body}" for i, body in enumerate(bodies)]


def test_running_header_and_page_numbers_removed() -> None:
    cleaned = clean_pages(_pages("Alpha text.", "Beta text.", "Gamma text."))
    assert cleaned == ["Alpha text.", "Beta text.", "Gamma text."]


def test_trailing_line_numbers_removed_only_in_numbered_documents() -> None:
    lines = [[f"Some line of text {n}" for n in range(30, 40)]]
    assert is_line_numbered(lines)
    assert not is_line_numbered([["Sentence one.", "Sentence two about 21 CFR part 11."]])
    cleaned = clean_pages(["\n".join(f"word{n} continues {n}" for n in range(30, 40))])
    assert "continues 30" not in cleaned[0] and "word30 continues word31" in cleaned[0]


def test_line_numbers_removed_from_a_numbered_appendix_only() -> None:
    body = [[f"Page {p} sentence number {n} of the guidance." for n in range(12)] for p in "AB"]  # unnumbered
    appendix = [f"appendix text line {n}" for n in range(271, 283)]
    cleaned = clean_pages(["\n".join(body[0]), "\n".join(body[1]), "\n".join(appendix)])
    assert "line 271" not in cleaned[2] and cleaned[2].startswith("appendix text line appendix")
    assert "sentence number 3 of" in cleaned[0]


def test_table_of_contents_lines_dropped() -> None:
    page = "I. INTRODUCTION ................................ 1\nII. BACKGROUND .......... 3\nReal body text."
    assert clean_pages([page]) == ["Real body text."]


def test_wrapped_lines_join_and_hyphenation_mends() -> None:
    assert (
        join_lines(["The sponsor should", "ensure that the trial is"]) == "The sponsor should ensure that the trial is"
    )
    assert join_lines(["a well-designed proto-", "col is needed."]) == "a well-designed protocol is needed."


def test_heading_followed_by_capitalised_line_starts_new_paragraph() -> None:
    assert join_lines(["3. Pivotal BE Studies", "General recommendations follow."]) == (
        "3. Pivotal BE Studies\nGeneral recommendations follow."
    )


def test_wrapped_numbered_provision_joins() -> None:
    joined = join_lines(["9.1 The quality and amount of information should", "be fit for purpose."])
    assert joined == "9.1 The quality and amount of information should be fit for purpose."


def test_capitalised_heading_wrapping_over_two_lines_joins() -> None:
    joined = join_lines(["A.2. A FRAMEWORK TO ALIGN PLANNING, DESIGN", "AND INTERPRETATION"])
    assert joined == "A.2. A FRAMEWORK TO ALIGN PLANNING, DESIGN AND INTERPRETATION"


def test_blank_line_is_a_paragraph_boundary() -> None:
    assert clean_pages(["First line without stop\n\nSecond paragraph."]) == [
        "First line without stop\nSecond paragraph."
    ]


def test_symbol_font_bullets_mapped_and_garbled_formulas_replaced() -> None:
    assert repair_glyphs("\uf0b7 Data element G.k.2.4") == "\u2022 Data element G.k.2.4"
    garbled = "For PK parameters with a \u074f\u0bd0\u0bcb \u0d52 0.294, use the reference-scaled procedure"
    assert (
        repair_glyphs(garbled)
        == f"For PK parameters with a {FORMULA_PLACEHOLDER} 0.294, use the reference-scaled procedure"
    )
    assert repair_glyphs("Plain text, 80.00-125.00%") == "Plain text, 80.00-125.00%"


PAGE_WITH_NOTES = [
    "Contains Nonbinding Recommendations",
    "the citizen petition process.4 To establish BE, the calculated confidence interval",
    "should fall within a BE limit, usually 80.00",
    "line three of the body",
    "line four of the body",
    "line five of the body",
    " ",
    "4 For example, to submit an ANDA that is not the same as its RLD, an applicant first must",
    "obtain permission from FDA.",
    "5 21 CFR 314.3(b).",
]


def test_footnote_zone_separated_from_body() -> None:
    body, zone = split_footnotes(PAGE_WITH_NOTES)
    assert body[-1] == " " and zone[0].startswith("4 For example")
    assert group_footnotes([z.strip() for z in zone if z.strip()]) == [
        "Footnote 4: For example, to submit an ANDA that is not the same as its RLD, an applicant first must "
        "obtain permission from FDA.",
        "Footnote 5: 21 CFR 314.3(b).",
    ]


def test_numbered_line_without_a_reference_mark_is_not_a_footnote() -> None:
    lines = [*PAGE_WITH_NOTES[:6], " ", "7 Annexes", "text of the annex"]
    assert split_footnotes(lines) == (lines, [])
