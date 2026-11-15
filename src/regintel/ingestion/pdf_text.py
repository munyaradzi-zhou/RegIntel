"""PDF text extraction and cleaning for regulatory guidance.

Guidance PDFs carry noise that hurts retrieval and, worse, breaks quote verification:
running headers ("Contains Nonbinding Recommendations"), page numbers ("6/27"), and in
FDA documents a line number at the end of every line ("... part 11.  While 36"). This module
removes that noise per document, using what repeats across pages rather than fixed rules.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pypdf import PdfReader

if TYPE_CHECKING:
    from pathlib import Path

_PAGE_NUMBER = re.compile(r"^\s*(?:page\s+)?\d{1,3}(?:\s*(?:/|of)\s*\d{1,3})?\s*$", re.IGNORECASE)
_TRAILING_LINE_NUMBER = re.compile(r"\s+(\d{1,4})\s*$")
_HYPHEN_BREAK = re.compile(r"(\w)-\s*$")
_TOC_LINE = re.compile(r"(?:\.\s?){4,}\s*\d*\s*$|…{2,}|^table of contents$", re.IGNORECASE)

# Symbol-font glyphs that some PDFs map into the Unicode Private Use Area.
_SYMBOL_FONT = str.maketrans({"\uf0b7": "\u2022", "\uf02d": "-", "\uf061": "\u03b1", "\uf8e9": "\u00a9"})
# Equations set in math fonts come out of pypdf as letters from unrelated scripts (Hebrew through
# Malayalam, Hangul Jamo, Ethiopic) or as unmapped private-use code points. A run of them, with the spaces and
# operators between, becomes one placeholder so the garbage is neither embedded nor quoted.
_GARBLED_CHAR = "[\u0590-\u0dff\u1100-\u137f\ue000-\uf8ff]"
_GARBLED_RUN = re.compile(rf"{_GARBLED_CHAR}+(?:[\s\W\d]*{_GARBLED_CHAR}+)*")
FORMULA_PLACEHOLDER = "[formula]"
# A footnote starts with its number at the start of a line ("4 For example, ...").
_FOOTNOTE_START = re.compile(r"^(\d{1,3})\s+(\S.*)$")


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, as printed by a PDF viewer
    text: str
    notes: tuple[str, ...] = ()  # the page's footnotes, as "Footnote 4: ..."


def _normalise_for_repeat(line: str) -> str:
    """Lines that differ only by digits (page numbers) count as the same running header."""
    return re.sub(r"\d+", "#", line.strip().lower())


def _repeated_lines(pages: list[list[str]], min_share: float = 0.5) -> set[str]:
    """Lines near the top or bottom of a page that recur on at least `min_share` of pages."""
    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for lines in pages:
        edge = [ln for ln in lines[:3] + lines[-3:] if ln.strip()]
        counts.update({_normalise_for_repeat(ln) for ln in edge})
    threshold = max(2, int(len(pages) * min_share))
    return {line for line, n in counts.items() if n >= threshold and line}


def is_line_numbered(pages: list[list[str]]) -> bool:
    """True when most non-empty lines end in an integer that mostly increases (FDA drafting style)."""
    numbers = []
    total = 0
    for lines in pages:
        for line in lines:
            if not line.strip():
                continue
            total += 1
            match = _TRAILING_LINE_NUMBER.search(line)
            if match:
                numbers.append(int(match.group(1)))
    if total == 0 or len(numbers) / total < 0.6:
        return False
    increasing = sum(1 for a, b in itertools.pairwise(numbers) if b > a)
    return increasing / max(1, len(numbers) - 1) > 0.8


def _is_line_numbered_page(lines: list[str], min_lines: int = 10) -> bool:
    """Some guidances number only part of the document (an appendix), so check a page on its own."""
    return sum(1 for ln in lines if ln.strip()) >= min_lines and is_line_numbered([lines])


def split_footnotes(lines: list[str]) -> tuple[list[str], list[str]]:
    """Separate a page's footnote zone from its body.

    The zone starts after a blank line in the lower part of the page, with a line that begins with
    a number that also appears in the body glued to a word or punctuation (the reference mark,
    "...process.4"). Without this, a sentence that runs onto the next page has the footnotes
    spliced into its middle, which breaks both retrieval and quote verification.
    """
    start = None
    for i in range(len(lines) - 1, 0, -1):
        match = _FOOTNOTE_START.match(lines[i].strip())
        if lines[i - 1].strip() or not match or i < len(lines) * 0.4:
            continue
        body = " ".join(lines[:i])
        if re.search(rf"[A-Za-z).,;:\u2019\u201d]{match.group(1)}(?!\d)", body):
            start = i
    return (lines, []) if start is None else (lines[:start], lines[start:])


def group_footnotes(lines: list[str]) -> list[str]:
    """Cleaned footnote-zone lines -> one string per footnote, wrapped lines re-joined."""
    notes: list[str] = []
    for line in lines:
        match = _FOOTNOTE_START.match(line)
        if match:
            notes.append(f"Footnote {match.group(1)}: {match.group(2)}")
        elif notes and _HYPHEN_BREAK.search(notes[-1]) and line[:1].islower():
            notes[-1] = _HYPHEN_BREAK.sub(r"\1", notes[-1]) + line
        elif notes:
            notes[-1] = f"{notes[-1]} {line}"
    return [re.sub(r"\s{2,}", " ", n) for n in notes]


def clean_pages(raw_pages: list[str]) -> list[str]:
    return [text for text, _ in clean_pages_with_notes(raw_pages)]


def clean_pages_with_notes(raw_pages: list[str]) -> list[tuple[str, list[str]]]:
    split = [page.splitlines() for page in raw_pages]
    repeated = _repeated_lines(split)
    numbered = is_line_numbered(split)
    cleaned = []
    for page_lines in split:
        body, zone = split_footnotes(page_lines)
        page_numbered = numbered or _is_line_numbered_page(body)

        def keep(line: str, page_numbered: bool = page_numbered) -> str | None:
            """The cleaned line; '' for a blank line; None for running headers, page numbers, TOC lines."""
            if page_numbered:
                line = _TRAILING_LINE_NUMBER.sub("", line)
            stripped = repair_glyphs(line).strip()
            if stripped and (
                _PAGE_NUMBER.match(stripped)
                or _TOC_LINE.search(stripped)
                or _normalise_for_repeat(stripped) in repeated
            ):
                return None
            return stripped

        kept: list[str] = []
        for line in body:
            stripped = keep(line)
            if stripped is None:
                continue
            if not stripped:
                if kept and kept[-1] != PARAGRAPH_BREAK:
                    kept.append(PARAGRAPH_BREAK)  # blank lines are the PDF's own paragraph boundaries
                continue
            kept.append(stripped)
        notes = group_footnotes([s for s in map(keep, zone) if s])
        cleaned.append((join_lines(kept), notes))
    return cleaned


PARAGRAPH_BREAK = ""


def repair_glyphs(line: str) -> str:
    """Map symbol-font bullets back to Unicode and replace garbled formula text with a placeholder."""
    return _GARBLED_RUN.sub(FORMULA_PLACEHOLDER, line.translate(_SYMBOL_FONT))


def join_lines(lines: list[str]) -> str:
    """Re-flow wrapped lines into paragraphs, mending words hyphenated across a line break.

    A paragraph ends at a blank line, at sentence punctuation, before a numbered line, and
    after a numbered heading ("3. Pivotal BE Studies") when the next line starts with a
    capital. A numbered line continued in lower case ("9.1 The quality ... should" / "be fit
    for purpose") is a wrapped provision and is joined, as are headings in capitals that wrap.
    """
    out: list[str] = []
    new_paragraph = True
    for line in lines:
        if line == PARAGRAPH_BREAK:
            new_paragraph = True
            continue
        if not new_paragraph and out and _HYPHEN_BREAK.search(out[-1]) and line[:1].islower():
            out[-1] = _HYPHEN_BREAK.sub(r"\1", out[-1]) + line
        elif not new_paragraph and out and _continues(out[-1], line):
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
        new_paragraph = False
    return "\n".join(re.sub(r"\s{2,}", " ", p) for p in out)


# Numbered or bulleted line: 4.1.1 / III. / A. / A.2. / Q4. / (a) / bullets and dashes (incl. U+2013)
_BLOCK_START = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?|[IVX]{1,5}\.|[A-H](?:\.\d+)*\.|[a-h]\.|Q\d{1,3}\.|\(\w{1,4}\)|[\u2022\u25cf\u25aa\-\u2013])\s"
)


def _starts_block(line: str) -> bool:
    return bool(_BLOCK_START.match(line))


def _continues(previous: str, line: str) -> bool:
    if _starts_block(line) or previous.endswith((".", ":", ";", "?")):
        return False
    if _starts_block(previous) and line[:1].isupper():
        return previous.isupper() and line.isupper()  # a heading in capitals wrapped over two lines
    return True


def read_pdf(path: Path) -> list[Page]:
    reader = PdfReader(path)
    raw = [page.extract_text() or "" for page in reader.pages]
    return [Page(i + 1, text, tuple(notes)) for i, (text, notes) in enumerate(clean_pages_with_notes(raw))]
