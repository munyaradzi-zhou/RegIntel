"""Code-enforced grounding: a statement is shown only if its citations survive these checks.

1. The cited passage exists (the model cannot cite a passage it was not given).
2. The quote appears in that passage, after normalising what PDFs and models change without
   changing meaning: Unicode form, curly quotes, case, hyphens and whitespace. PDF extraction
   joins "life-" + "threatening" at line breaks and splits words ("doc uments", "proce dure"),
   so the comparison ignores hyphens and spaces; the characters must still match in order.
   A quote may skip text with "..." if its parts appear in order.
3. Every number in the statement appears, by value, in one of its verified quotes or in the
   cited passage's header (document reference, section locator); 80 = 80.00, .025 = 0.025.
   This catches the most damaging error in regulatory answers: a right-sounding citation
   attached to a wrong time limit.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING

from regintel.answering.schema import CheckedStatement, DraftStatement, VerifiedCitation

if TYPE_CHECKING:
    from regintel.corpus.manifest import Manifest
    from regintel.ingestion.chunking import Chunk

_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"), "-")
_QUOTES = {ord("\u2018"): "'", ord("\u2019"): "'", ord("\u201c"): '"', ord("\u201d"): '"'}
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|\u2026)\s*")
# A number, but not the tail of a locator or page reference ("III.B.1", "p.15").
_NUMBER = re.compile(r"(?<![A-Za-z]\.)(?<![\d.])\d+(?:[.,]\d+)*|(?<![\w.])\.\d+")
_THOUSANDS = re.compile(r"\d{1,3}(?:,\d{3})+")
MIN_QUOTE_WORDS = 3
IGNORE_SPACES = True  # False reproduces the first verifier (for measuring what the change is worth)


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_DASHES).translate(_QUOTES)
    return re.sub(r"\s+", " ", re.sub(r"\s*-\s*", "", text)).strip().casefold()


def normalise_numbers(text: str) -> str:
    """For number checks: keep the hyphen as a separator so '80.00-125.00' stays two numbers."""
    text = unicodedata.normalize("NFKC", text).translate(_DASHES)
    return re.sub(r"\s+", " ", text).casefold()


def _compact(text: str) -> str:
    return normalise(text).replace(" ", "") if IGNORE_SPACES else normalise(text)


def quote_in_passage(quote: str, passage: str) -> bool:
    """Every '...'-separated part of the quote appears in the passage, in order (spaces ignored)."""
    parts = [p for p in (normalise(part) for part in _ELLIPSIS.split(quote)) if p]
    if not parts or len(" ".join(parts).split()) < MIN_QUOTE_WORDS:
        return False
    haystack = _compact(passage)
    position = 0
    for part in parts:
        needle = part.replace(" ", "") if IGNORE_SPACES else part
        found = haystack.find(needle, position)
        if found < 0:
            return False
        position = found + len(needle)
    return True


def _value(token: str) -> Decimal | str:
    """Numeric value of a token ('80.00' -> 80, '1,000' -> 1000, '.025' -> 0.025); locators such as
    '4.1.8' are not numbers and compare as text."""
    cleaned = token.replace(",", "") if _THOUSANDS.fullmatch(token) else token.replace(",", ".")
    try:
        return Decimal(cleaned).normalize()
    except InvalidOperation:
        return token


def numbers_in(text: str) -> set[str]:
    """Numbers stated in the text. Zero is skipped: it appears in notation such as AUC(0-t), not as a claim."""
    tokens = {m.rstrip(".,") for m in _NUMBER.findall(normalise_numbers(text))}
    return {t for t in tokens if t and _value(t) != 0}


def contains_number(number: str, text: str) -> bool:
    return _value(number.rstrip(".,%")) in {_value(t) for t in _NUMBER.findall(normalise_numbers(text))}


def source_label(chunk: Chunk, manifest: Manifest | None) -> str:
    if not chunk.page_start:  # a computed record (terminology entry, FAERS count): its label says what it is
        return chunk.section_title or chunk.doc_id
    doc = manifest.by_id().get(chunk.doc_id) if manifest else None
    name = doc.citation_label if doc else chunk.doc_id
    pages = f"p.{chunk.page_start}" if chunk.page_start == chunk.page_end else f"pp.{chunk.page_start}-{chunk.page_end}"
    section = "" if chunk.section in ("", "preamble") else f" · §{chunk.section}"
    return f"{name}{section}, {pages}"


def check_statement(statement: DraftStatement, passages: list[Chunk], manifest: Manifest | None) -> CheckedStatement:
    problems: list[str] = []
    verified: list[VerifiedCitation] = []
    for citation in statement.citations:
        if not 1 <= citation.passage <= len(passages):
            problems.append(f"cites P{citation.passage}, which was not retrieved")
            continue
        chunk = passages[citation.passage - 1]
        if not quote_in_passage(citation.quote, chunk.text):
            problems.append(f"quote not found in P{citation.passage}: {citation.quote[:80]!r}")
            continue
        if not chunk.page_start:
            pages = ""
        elif chunk.page_start == chunk.page_end:
            pages = str(chunk.page_start)
        else:
            pages = f"{chunk.page_start}-{chunk.page_end}"
        verified.append(
            VerifiedCitation(
                chunk.chunk_id, chunk.doc_id, chunk.section, pages, source_label(chunk, manifest), citation.quote
            )
        )
    if not statement.citations:
        problems.append("no citation")
    support = " ".join(
        [c.quote for c in verified] + [p.header for p in passages if p.chunk_id in {c.chunk_id for c in verified}]
    )
    unsupported = sorted(n for n in numbers_in(statement.text) if not contains_number(n, support))
    if verified and unsupported:
        problems.append(f"numbers not in the cited quotes: {', '.join(unsupported)}")
    ok = bool(verified) and not unsupported
    return CheckedStatement(statement.text, verified, ok, problems)
