"""Section-aware chunking.

Regulatory text is cited by section ("ICH E6(R3) 9.1", "FDA BE guidance III.A.1"), so a
chunk never crosses a section boundary and carries its section path. Each chunk also gets a
one-line contextual header naming its document, so that "the sponsor should..." from an EMA
guideline is not confused with the same words from an FDA one (LegalMind's "right number,
wrong document" lesson). The header is used for search only; `text` stays verbatim so cited
quotes can be checked against it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from regintel.corpus.manifest import GuidanceDoc
    from regintel.ingestion.pdf_text import Page

# Heading patterns, most specific first. Level decides where a heading sits in the section path.
_DECIMAL = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,4})\.?\s+(\S.*)$")  # 4.1.1 Study design / 8. Clinical ...
_ROMAN = re.compile(r"^([IVX]{1,5})\.\s+([A-Z].*)$")  # III. ESTABLISHING BE
_LETTER = re.compile(r"^([A-H])\.\s+([A-Z].*)$")  # A. In Vivo BE Studies
_LETTER_DECIMAL = re.compile(r"^([A-H]\.\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+([A-Z].*)$")  # A.2. A FRAMEWORK (ICH E9(R1))
_ROMAN_PATH = re.compile(r"^([IVX]{1,4}\.[A-H](?:\.\d{1,2}){0,4})\.?\s+([A-Z].*)$")  # VI.B.7. (EMA GVP)
_QUESTION = re.compile(r"^(Q\d{1,3})\.\s+(\S.*)$")  # Q4. Is FDA recommending ... (FDA Q&A guidances)

SHORT_HEADING = 90  # a line longer than this is body text that starts with a number
TARGET_WORDS = 350
MAX_WORDS = 500
MIN_WORDS = 8
MIN_PROSE_SHARE = 0.3  # below this share of word tokens, a chunk is a table, schedule or formula block
_PROSE_WORD = re.compile(r"[A-Za-z][a-z]+[.,;:)]?")


def prose_share(text: str) -> float:
    """Share of whitespace tokens that are ordinary words (not numbers, symbols, codes or 'x' marks)."""
    tokens = text.split()
    return sum(1 for t in tokens if _PROSE_WORD.fullmatch(t)) / len(tokens) if tokens else 0.0


@dataclass(frozen=True)
class Heading:
    number: str
    title: str
    level: int


@dataclass
class Section:
    path: list[Heading]
    paragraphs: list[tuple[int, str]] = field(default_factory=list)  # (page, paragraph)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    jurisdiction: str
    section: str  # e.g. "III.A.1" or "4.1.1"
    section_title: str
    page_start: int
    page_end: int
    text: str  # verbatim cleaned text: quotes are verified against this
    header: str  # contextual header, used for search only

    @property
    def search_text(self) -> str:
        return f"{self.header}\n{self.text}"


def parse_heading(line: str) -> Heading | None:
    """Recognise a heading or numbered provision.

    One hierarchy covers every outline style in the corpus: Roman numerals (FDA, and the top
    level of ICH E6/E9) are level 1, capital letters level 2, and decimal numbers sit below,
    deeper by one level per dot ("4.1.1" under "4.1" under "4").
    """
    if m := _QUESTION.match(line):
        return Heading(m.group(1), _short_title(m.group(2)), 3)
    if (m := _ROMAN_PATH.match(line)) and len(line) <= SHORT_HEADING:
        return Heading(m.group(1), _short_title(m.group(2)), 1 + m.group(1).count("."))
    if m := _LETTER_DECIMAL.match(line):
        return Heading(m.group(1), _short_title(m.group(2)), 2 + m.group(1).count("."))
    if m := _ROMAN.match(line):
        return Heading(m.group(1), _short_title(m.group(2)), 1) if len(line) <= SHORT_HEADING else None
    if m := _LETTER.match(line):
        return Heading(m.group(1), _short_title(m.group(2)), 2) if len(line) <= SHORT_HEADING else None
    if m := _DECIMAL.match(line):
        number, title = m.group(1), m.group(2).strip()
        if int(number.split(".")[0]) == 0 or not title[:1].isupper():
            return None
        has_dot = "." in number or line.startswith(f"{number}.")
        # "1 This guidance has been prepared by..." is a footnote, not a section: a bare number
        # without a dot only counts as a heading when the line is short and not a sentence.
        if not has_dot and (len(line) > SHORT_HEADING or line.endswith(".")):
            return None
        return Heading(number, _short_title(title), 3 + number.count("."))
    return None


def _short_title(text: str) -> str:
    words = text.split()
    return " ".join(words[:12]) + ("…" if len(words) > 12 else "")


def _continues_across_pages(previous: str, paragraph: str) -> bool:
    """The page broke mid-sentence: the last paragraph has no closing punctuation and the next page
    starts in lower case ("...usually 80.00" / "percent to 125.00 percent ...")."""
    return bool(previous) and not previous.rstrip().endswith((".", ":", ";", "?", "!")) and paragraph[:1].islower()


def split_sections(pages: list[Page]) -> list[Section]:
    """Paragraphs grouped under their heading path. A sentence broken by a page boundary is
    re-joined, and each page's footnotes follow the paragraph that was open when the page ended,
    so they never land inside a sentence."""
    sections = [Section(path=[])]
    path: list[Heading] = []
    pending: list[tuple[int, str]] = []  # the previous page's footnotes
    for page in pages:
        paragraphs = [p for p in page.text.splitlines() if p]
        current = sections[-1].paragraphs
        if (
            paragraphs
            and current
            and parse_heading(paragraphs[0]) is None
            and _continues_across_pages(current[-1][1], paragraphs[0])
        ):
            current[-1] = (current[-1][0], f"{current[-1][1]} {paragraphs.pop(0)}")
        current.extend(pending)
        for paragraph in paragraphs:
            heading = parse_heading(paragraph)
            if heading is not None:
                path = [h for h in path if h.level < heading.level] + [heading]
                sections.append(Section(path=list(path)))
                if len(paragraph) <= SHORT_HEADING and not heading.number.startswith("Q"):
                    continue  # pure heading line: the title is in the path, not the body
            sections[-1].paragraphs.append((page.number, paragraph))
        pending = [(page.number, note) for note in page.notes]
    sections[-1].paragraphs.extend(pending)
    return [s for s in sections if s.paragraphs]


def section_locator(path: list[Heading]) -> str:
    """How the section is cited: "4.1.1" or "Q4" when self-contained, else the outline path ("III.A.1")."""
    if not path:
        return "preamble"
    if "." in path[-1].number or path[-1].number.startswith("Q"):
        return path[-1].number
    return ".".join(h.number for h in path)


def make_header(doc: GuidanceDoc, locator: str, title: str) -> str:
    where = "" if locator == "preamble" else f" — section {locator} {title}".rstrip()
    return f"[{doc.issuer} · {doc.reference} · {doc.date}] {doc.title}{where}"


def chunk_document(doc: GuidanceDoc, pages: list[Page]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in split_sections(pages):
        locator = section_locator(section.path)
        title = section.path[-1].title if section.path else ""
        for group in _group_paragraphs(section.paragraphs):
            text = "\n".join(p for _, p in group)
            if len(text.split()) < MIN_WORDS or prose_share(text) < MIN_PROSE_SHARE:
                continue  # table debris, stray captions, formula blocks
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.id}#{len(chunks):04d}",
                    doc_id=doc.id,
                    jurisdiction=doc.jurisdiction,
                    section=locator,
                    section_title=title,
                    page_start=group[0][0],
                    page_end=group[-1][0],
                    text=text,
                    header=make_header(doc, locator, title),
                )
            )
    return chunks


def _group_paragraphs(paragraphs: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    """Pack paragraphs into ~TARGET_WORDS groups; split a single over-long paragraph by sentence."""
    groups: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    words = 0
    for page, paragraph in _split_long(paragraphs):
        n = len(paragraph.split())
        if current and words + n > TARGET_WORDS:
            groups.append(current)
            current, words = [], 0
        current.append((page, paragraph))
        words += n
    if current:
        groups.append(current)
    return groups


def _split_long(paragraphs: list[tuple[int, str]]) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for page, paragraph in paragraphs:
        if len(paragraph.split()) <= MAX_WORDS:
            out.append((page, paragraph))
            continue
        sentences = re.split(r"(?<=[.;])\s+(?=[A-Z(])", paragraph)
        buffer: list[str] = []
        for sentence in sentences:
            if buffer and len(" ".join([*buffer, sentence]).split()) > TARGET_WORDS:
                out.append((page, " ".join(buffer)))
                buffer = []
            buffer.append(sentence)
        if buffer:
            out.append((page, " ".join(buffer)))
    return out
