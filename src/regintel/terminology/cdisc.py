"""CDISC Controlled Terminology lookup.

Terminology questions ("what does AESER mean?", "which codelist holds 'MILD'?") need exact
answers, so they are served from the NCI EVS tab-delimited files rather than from text search.
A term is matched on its submission value, synonyms or preferred term, exactly first, then
case-insensitively; the codelist a term belongs to is returned with it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

STANDARDS = ("sdtm", "adam", "send", "ddf")


@dataclass(frozen=True)
class Term:
    standard: str  # sdtm, adam, send, ddf, glossary
    code: str  # NCI C-code, e.g. C41338
    codelist_code: str  # empty for a codelist row itself
    codelist_name: str
    submission_value: str
    synonyms: tuple[str, ...]
    definition: str
    preferred_term: str

    @property
    def is_codelist(self) -> bool:
        return not self.codelist_code


class Terminology:
    def __init__(self, terms: list[Term]) -> None:
        self.terms = terms
        self._by_key: dict[str, list[Term]] = {}
        for term in terms:
            for key in {k.casefold() for k in (term.submission_value, term.preferred_term, *term.synonyms) if k}:
                self._by_key.setdefault(key, []).append(term)

    def lookup(self, query: str, standard: str | None = None, limit: int = 10) -> list[Term]:
        """Exact matches first (case-sensitive submission value), then case-insensitive matches."""
        hits = self._by_key.get(query.strip().casefold(), [])
        if standard:
            hits = [t for t in hits if t.standard == standard]
        exact = [t for t in hits if t.submission_value == query.strip()]
        rest = [t for t in hits if t not in exact]
        return (exact + rest)[:limit]

    def codelist(self, code: str) -> list[Term]:
        return [t for t in self.terms if t.codelist_code == code]

    def __len__(self) -> int:
        return len(self.terms)


def _parse_tsv(path: Path, standard: str) -> list[Term]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return [
            Term(
                standard=standard,
                code=row["Code"].strip(),
                codelist_code=(row.get("Codelist Code") or "").strip(),
                codelist_name=(row.get("Codelist Name") or "").strip(),
                submission_value=(row.get("CDISC Submission Value") or "").strip(),
                synonyms=tuple(s.strip() for s in (row.get("CDISC Synonym(s)") or "").split(";") if s.strip()),
                definition=(row.get("CDISC Definition") or "").strip(),
                preferred_term=(row.get("NCI Preferred Term") or "").strip(),
            )
            for row in reader
        ]


def load_terminology(cdisc_dir: Path) -> Terminology:
    terms: list[Term] = []
    for standard in (*STANDARDS, "glossary"):
        path = cdisc_dir / f"{standard}.tsv"
        if path.exists():
            terms.extend(_parse_tsv(path, standard))
    if not terms:
        raise FileNotFoundError(f"no CDISC terminology in {cdisc_dir}: run `make corpus` first")
    return Terminology(terms)
