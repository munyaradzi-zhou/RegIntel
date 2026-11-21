"""Drug label sections and recalls from openFDA, rendered as citable text.

The text returned here becomes evidence like a guidance passage: the writer quotes it and the
verifier checks every quote and number against it. Each result names its source exactly (label
set id, version and effective date; recall numbers), because labels and recalls change over time.

Drug names are reduced to letters, digits, spaces and hyphens before they go into the openFDA
query syntax, so a name cannot add search clauses of its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from regintel.openfda.client import OpenFDAClient

LABEL_ENDPOINT = "/drug/label.json"
ENFORCEMENT_ENDPOINT = "/drug/enforcement.json"
LABELS_CONSIDERED = 5
MAX_SECTION_CHARS = 6_000
MAX_RECALLS = 10
DEFAULT_RECALL_YEARS = 2

Section = Literal[
    "boxed_warning",
    "indications_and_usage",
    "dosage_and_administration",
    "contraindications",
    "warnings_and_cautions",
    "adverse_reactions",
    "drug_interactions",
    "use_in_specific_populations",
]
SECTION_TITLES: dict[str, str] = {
    "boxed_warning": "Boxed warning",
    "indications_and_usage": "Indications and usage",
    "dosage_and_administration": "Dosage and administration",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and precautions",
    "adverse_reactions": "Adverse reactions",
    "drug_interactions": "Drug interactions",
    "use_in_specific_populations": "Use in specific populations",
}
CAVEAT = "openFDA data is not for clinical decisions."
LABEL_CAVEAT = CAVEAT[:-1] + "; the label on DailyMed is authoritative."

_UNSAFE = re.compile(r"[^A-Za-z0-9 \-]")
_SPACE = re.compile(r"\s+")


def clean_name(drug: str) -> str:
    return _SPACE.sub(" ", _UNSAFE.sub(" ", drug)).strip()[:80]


def _name_clause(name: str, *, product_description: bool = False) -> str:
    fields = ["openfda.generic_name", "openfda.brand_name"] + (["product_description"] if product_description else [])
    return "(" + " OR ".join(f'{f}:"{name}"' for f in fields) + ")"


def _iso(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}" if len(yyyymmdd) == 8 else yyyymmdd


def _first(record: dict[str, Any], key: str) -> str:
    values = record.get("openfda", {}).get(key) or []
    return str(values[0]) if values else ""


@dataclass(frozen=True)
class LabelSection:
    drug: str
    section: str
    found: bool
    text: str = ""
    brand: str = ""
    generic: str = ""
    route: str = ""
    set_id: str = ""
    version: str = ""
    effective: str = ""  # YYYY-MM-DD
    truncated: bool = False
    labels_without_section: int = 0

    @property
    def record_id(self) -> str:
        return f"openfda:label:{self.set_id}:v{self.version}:{self.section}"

    @property
    def source_label(self) -> str:
        return f"FDA drug label via openFDA · {self.brand} ({self.generic}) · effective {self.effective}"

    def to_text(self) -> str:
        title = SECTION_TITLES.get(self.section, self.section)
        if not self.found:
            if self.labels_without_section:
                return (
                    f"openFDA has {self.labels_without_section} label(s) for '{self.drug}', but none has a "
                    f"{title.lower()} section."
                )
            return f"openFDA has no drug label matching '{self.drug}'."
        cut = " [Section truncated.]" if self.truncated else ""
        return (
            f"{title} from the FDA drug label for {self.brand} ({self.generic}, {self.route.lower()}), label set "
            f"{self.set_id}, version {self.version}, effective {self.effective}. Retrieved live from openFDA. "
            f"{LABEL_CAVEAT}\n{self.text}{cut}"
        )


async def label_section(client: OpenFDAClient, drug: str, section: Section) -> LabelSection:
    """The section from the most recently effective label for the drug (generic or brand name)."""
    name = clean_name(drug)
    if not name:
        return LabelSection(drug, section, False)
    page = await client.search(LABEL_ENDPOINT, _name_clause(name), limit=LABELS_CONSIDERED, sort="effective_time:desc")
    for label in page.results:
        parts = [p for p in label.get(section) or [] if isinstance(p, str) and p.strip()]
        if not parts:
            continue
        text = _SPACE.sub(" ", " ".join(parts)).strip()
        return LabelSection(
            drug,
            section,
            True,
            text[:MAX_SECTION_CHARS],
            brand=_first(label, "brand_name"),
            generic=_first(label, "generic_name"),
            route=_first(label, "route"),
            set_id=str(label.get("set_id", "")),
            version=str(label.get("version", "")),
            effective=_iso(str(label.get("effective_time", ""))),
            truncated=len(text) > MAX_SECTION_CHARS,
        )
    return LabelSection(drug, section, False, labels_without_section=len(page.results))


@dataclass(frozen=True)
class Recall:
    number: str
    report_date: str
    initiated: str
    classification: str
    status: str
    firm: str
    reason: str
    product: str

    def to_text(self) -> str:
        return (
            f"Recall {self.number}: {self.classification}, status {self.status}, reported {self.report_date} "
            f"(initiated {self.initiated}) by {self.firm}. Reason: {self.reason} Product: {self.product}"
        )


@dataclass(frozen=True)
class Recalls:
    drug: str
    since: str  # YYYY-MM-DD
    until: str
    total: int
    recalls: list[Recall]

    @property
    def record_id(self) -> str:
        return f"openfda:recalls:{self.drug.lower()}:{self.since}:{self.until}"

    @property
    def source_label(self) -> str:
        return f"FDA enforcement reports via openFDA · {self.drug} · reported {self.since} to {self.until}"

    def to_text(self) -> str:
        head = (
            f"FDA drug recalls (enforcement reports) for '{self.drug}' with a report date from {self.since} "
            f"to {self.until}, retrieved live from openFDA: {self.total} recall(s)."
        )
        if not self.recalls:
            return head
        shown = f" The {len(self.recalls)} most recent are listed." if self.total > len(self.recalls) else ""
        return head + shown + " " + CAVEAT + "\n" + "\n".join(r.to_text() for r in self.recalls)


def parse_day(value: str | None, default: date) -> date:
    if not value:
        return default
    return date.fromisoformat(value.strip())


async def recalls(client: OpenFDAClient, drug: str, since: str | None = None, until: str | None = None) -> Recalls:
    """Recalls whose report date falls in [since, until] (ISO dates; default the last two years)."""
    today = date.today()
    start = parse_day(since, today - timedelta(days=365 * DEFAULT_RECALL_YEARS))
    end = parse_day(until, today)
    name = clean_name(drug)
    if not name or start > end:
        return Recalls(drug, start.isoformat(), end.isoformat(), 0, [])
    window = f"report_date:[{start:%Y%m%d} TO {end:%Y%m%d}]"
    page = await client.search(
        ENFORCEMENT_ENDPOINT,
        f"{_name_clause(name, product_description=True)} AND {window}",
        limit=MAX_RECALLS,
        sort="report_date:desc",
    )
    items = [
        Recall(
            str(r.get("recall_number", "")),
            _iso(str(r.get("report_date", ""))),
            _iso(str(r.get("recall_initiation_date", ""))),
            str(r.get("classification", "")),
            str(r.get("status", "")),
            str(r.get("recalling_firm", "")),
            _SPACE.sub(" ", str(r.get("reason_for_recall", ""))).strip(),
            _SPACE.sub(" ", str(r.get("product_description", ""))).strip()[:300],
        )
        for r in page.results
    ]
    return Recalls(name, start.isoformat(), end.isoformat(), page.total, items)
