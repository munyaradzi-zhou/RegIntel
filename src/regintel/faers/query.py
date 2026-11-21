"""FAERS questions answered by computation, not generation.

The agent never writes SQL. It calls these methods with a drug name (and a reaction), and they
build DuckDB expressions from typed values, so there is no injection surface and every number is
computed from the Parquet tables built by `faers.build`. Results render as plain sentences; the
answer verifier then requires each number in an answer to appear in the cited tool output.

Scope, stated in every result: the openFDA FAERS 2025 Q1 extract loaded here (parts 1-3), not all
of FAERS. Disproportionality is a screening statistic, not evidence that a drug causes a reaction.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import duckdb
from duckdb import ColumnExpression as Col
from duckdb import ConstantExpression as Const
from duckdb import FunctionExpression as Fn

if TYPE_CHECKING:
    from pathlib import Path

SCOPE = "openFDA FAERS extract, 2025 Q1 (parts 1-3)"
CAVEAT = "Disproportionality is a signal-screening statistic; it does not show that the drug causes the reaction."


def _n(value: int) -> str:
    return f"{value:,}"


def _ranked(counts: list[tuple[str, int]]) -> list[tuple[int, bool, str, int]]:
    """Competition ranking (1, 2, 3, 3, 5): equal counts share a rank and are marked as tied."""
    values = [n for _, n in counts]
    return [(1 + sum(v > n for v in values), values.count(n) > 1, name, n) for name, n in counts]


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


@dataclass(frozen=True)
class DrugSummary:
    drug: str
    matched_names: list[str]
    total_reports: int
    reports: int
    serious: int
    deaths: int
    top_reactions: list[tuple[str, int]]

    def to_text(self) -> str:
        if not self.reports:
            return f"{SCOPE}: no reports list {self.drug.upper()} as a suspect drug."
        # Each reaction is a self-contained phrase with its rank, so a quote of one entry still says
        # it is among the most frequent (a bare "Death 48" is too short to quote, and the reviewer
        # rightly calls "the second most frequent" overreach when the quote omits the ranking).
        reactions = " ".join(
            f"Most frequent reaction {rank}{' (tied)' if tied else ''}: {name}, {_n(n)} reports."
            for rank, tied, name, n in _ranked(self.top_reactions)
        )
        return (
            f"{SCOPE}, {_n(self.total_reports)} reports in total. "
            f"Reports listing {self.drug.upper()} as a suspect drug: {_n(self.reports)} "
            f"(matched names: {', '.join(self.matched_names)}). "
            f"Serious: {_n(self.serious)} ({_pct(self.serious, self.reports)}). "
            f"Death reported: {_n(self.deaths)} ({_pct(self.deaths, self.reports)}). "
            f"{reactions}"
        )


@dataclass(frozen=True)
class Disproportionality:
    drug: str
    reaction: str
    a: int  # drug and reaction
    b: int  # drug, other reactions
    c: int  # other drugs, the reaction
    d: int  # neither

    @property
    def ror(self) -> float | None:
        if 0 in (self.a, self.b, self.c, self.d):
            return None
        return (self.a / self.b) / (self.c / self.d)

    @property
    def ci95(self) -> tuple[float, float] | None:
        ror = self.ror
        if ror is None:
            return None
        se = math.sqrt(1 / self.a + 1 / self.b + 1 / self.c + 1 / self.d)
        return math.exp(math.log(ror) - 1.96 * se), math.exp(math.log(ror) + 1.96 * se)

    def to_text(self) -> str:
        counts = (
            f"{SCOPE}. Reports with {self.drug.upper()} as a suspect drug and the reaction "
            f"'{self.reaction}': {_n(self.a)}; with {self.drug.upper()} but not the reaction: {_n(self.b)}; "
            f"with the reaction but not {self.drug.upper()}: {_n(self.c)}; with neither: {_n(self.d)}."
        )
        if self.ror is None or self.ci95 is None:
            return f"{counts} The reporting odds ratio cannot be computed because a cell is zero."
        low, high = self.ci95
        # The ratio sentence names drug and reaction itself, so a quote of it alone is self-contained.
        return (
            f"{counts} Reporting odds ratio for {self.drug.upper()} and '{self.reaction}': {self.ror:.2f} "
            f"(95% CI {low:.2f} to {high:.2f}). {CAVEAT}"
        )


class FaersStore:
    def __init__(self, faers_dir: Path) -> None:
        self.con = duckdb.connect()
        self.reports = self.con.read_parquet(str(faers_dir / "reports.parquet"))
        self.drugs = self.con.read_parquet(str(faers_dir / "drugs.parquet"))
        self.reactions = self.con.read_parquet(str(faers_dir / "reactions.parquet"))
        self.total_reports = int(self.reports.shape[0])
        # One DuckDB connection is not safe for concurrent queries, and the agent runs tools in parallel.
        self._lock = threading.Lock()

    def _suspect_reports(self, drug: str) -> tuple[set[int], list[str]]:
        """Reports listing the drug as suspect: exact match on active substance, generic or product
        name, or an active substance that starts with it ('AMLODIPINE' -> 'AMLODIPINE BESYLATE')."""
        name = Const(drug.strip().upper())
        substance = Fn("upper", Col("active_substance"))
        matches = (
            (substance == name)
            | (Fn("upper", Col("generic_name")) == name)
            | (Fn("upper", Col("medicinalproduct")) == name)
            | Fn("starts_with", substance, Fn("concat", name, Const(" ")))
        )
        rows = self.drugs.filter(matches & (Col("role") == Const("suspect"))).select(
            Col("safetyreportid"), Col("active_substance")
        )
        result = rows.fetchall()
        return {int(r[0]) for r in result}, sorted({str(r[1]) for r in result if r[1]})[:5]

    def _ids(self, relation: duckdb.DuckDBPyRelation) -> set[int]:
        return {int(r[0]) for r in relation.select(Col("safetyreportid")).distinct().fetchall()}

    def drug_summary(self, drug: str, top: int = 5) -> DrugSummary:
        with self._lock:
            return self._drug_summary(drug, top)

    def disproportionality(self, drug: str, reaction: str) -> Disproportionality:
        with self._lock:
            return self._disproportionality(drug, reaction)

    def _drug_summary(self, drug: str, top: int) -> DrugSummary:
        ids, names = self._suspect_reports(drug)
        if not ids:
            return DrugSummary(drug, [], self.total_reports, 0, 0, 0, [])
        in_ids = Col("safetyreportid").isin(*[Const(i) for i in ids])
        flags = self.reports.filter(in_ids).select(Col("serious"), Col("death")).fetchall()
        counts: dict[str, int] = {}
        for (reaction,) in self.reactions.filter(in_ids).select(Col("reaction")).fetchall():
            if reaction:
                counts[str(reaction)] = counts.get(str(reaction), 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
        return DrugSummary(
            drug,
            names,
            self.total_reports,
            len(ids),
            sum(1 for serious, _ in flags if serious),
            sum(1 for _, death in flags if death),
            ranked,
        )

    def _disproportionality(self, drug: str, reaction: str) -> Disproportionality:
        drug_ids, _ = self._suspect_reports(drug)
        reaction_ids = self._ids(self.reactions.filter(Fn("lower", Col("reaction")) == Const(reaction.strip().lower())))
        a = len(drug_ids & reaction_ids)
        b = len(drug_ids) - a
        c = len(reaction_ids) - a
        return Disproportionality(drug, reaction, a, b, c, self.total_reports - a - b - c)
