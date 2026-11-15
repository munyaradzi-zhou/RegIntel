import math
from typing import TYPE_CHECKING

import duckdb
import pytest

from regintel.faers.query import Disproportionality, FaersStore

if TYPE_CHECKING:
    from pathlib import Path

# 8 reports. DRUGA suspect in 1-4 (as "DRUGA HCL" in 4), concomitant in 5. Rash in 1, 2, 5, 6.
REPORTS = [(i, i <= 2, i == 1) for i in range(1, 9)]  # (id, serious, death)
DRUGS = [
    (1, "DRUGA", "suspect"),
    (2, "DRUGA", "suspect"),
    (3, "DRUGA", "suspect"),
    (4, "DRUGA HCL", "suspect"),
    (5, "DRUGA", "concomitant"),
    (6, "OTHER", "suspect"),
    (7, "OTHER", "suspect"),
    (8, "OTHER", "suspect"),
]
REACTIONS = [
    (1, "Rash"),
    (1, "Nausea"),
    (2, "Rash"),
    (3, "Nausea"),
    (4, "Headache"),
    (5, "Rash"),
    (6, "Rash"),
    (7, "Nausea"),
    (8, "Headache"),
]


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> FaersStore:
    root: Path = tmp_path_factory.mktemp("faers")
    con = duckdb.connect()
    tables = {
        "reports": ("safetyreportid INTEGER, serious BOOLEAN, death BOOLEAN", REPORTS),
        "drugs": ("safetyreportid INTEGER, active_substance VARCHAR, role VARCHAR", DRUGS),
        "reactions": ("safetyreportid INTEGER, reaction VARCHAR", REACTIONS),
    }
    for name, (schema, rows) in tables.items():
        con.execute(f"CREATE TABLE {name} ({schema})")  # test fixture, fixed strings
        con.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' * len(rows[0]))})", rows)
        if name == "drugs":
            con.execute("ALTER TABLE drugs ADD COLUMN generic_name VARCHAR")
            con.execute("ALTER TABLE drugs ADD COLUMN medicinalproduct VARCHAR")
        con.table(name).write_parquet(str(root / f"{name}.parquet"))
    return FaersStore(root)


def test_drug_summary_counts_suspect_reports_only(store: FaersStore) -> None:
    summary = store.drug_summary("druga")
    assert (summary.reports, summary.serious, summary.deaths) == (4, 2, 1)  # report 5 is concomitant
    assert summary.matched_names == ["DRUGA", "DRUGA HCL"]  # salt form matched by prefix
    assert summary.top_reactions[:2] == [("Nausea", 2), ("Rash", 2)]
    text = summary.to_text()
    assert "8 reports in total" in text and "as a suspect drug: 4" in text and "Serious: 2 (50.0%)" in text


def test_unknown_drug(store: FaersStore) -> None:
    assert "no reports list NOPE" in store.drug_summary("nope").to_text()


def test_disproportionality_two_by_two(store: FaersStore) -> None:
    result = store.disproportionality("DRUGA", "rash")
    assert (result.a, result.b, result.c, result.d) == (2, 2, 2, 2)  # rash: reports 1, 2 | 5, 6
    assert result.ror == 1.0
    assert (
        "cannot be computed" not in result.to_text()
        and "Reporting odds ratio for DRUGA and 'rash': 1.00" in result.to_text()
    )


def test_ror_and_confidence_interval() -> None:
    result = Disproportionality("x", "y", a=20, b=80, c=100, d=9800)
    assert result.ror == pytest.approx((20 / 80) / (100 / 9800))
    low, high = result.ci95 or (0, 0)
    se = math.sqrt(1 / 20 + 1 / 80 + 1 / 100 + 1 / 9800)
    assert low == pytest.approx(result.ror * math.exp(-1.96 * se))
    assert high == pytest.approx(result.ror * math.exp(1.96 * se))
    assert Disproportionality("x", "y", 0, 5, 5, 5).ror is None


def test_parallel_queries_share_one_connection_safely(store: FaersStore) -> None:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: store.disproportionality("DRUGA", "rash").a, range(32)))
    assert results == [2] * 32
