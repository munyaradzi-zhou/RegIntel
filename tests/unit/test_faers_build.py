import json
from pathlib import Path

import duckdb
import pytest

from regintel.faers.build import _age_years, build, flatten_report

REPORT = {
    "safetyreportid": "100",
    "receivedate": "20250331",
    "serious": "1",
    "seriousnessdeath": "1",
    "seriousnesshospitalization": "2",
    "occurcountry": "CA",
    "primarysource": {"qualification": "1"},
    "patient": {
        "patientonsetage": "37",
        "patientonsetageunit": "801",
        "patientsex": "2",
        "drug": [
            {
                "medicinalproduct": "fentanyl",
                "drugcharacterization": "1",
                "activesubstance": {"activesubstancename": "FENTANYL"},
                "openfda": {"generic_name": ["FENTANYL"], "brand_name": ["DURAGESIC"]},
            },
            {"medicinalproduct": "Paracetamol", "drugcharacterization": "2"},
        ],
        "reaction": [
            {"reactionmeddrapt": "Toxicity to various agents", "reactionoutcome": "5"},
            {"reactionoutcome": "1"},
        ],
    },
}


def test_flatten_decodes_codes() -> None:
    row, drugs, reactions = flatten_report(REPORT)
    assert row == [
        "100",
        "2025-03-31",
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        "CA",
        "physician",
        "female",
        37.0,
    ]
    assert drugs[0] == ["100", 0, "FENTANYL", "FENTANYL", "FENTANYL", "DURAGESIC", "suspect"]
    assert drugs[1][2] == "PARACETAMOL" and drugs[1][6] == "concomitant" and drugs[1][4] is None
    assert reactions == [["100", "Toxicity to various agents", "fatal"]]  # reaction without a term is dropped


@pytest.mark.parametrize(
    ("age", "unit", "years"),
    [
        ("6", "802", 0.5),
        ("3", "800", 30.0),
        ("14", "804", 0.04),
        ("200", "801", None),
        ("x", "801", None),
        ("5", "999", None),
    ],
)
def test_age_normalised_to_years(age: str, unit: str, years: float | None) -> None:
    assert _age_years({"patientonsetage": age, "patientonsetageunit": unit}) == years


def test_build_writes_parquet_and_skips_duplicates(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    duplicate = dict(REPORT)
    (raw / "drug-event-0001-of-0002.json").write_text(json.dumps({"results": [REPORT]}))
    (raw / "drug-event-0002-of-0002.json").write_text(json.dumps({"results": [duplicate]}))
    counts = build(raw, tmp_path / "out")
    assert counts == {"reports": 1, "drugs": 2, "reactions": 1}
    parquet = (tmp_path / "out" / "reports.parquet").as_posix()
    sex = duckdb.connect().execute(f"SELECT sex FROM '{parquet}'").fetchone()
    assert sex == ("female",)


def test_build_without_files_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make corpus"):
        build(tmp_path, tmp_path / "out")
