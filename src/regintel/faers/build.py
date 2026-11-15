"""Build the FAERS safety tables: `python -m regintel.faers.build`.

openFDA ships each partition as one ~900 MB JSON object, too large to load or for DuckDB to
read in one piece, so reports are streamed with ijson and flattened into three tables:

    reports    one row per safety report (dates, seriousness, patient age and sex, country)
    drugs      one row per drug on a report (role: suspect / concomitant / interacting)
    reactions  one row per MedDRA preferred term on a report (with outcome)

Coded fields are decoded here (age units to years, outcome codes to words) so that the
query tool and the model never have to know openFDA's codes.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Any

import duckdb
import ijson
import structlog

from regintel.config import get_settings

log = structlog.get_logger()

AGE_UNIT_TO_YEARS = {"800": 10.0, "801": 1.0, "802": 1 / 12, "803": 1 / 52.1775, "804": 1 / 365.25, "805": 1 / 8766}
SEX = {"0": "unknown", "1": "male", "2": "female"}
DRUG_ROLE = {"1": "suspect", "2": "concomitant", "3": "interacting", "4": "not administered"}
OUTCOME = {
    "1": "recovered",
    "2": "recovering",
    "3": "not recovered",
    "4": "recovered with sequelae",
    "5": "fatal",
    "6": "unknown",
}
REPORTER = {"1": "physician", "2": "pharmacist", "3": "other health professional", "4": "lawyer", "5": "consumer"}

REPORT_COLUMNS = [
    "safetyreportid", "receivedate", "serious", "death", "life_threatening", "hospitalization",
    "disabling", "congenital_anomaly", "other_serious", "occurcountry", "reporter", "sex", "age_years",
]  # fmt: skip
DRUG_COLUMNS = [
    "safetyreportid",
    "drug_seq",
    "medicinalproduct",
    "active_substance",
    "generic_name",
    "brand_name",
    "role",
]
REACTION_COLUMNS = ["safetyreportid", "reaction", "outcome"]


def _flag(value: str | None) -> bool:
    return value == "1"


def _age_years(patient: dict[str, Any]) -> float | None:
    age, unit = patient.get("patientonsetage"), patient.get("patientonsetageunit")
    if age is None or unit not in AGE_UNIT_TO_YEARS:
        return None
    try:
        years = float(age) * AGE_UNIT_TO_YEARS[unit]
    except ValueError:
        return None
    return round(years, 2) if 0 <= years < 130 else None


def _date(value: str | None) -> str | None:
    """openFDA dates are YYYYMMDD strings."""
    if not value or len(value) != 8 or not value.isdigit():
        return None
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def flatten_report(report: dict[str, Any]) -> tuple[list[Any], list[list[Any]], list[list[Any]]]:
    rid = report["safetyreportid"]
    patient = report.get("patient") or {}
    source = report.get("primarysource") or {}
    row = [
        rid,
        _date(report.get("receivedate")),
        _flag(report.get("serious")),
        _flag(report.get("seriousnessdeath")),
        _flag(report.get("seriousnesslifethreatening")),
        _flag(report.get("seriousnesshospitalization")),
        _flag(report.get("seriousnessdisabling")),
        _flag(report.get("seriousnesscongenitalanomali")),
        _flag(report.get("seriousnessother")),
        report.get("occurcountry"),
        REPORTER.get(source.get("qualification") or ""),
        SEX.get(patient.get("patientsex") or "", "unknown"),
        _age_years(patient),
    ]
    drugs = []
    for seq, drug in enumerate(patient.get("drug") or []):
        openfda = drug.get("openfda") or {}
        drugs.append([
            rid,
            seq,
            (drug.get("medicinalproduct") or "").upper().strip() or None,
            ((drug.get("activesubstance") or {}).get("activesubstancename") or "").upper().strip() or None,
            "; ".join(openfda.get("generic_name") or []).upper() or None,
            "; ".join(openfda.get("brand_name") or []).upper() or None,
            DRUG_ROLE.get(drug.get("drugcharacterization") or ""),
        ])  # fmt: skip
    reactions = [
        [rid, r.get("reactionmeddrapt"), OUTCOME.get(r.get("reactionoutcome") or "", "unknown")]
        for r in patient.get("reaction") or []
        if r.get("reactionmeddrapt")
    ]
    return row, drugs, reactions


def build(raw_dir: Path, out_dir: Path) -> dict[str, int]:
    files = sorted(raw_dir.glob("drug-event-*.json"))
    if not files:
        raise FileNotFoundError(f"no FAERS files in {raw_dir}: run `make corpus` first")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        paths = {name: Path(tmp) / f"{name}.tsv" for name in ("reports", "drugs", "reactions")}
        handles = {name: path.open("w", encoding="utf-8", newline="") for name, path in paths.items()}
        writers = {name: csv.writer(fh, delimiter="\t") for name, fh in handles.items()}
        writers["reports"].writerow(REPORT_COLUMNS)
        writers["drugs"].writerow(DRUG_COLUMNS)
        writers["reactions"].writerow(REACTION_COLUMNS)
        seen: set[str] = set()
        for path in files:
            with path.open("rb") as source:
                for report in ijson.items(source, "results.item"):
                    if report["safetyreportid"] in seen:  # a report can recur across partitions
                        continue
                    seen.add(report["safetyreportid"])
                    row, drugs, reactions = flatten_report(report)
                    writers["reports"].writerow(row)
                    writers["drugs"].writerows(drugs)
                    writers["reactions"].writerows(reactions)
            log.info("streamed", file=path.name, reports_so_far=len(seen))
        for handle in handles.values():
            handle.close()

        counts: dict[str, int] = {}
        con = duckdb.connect()
        for name, path in paths.items():
            target = out_dir / f"{name}.parquet"
            table = con.read_csv(path.as_posix(), delimiter="\t", header=True, quotechar='"')
            table.write_parquet(target.as_posix(), compression="zstd")
            counts[name] = con.read_parquet(target.as_posix()).shape[0]
    return counts


def main() -> None:
    data_dir = get_settings().data_dir
    counts = build(data_dir / "raw" / "faers", data_dir / "processed" / "faers")
    log.info("done", **counts)


if __name__ == "__main__":
    main()
