from pathlib import Path

import pytest

from regintel.terminology.cdisc import load_terminology

COLUMNS = (
    "Code\tCodelist Code\tCodelist Extensible (Yes/No)\tCodelist Name\tCDISC Submission Value\t"
    "CDISC Synonym(s)\tCDISC Definition\tNCI Preferred Term\n"
)
SDTM = (
    COLUMNS
    + "C66769\t\tNo\tSeverity/Intensity Scale for Adverse Events\tAESEV\tSeverity\tA scale.\tCDISC SDTM Severity\n"
    "C41338\tC66769\t\tSeverity/Intensity Scale for Adverse Events\tMILD\tMild; 1\t"
    "Slight in degree.\tMild Adverse Event\n"
)
GLOSSARY = (
    COLUMNS
    + "C188813\tC67497\t\tCDISC Glossary\testimand\t\tA precise description of the treatment effect.\tEstimand\n"
)


@pytest.fixture
def cdisc_dir(tmp_path: Path) -> Path:
    (tmp_path / "sdtm.tsv").write_text(SDTM, encoding="utf-8")
    (tmp_path / "glossary.tsv").write_text(GLOSSARY, encoding="utf-8")
    return tmp_path


def test_exact_and_case_insensitive_lookup(cdisc_dir: Path) -> None:
    terms = load_terminology(cdisc_dir)
    hits = terms.lookup("MILD")
    assert len(hits) == 1  # "MILD" and "Mild" fold to one key, not two hits
    assert hits[0].code == "C41338" and hits[0].codelist_name.startswith("Severity")
    assert terms.lookup("mild adverse event")[0].code == "C41338"


def test_glossary_and_standard_filter(cdisc_dir: Path) -> None:
    terms = load_terminology(cdisc_dir)
    assert terms.lookup("Estimand")[0].standard == "glossary"
    assert terms.lookup("estimand", standard="sdtm") == []


def test_codelist_members_and_codelist_rows(cdisc_dir: Path) -> None:
    terms = load_terminology(cdisc_dir)
    assert [t.submission_value for t in terms.codelist("C66769")] == ["MILD"]
    assert terms.lookup("AESEV")[0].is_codelist


def test_unknown_term_returns_nothing(cdisc_dir: Path) -> None:
    assert load_terminology(cdisc_dir).lookup("AESER") == []  # a variable name, not controlled terminology


def test_missing_files_fail_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make corpus"):
        load_terminology(tmp_path)
