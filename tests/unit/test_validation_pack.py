"""The validation pack stays honest: every requirement is traced, every cited test exists."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "docs" / "validation"
TEST_REF = re.compile(r"`(tests/unit/test_\w+\.py)::(test_\w+)`")


def _ids(text: str) -> list[str]:
    return re.findall(r"^\| (URS-\d+) \|", text, flags=re.MULTILINE)


def _tests_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)}


def test_every_requirement_has_a_row_with_a_unit_test() -> None:
    required = _ids((PACK / "urs.md").read_text(encoding="utf-8"))
    matrix = (PACK / "traceability.md").read_text(encoding="utf-8")
    assert required and _ids(matrix) == required  # same requirements, same order
    for line in matrix.splitlines():
        if line.startswith("| URS-"):
            assert TEST_REF.search(line), f"no unit test linked: {line[:40]}"


def test_every_cited_test_exists() -> None:
    matrix = (PACK / "traceability.md").read_text(encoding="utf-8")
    cited = TEST_REF.findall(matrix)
    assert len(cited) > 30
    for file, name in cited:
        assert (ROOT / file).exists(), file
        assert name in _tests_in(ROOT / file), f"{file}::{name} not found"


def test_every_risk_points_at_known_requirements() -> None:
    required = set(_ids((PACK / "urs.md").read_text(encoding="utf-8")))
    risks = (PACK / "risk-assessment.md").read_text(encoding="utf-8")
    referenced = set(re.findall(r"URS-\d+", risks))
    assert referenced and referenced <= required
