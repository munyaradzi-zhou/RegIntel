"""The CI regression gate: compare offline evaluation metrics and prompt versions with a baseline.

Everything here is deterministic and free: no model call, no Azure. ``tests/evaluation/ci_gate.py``
computes the metrics; this module decides what passes and renders the report.

Rules:
- A metric may not fall more than ``tolerance`` below its baseline (higher is better for all).
- A metric in the baseline must still be reported (a silently dropped metric is a failure).
- Verifier decisions on the recorded drafts must match exactly: tolerance does not apply.
- A prompt whose text changed must have a new version; a new version is reported, not failed,
  because it calls for the paid evaluations (``make eval-answers``) before merging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from regintel.answering.pipeline import verify_draft

if TYPE_CHECKING:
    from regintel.answering.schema import DraftAnswer
    from regintel.ingestion.chunking import Chunk

DEFAULT_TOLERANCE = 0.02
EXACT_PREFIX = "verifier."  # metrics that must match the baseline exactly

Status = Literal["ok", "improved", "regressed", "missing", "new"]


@dataclass(frozen=True)
class MetricRow:
    name: str
    baseline: float | None
    current: float | None
    status: Status

    @property
    def delta(self) -> float | None:
        if self.baseline is None or self.current is None:
            return None
        return round(self.current - self.baseline, 3)


def compare_metrics(
    baseline: dict[str, float], current: dict[str, float], tolerance: float = DEFAULT_TOLERANCE
) -> list[MetricRow]:
    rows: list[MetricRow] = []
    for name in sorted(set(baseline) | set(current)):
        old, new = baseline.get(name), current.get(name)
        if old is None:
            rows.append(MetricRow(name, None, new, "new"))
        elif new is None:
            rows.append(MetricRow(name, old, None, "missing"))
        else:
            allowed = 0.0 if name.startswith(EXACT_PREFIX) else tolerance
            if new < old - allowed - 1e-9:
                status: Status = "regressed"
            elif new > old + 1e-9:
                status = "improved"
            else:
                status = "ok"
            rows.append(MetricRow(name, old, new, status))
    return rows


@dataclass(frozen=True)
class PromptRow:
    name: str
    baseline: str | None  # "v5+1a2b…"
    current: str | None
    status: Literal["ok", "bumped", "unversioned_change", "new", "missing"]


def compare_prompts(baseline: dict[str, dict[str, str]], current: dict[str, dict[str, str]]) -> list[PromptRow]:
    def tag(p: dict[str, str] | None) -> str | None:
        return f"{p['version']}+{p['sha256']}" if p else None

    rows: list[PromptRow] = []
    for name in sorted(set(baseline) | set(current)):
        old, new = baseline.get(name), current.get(name)
        if old is None:
            rows.append(PromptRow(name, None, tag(new), "new"))
        elif new is None:
            rows.append(PromptRow(name, tag(old), None, "missing"))
        elif old["sha256"] == new["sha256"]:
            rows.append(PromptRow(name, tag(old), tag(new), "ok"))
        elif old["version"] == new["version"]:
            rows.append(PromptRow(name, tag(old), tag(new), "unversioned_change"))
        else:
            rows.append(PromptRow(name, tag(old), tag(new), "bumped"))
    return rows


@dataclass
class VerifierCase:
    id: str
    kind: str  # "recorded" or the mutation applied, e.g. "misquote"
    question: str
    passages: list[str]  # chunk ids, in passage order
    draft: DraftAnswer
    expected_verified: list[bool]
    expected_label: str


@dataclass
class VerifierOutcome:
    cases: int = 0
    statements: int = 0
    matching_statements: int = 0
    matching_labels: int = 0
    kept: int = 0
    dropped: int = 0
    mismatches: list[str] = field(default_factory=list)

    def metrics(self) -> dict[str, float]:
        return {
            "verifier.statement_decisions_match": round(self.matching_statements / self.statements, 3)
            if self.statements
            else 1.0,
            "verifier.labels_match": round(self.matching_labels / self.cases, 3) if self.cases else 1.0,
        }


def run_verifier_cases(cases: list[VerifierCase], chunks: dict[str, Chunk]) -> VerifierOutcome:
    """Re-run the verifier on recorded and mutated drafts; every keep/drop must match the recording."""
    outcome = VerifierOutcome()
    for case in cases:
        passages = [chunks[cid] for cid in case.passages]
        result = verify_draft(case.question, case.draft, passages, manifest=None)
        got = [s.verified for s in result.statements]
        outcome.cases += 1
        outcome.statements += len(case.expected_verified)
        outcome.kept += sum(got)
        outcome.dropped += len(got) - sum(got)
        same = sum(a == b for a, b in zip(got, case.expected_verified, strict=False))
        outcome.matching_statements += same if len(got) == len(case.expected_verified) else 0
        if result.label == case.expected_label:
            outcome.matching_labels += 1
        if got != case.expected_verified or result.label != case.expected_label:
            outcome.mismatches.append(
                f"{case.id} ({case.kind}): expected {case.expected_verified} / {case.expected_label}, "
                f"got {got} / {result.label}"
            )
    return outcome


def passed(metric_rows: list[MetricRow], prompt_rows: list[PromptRow]) -> bool:
    return not any(r.status in ("regressed", "missing") for r in metric_rows) and not any(
        r.status in ("unversioned_change", "missing") for r in prompt_rows
    )


_ICON = {"ok": "✅", "improved": "⬆️", "regressed": "❌", "missing": "❌", "new": "🆕",
         "bumped": "🔁", "unversioned_change": "❌"}  # fmt: skip


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def render_markdown(
    metric_rows: list[MetricRow],
    prompt_rows: list[PromptRow],
    tolerance: float,
    notes: list[str] | None = None,
    mismatches: list[str] | None = None,
) -> str:
    ok = passed(metric_rows, prompt_rows)
    lines = [
        "## Evaluation gate " + ("✅ passed" if ok else "❌ failed"),
        "",
        f"Offline, no model calls. A metric fails when it drops more than {tolerance:.2f} below the baseline; "
        "`verifier.*` must match exactly.",
        "",
        "| Metric | Baseline | This change | Δ | |",
        "|---|---|---|---|---|",
    ]
    for r in metric_rows:
        delta = "" if r.delta is None else f"{r.delta:+.3f}"
        lines.append(f"| `{r.name}` | {_fmt(r.baseline)} | {_fmt(r.current)} | {delta} | {_ICON[r.status]} |")
    lines += ["", "| Prompt | Baseline | This change | |", "|---|---|---|---|"]
    for p in prompt_rows:
        lines.append(f"| {p.name} | `{p.baseline or '—'}` | `{p.current or '—'}` | {_ICON[p.status]} |")
    if any(p.status == "unversioned_change" for p in prompt_rows):
        lines += ["", "❌ A prompt's text changed but its version did not. Bump the version constant next to it."]
    if any(p.status == "bumped" for p in prompt_rows):
        lines += ["", "🔁 A prompt has a new version: run the paid evaluations (`make eval-answers`, "
                  "`make eval-agent`) and record the results before merging."]  # fmt: skip
    if mismatches:
        lines += ["", "<details><summary>Verifier decisions that changed</summary>", ""]
        lines += [f"- {m}" for m in mismatches]
        lines += ["", "</details>"]
    if notes:
        lines += [""] + [f"> {n}" for n in notes]
    return "\n".join(lines) + "\n"
