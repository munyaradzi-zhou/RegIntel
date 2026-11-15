"""Answer metrics against the golden set: facts stated, right sources cited, unanswerables declined.

Per question:
- answered: the label is grounded or partially_grounded (something verified is shown);
- fact_recall: share of the expected facts found in the shown answer (normalised text);
- cited_expected: a shown citation sits in an expected document and section;
- cited_all_docs: every expected document is cited (comparison and multi-hop questions);
- statements drafted vs verified: how often the model's own citations survive verification.

For unanswerable questions the right outcome is not answering (declined or withheld).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from regintel.answering.verify import contains_number, normalise
from regintel.evaluation.retrieval_metrics import section_matches

if TYPE_CHECKING:
    from regintel.answering.schema import AnswerResult
    from regintel.evaluation.retrieval_metrics import GoldenQuestion

ANSWERED = ("grounded", "partially_grounded")
_NUMERIC = re.compile(r"\d+(?:[.,]\d+)*%?")


def fact_found(fact: str, answer: str) -> bool:
    """A numeric fact must appear as a whole number ("12" is not found in "120"); a wording fact
    is matched ignoring case, hyphens and spacing ("half-life" = "half life")."""
    if _NUMERIC.fullmatch(fact.strip()):
        return contains_number(fact.strip().rstrip("%"), answer)
    return normalise(fact).replace(" ", "") in normalise(answer).replace(" ", "")


@dataclass
class AnswerScore:
    id: str
    type: str
    label: str
    answered: bool
    fact_recall: float | None  # None for unanswerable questions
    facts_missing: list[str]
    cited_expected: bool
    cited_all_docs: bool
    statements: int
    verified: int
    problems: list[str] = field(default_factory=list)
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    generation_seconds: float = 0.0
    answer: str = ""
    retrieved: list[str] = field(default_factory=list)
    draft: dict[str, object] | None = None


def score_answer(question: GoldenQuestion, result: AnswerResult) -> AnswerScore:
    answered = result.label in ANSWERED
    text = result.text if answered else ""
    missing = [f for f in question.facts if not fact_found(f, text)]
    cited = [c for s in result.shown for c in s.citations]
    cited_expected = any(
        c.doc_id == e.doc_id and (not e.sections or any(section_matches(c.section, s) for s in e.sections))
        for c in cited
        for e in question.expected
    )
    cited_docs = {c.doc_id for c in cited}
    return AnswerScore(
        id=question.id,
        type=question.type,
        label=result.label,
        answered=answered,
        fact_recall=None if question.type == "unanswerable" else (1 - len(missing) / len(question.facts)),
        facts_missing=missing,
        cited_expected=cited_expected,
        cited_all_docs=bool(question.expected) and all(e.doc_id in cited_docs for e in question.expected),
        statements=len(result.statements),
        verified=sum(s.verified for s in result.statements),
        problems=[p for s in result.statements for p in s.problems],
        input_tokens=result.usage.input_tokens,
        cached_tokens=result.usage.cached_tokens,
        output_tokens=result.usage.output_tokens,
        reasoning_tokens=result.usage.reasoning_tokens,
        generation_seconds=round(result.generation_seconds, 2),
        answer=result.text,
        retrieved=result.retrieved,
        draft=result.draft.model_dump() if result.draft else None,
    )


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _problem_kind(problem: str) -> str:
    for kind in ("quote not found", "numbers not in", "not retrieved", "no citation"):
        if problem.startswith(kind) or kind in problem:
            return kind
    return "other"


def summarise_answers(scores: list[AnswerScore]) -> dict[str, object]:
    answerable = [s for s in scores if s.type != "unanswerable"]
    unanswerable = [s for s in scores if s.type == "unanswerable"]
    multi = [s for s in answerable if s.type in ("comparison", "multihop")]
    drafted = sum(s.statements for s in scores)
    return {
        "questions": len(scores),
        "answer_rate": _mean([float(s.answered) for s in answerable]),
        "abstention_rate": _mean([float(not s.answered) for s in unanswerable]),
        "fact_recall": _mean([s.fact_recall or 0.0 for s in answerable]),
        "all_facts": _mean([float(s.fact_recall == 1.0) for s in answerable]),
        "cited_expected_source": _mean([float(s.cited_expected) for s in answerable]),
        "cited_all_docs_multi": _mean([float(s.cited_all_docs) for s in multi]),
        "statements_verified": round(sum(s.verified for s in scores) / drafted, 3) if drafted else 0.0,
        "labels": dict(Counter(s.label for s in scores)),
        "dropped_by_reason": dict(Counter(_problem_kind(p) for s in scores for p in s.problems)),
        "mean_generation_seconds": _mean([s.generation_seconds for s in scores]),
        "tokens_per_question": {
            "input": sum(s.input_tokens for s in scores) // max(1, len(scores)),
            "cached": sum(s.cached_tokens for s in scores) // max(1, len(scores)),
            "output": sum(s.output_tokens for s in scores) // max(1, len(scores)),
            "reasoning": sum(s.reasoning_tokens for s in scores) // max(1, len(scores)),
        },
    }
