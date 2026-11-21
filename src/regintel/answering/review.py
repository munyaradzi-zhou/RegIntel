"""Answer review: does each verified statement say only what its quotes say?

The verifier proves quotes are real and numbers match. It cannot tell when a statement stretches a
real quote: "Annex 11 applies to all computerised systems" quoted, "so it sets requirements for
machine learning models" stated. The reviewer is a second model, from a different family than the
writer (gpt-oss-120b vs gpt-5-mini), so it does not share the writer's blind spots. It sees only the
question and each statement with its verified quotes, never the writer's reasoning.

Verdicts per statement: supported / overreach / unsupported. Anything but supported is dropped,
like a failed quote. Separately, if the question asks about a specific subject that none of the
quotes mention, the answer becomes a decline that names the missing subject.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from regintel.answering.pipeline import NOT_GROUNDED_MESSAGE, label_for
from regintel.answering.schema import Usage
from regintel.answering.verify import normalise

if TYPE_CHECKING:
    from collections.abc import Callable

    from openai import OpenAI

    from regintel.answering.schema import AnswerResult
    from regintel.config import Settings

REVIEW_PROMPT_VERSION = "v1"  # bump when REVIEW_PROMPT changes (checked in CI by regintel.prompts)

REVIEW_PROMPT = """\
You review answers about regulatory guidance for medicines. Each statement comes with the exact \
quotes that support it; the quotes are verified to be real text from the guidance. Judge only \
whether each statement says what its quotes say. Do not use outside knowledge.

For each statement give a verdict:
- supported: the quotes state the content of the statement. Rewording, summarising and combining \
its own quotes are fine.
- overreach: the statement goes beyond its quotes. For example it applies a general rule to a \
specific technology, product, population or situation the quotes do not mention; it turns "may" \
or "should" into "must"; it adds a cause, a reason, a condition, a scope or a jurisdiction that \
the quotes do not give.
- unsupported: the quotes do not support the statement at all.

A statement that concludes a general rule covers a subject the quotes never name (for example \
"it applies to all computerised systems, so it covers machine learning models") is overreach, even \
if the inference seems reasonable: the guidance did not say it.

Then decide addresses_question. If the question asks what the guidance says about a specific \
subject (a technology, product, population or situation) and no quote names that subject, the \
answer only offers general rules, so addresses_question is false; name the subject in \
missing_subject. Otherwise it is true.\
"""


class StatementVerdict(BaseModel):
    statement: int = Field(description="Statement number, as in S1, S2, ...")
    verdict: Literal["supported", "overreach", "unsupported"]
    reason: str = Field(description="One short sentence.")


class Review(BaseModel):
    verdicts: list[StatementVerdict]
    addresses_question: bool
    missing_subject: str | None = Field(description="The specific subject the quotes never mention, if any.")


def review_input(result: AnswerResult) -> str:
    blocks = [f"Question: {result.question}", ""]
    for number, statement in enumerate(result.shown, start=1):
        quotes = "\n".join(f'  - "{c.quote}" ({c.source_label})' for c in statement.citations)
        blocks.append(f"S{number}: {statement.text}\n  Quotes:\n{quotes}")
    return "\n".join(blocks)


class AnswerReviewer:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def judge(self, result: AnswerResult) -> tuple[Review, Usage]:
        response = self.client.responses.parse(
            model=self.model,
            instructions=REVIEW_PROMPT,
            input=review_input(result),
            text_format=Review,
            reasoning={"effort": "low"},
            store=False,
        )
        review = response.output_parsed
        if review is None:
            raise ValueError(f"reviewer returned no parsable review (status {response.status})")
        usage = Usage(calls=1)
        if response.usage is not None:
            usage.input_tokens = response.usage.input_tokens
            usage.output_tokens = response.usage.output_tokens
        return review, usage

    def review(self, result: AnswerResult, passages: list[str]) -> AnswerResult:
        """Return the result with overreaching statements dropped and the label recomputed. Answers
        with nothing shown (declined, blocked, withheld) are returned unchanged, without a call.
        `passages` are the texts the answer was written from (for the subject check)."""
        if not result.shown:
            return result
        review, usage = self.judge(result)
        return apply_review(result, review, passages, usage)


_STEM = re.compile(r"[a-z0-9]+")
_GENERIC = frozenset({"the", "and", "for", "with", "that", "this", "from", "into", "about", "under", "specific",
                      "requirement", "requirements", "guidance", "subject", "asked", "mandatory"})  # fmt: skip


def subject_in_passages(subject: str, passages: list[str]) -> bool:
    """True if one passage contains every content word of the subject, compared on 5-letter stems
    after dropping a plural s, so plurals and hyphenation match. The reviewer proposes "not
    addressed"; code confirms it."""
    words = {_stem(w) for w in _STEM.findall(normalise(subject)) if len(w) > 2 and w not in _GENERIC}
    if not words:
        return True  # nothing concrete was named: do not decline on it
    return any(words <= {_stem(w) for w in _STEM.findall(normalise(text))} for text in passages)


def _stem(word: str) -> str:
    return word.rstrip("s")[:5]


def apply_review(result: AnswerResult, review: Review, passages: list[str], usage: Usage | None = None) -> AnswerResult:
    verdicts = {v.statement: v for v in review.verdicts}
    checked = []
    shown_number = 0
    for statement in result.statements:
        if not statement.verified:
            checked.append(statement)
            continue
        shown_number += 1  # the reviewer saw shown statements as S1, S2, ... in this order
        verdict = verdicts.get(shown_number)
        if verdict is None or verdict.verdict == "supported":
            checked.append(statement)
        else:
            problem = f"reviewer: {verdict.verdict}: {verdict.reason}"
            checked.append(replace(statement, verified=False, problems=[*statement.problems, problem]))
    label = label_for(True, checked)
    reason = NOT_GROUNDED_MESSAGE if label == "not_grounded" else result.decline_reason
    subject = review.missing_subject or ""
    if not review.addresses_question and not subject_in_passages(subject, passages):
        label = "declined"
        reason = f"The indexed guidance does not specifically address {subject}."
    return replace(result, statements=checked, label=label, decline_reason=reason, review_usage=usage)


def reviewer_from_settings(settings: Settings, token_provider: Callable[[], str]) -> AnswerReviewer | None:
    """The reviewer on the judge endpoint (Entra ID), or None when no judge endpoint is configured."""
    from openai import OpenAI

    if not settings.azure_judge_endpoint:
        return None
    client = OpenAI(base_url=settings.azure_judge_endpoint.rstrip("/") + "/openai/v1/", api_key=token_provider)
    return AnswerReviewer(client, settings.azure_judge_deployment)
