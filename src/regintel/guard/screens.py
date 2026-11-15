"""Pattern screens for prompt injection. Patterns target the *form* of an injection (an instruction
aimed at the assistant), not topics, so regulatory questions that mention "ignore", "system" or
"instructions" in their ordinary sense pass. Their false-positive rate is measured on benign
questions and on every chunk of the corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from regintel.ingestion.chunking import Chunk

MAX_QUESTION_CHARS = 2_000

_OVERRIDE = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass)\b[^.?!\n]{0,40}"
    r"\b(?:previous|prior|above|earlier|all|your|the|these|system)\b[^.?!\n]{0,30}"
    r"\b(?:instructions?|rules?|prompts?|guidelines?|constraints?|directives?)\b",
    re.IGNORECASE,
)
_ROLE = re.compile(
    r"\b(?:you are now|from now on,? you|act as (?:an?|the) (?:unrestricted|different|new)|"
    r"pretend (?:to be|you are)|developer mode|jailbreak|DAN\b)",
    re.IGNORECASE,
)
_LEAK = re.compile(
    r"\b(?:reveal|print|show|repeat|output|tell me|what (?:is|are))\b[^.?!\n]{0,40}"
    r"\b(?:system prompt|your (?:instructions|prompt|rules)|hidden (?:instructions|prompt))\b",
    re.IGNORECASE,
)
# Instructions addressed to a model, inside a passage: role tags or imperative overrides.
_PASSAGE = re.compile(
    r"(?:^|\n)\s*(?:system|assistant|user)\s*:|<\s*/?\s*(?:system|instructions?)\s*>|"
    r"\b(?:AI|assistant|language model|LLM)s?\b[^.\n]{0,40}\b(?:must|should) (?:now )?(?:answer|say|state|reply)",
    re.IGNORECASE,
)

Category = Literal["ok", "empty", "too_long", "instruction_override", "role_change", "prompt_extraction"]


@dataclass(frozen=True)
class Screen:
    category: Category
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.category == "ok"


def screen_question(question: str) -> Screen:
    text = question.strip()
    if not text:
        return Screen("empty", "The question is empty.")
    if len(text) > MAX_QUESTION_CHARS:
        return Screen("too_long", f"The question is longer than {MAX_QUESTION_CHARS} characters.")
    for category, pattern in (
        ("instruction_override", _OVERRIDE),
        ("role_change", _ROLE),
        ("prompt_extraction", _LEAK),
    ):
        match = pattern.search(text)
        if match:
            return Screen(category, f"The question contains an instruction to the assistant: {match.group(0)!r}.")  # type: ignore[arg-type]
    return Screen("ok")


def passage_is_suspicious(text: str) -> bool:
    return bool(_PASSAGE.search(text) or _OVERRIDE.search(text) or _ROLE.search(text))


def screen_passages(chunks: list[Chunk]) -> tuple[list[Chunk], list[Chunk]]:
    """(kept, dropped): passages that address instructions to a model are dropped before writing."""
    kept = [c for c in chunks if not passage_is_suspicious(c.text)]
    dropped = [c for c in chunks if passage_is_suspicious(c.text)]
    return kept, dropped
