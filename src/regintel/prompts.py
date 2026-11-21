"""Every model prompt in RegIntel, with its version and a hash of its text.

Each prompt keeps a hand-set version next to it (``PROMPT_VERSION`` and friends). The hash catches
the case where the text changed but the version did not: the CI gate fails on it, so every
evaluation result and every answer can be traced to the exact prompt that produced it.
"""

from __future__ import annotations

import hashlib
from functools import cache


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


@cache
def prompts() -> dict[str, tuple[str, str]]:
    """name -> (version, text). Imported lazily: the prompt modules pull in the whole service."""
    from regintel.agent.research import INSTRUCTIONS, INSTRUCTIONS_VERSION
    from regintel.answering.prompt import PROMPT_VERSION, SYSTEM_PROMPT
    from regintel.answering.review import REVIEW_PROMPT, REVIEW_PROMPT_VERSION
    from regintel.conversation import REWRITE_PROMPT, REWRITE_PROMPT_VERSION
    from regintel.router import ROUTER_PROMPT, ROUTER_PROMPT_VERSION

    return {
        "writer": (PROMPT_VERSION, SYSTEM_PROMPT),
        "router": (ROUTER_PROMPT_VERSION, ROUTER_PROMPT),
        "reviewer": (REVIEW_PROMPT_VERSION, REVIEW_PROMPT),
        "rewriter": (REWRITE_PROMPT_VERSION, REWRITE_PROMPT),
        "agent": (INSTRUCTIONS_VERSION, INSTRUCTIONS),
    }


def prompt_versions() -> dict[str, dict[str, str]]:
    """name -> {"version": "v5", "sha256": "<first 12 hex>"}, for result files and answer metadata."""
    return {name: {"version": version, "sha256": _digest(text)} for name, (version, text) in prompts().items()}


def prompt_tags() -> dict[str, str]:
    """name -> "v5+1a2b3c4d5e6f": the compact form carried by each API answer."""
    return {name: f"{p['version']}+{p['sha256']}" for name, p in prompt_versions().items()}
