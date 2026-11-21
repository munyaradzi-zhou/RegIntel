"""Route each question to the cheapest path that the evaluations show can answer it.

- needs FAERS or CDISC data            -> research agent (tools)
- guidance only, several parts         -> fixed pipeline, knowledge base at medium effort
  (query planning found every needed passage for 9/10 multi-hop questions vs 5/10 at minimal)
- guidance only, one focused question  -> fixed pipeline, knowledge base at minimal effort
  (best section@1 and ~1 s retrieval on single questions)

The classification is one structured-output call at minimal reasoning effort. Its accuracy is
measured against the golden and tool question sets (tests/evaluation/evaluate_router.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from regintel.answering.schema import Usage

if TYPE_CHECKING:
    from openai import OpenAI

Path = Literal["agent", "guidance_multi", "guidance_single"]

ROUTER_PROMPT_VERSION = "v1"  # bump when ROUTER_PROMPT changes (checked in CI by regintel.prompts)

ROUTER_PROMPT = """\
Classify a question for a regulatory question-answering system for medicines. The system has \
indexed FDA, EMA/EU and ICH guidance documents, plus two data tools: counts of adverse event \
reports from FAERS (including disproportionality signals), and CDISC Controlled Terminology \
(codelists, submission values, NCI codes).

needs_data_tools: true only if answering requires FAERS report counts or signals, or a CDISC \
terminology code, codelist or submission value. Questions about what guidance says on safety \
reporting, terminology or data standards in general are guidance questions (false).

multi_part: true if the question asks about two or more regulators or documents (for example a \
comparison), or asks for two or more distinct facts that are likely in different sections.\
"""


class RouteDecision(BaseModel):
    needs_data_tools: bool = Field(description="Needs FAERS counts/signals or CDISC terminology codes.")
    multi_part: bool = Field(description="Several regulators, documents or distinct facts.")
    reason: str = Field(description="One short sentence.")

    @property
    def path(self) -> Path:
        if self.needs_data_tools:
            return "agent"
        return "guidance_multi" if self.multi_part else "guidance_single"


class Router:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def route(self, question: str) -> tuple[RouteDecision, Usage]:
        response = self.client.responses.parse(
            model=self.model,
            instructions=ROUTER_PROMPT,
            input=question,
            text_format=RouteDecision,
            reasoning={"effort": "minimal"},
            store=False,
        )
        decision = response.output_parsed
        if decision is None:  # fail towards the most capable path
            return RouteDecision(needs_data_tools=True, multi_part=True, reason="unparsable route"), Usage(calls=1)
        usage = Usage(calls=1)
        if response.usage is not None:
            usage.input_tokens = response.usage.input_tokens
            usage.output_tokens = response.usage.output_tokens
            usage.reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens or 0
        return decision, usage
