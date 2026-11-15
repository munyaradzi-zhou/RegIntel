"""Answer-writing models behind one interface, so the pipeline and its evaluation are provider-agnostic.

`ResponsesAnswerModel` uses the OpenAI Responses API with structured output (the DraftAnswer
schema), which serves both Azure OpenAI (Entra ID token on the resource's /openai/v1/ endpoint,
no keys) and OpenAI itself (API key). The system prompt is a fixed prefix and the requests share
a cache key, so repeated prefixes are served from the prompt cache; cached tokens are reported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

from openai import OpenAI

from regintel.answering.prompt import PROMPT_VERSION, SYSTEM_PROMPT, user_message
from regintel.answering.schema import DraftAnswer, Usage

if TYPE_CHECKING:
    from collections.abc import Callable

    from regintel.config import Settings
    from regintel.retrieval.base import Hit


class AnswerModel(Protocol):
    name: str

    def draft(self, question: str, hits: list[Hit]) -> tuple[DraftAnswer, Usage]: ...


class ResponsesAnswerModel:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = "low",
    ) -> None:
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.name = f"{model}" + (f"-{reasoning_effort}" if reasoning_effort else "") + f" (prompt {PROMPT_VERSION})"

    def draft(self, question: str, hits: list[Hit]) -> tuple[DraftAnswer, Usage]:
        response = self.client.responses.parse(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=user_message(question, hits),
            text_format=DraftAnswer,
            reasoning={"effort": self.reasoning_effort} if self.reasoning_effort else None,
            prompt_cache_key=f"regintel-answer-{PROMPT_VERSION}",
            store=False,
        )
        draft = response.output_parsed
        if draft is None:
            raise ValueError(f"model returned no parsable answer (status {response.status})")
        usage = Usage(calls=1)
        if response.usage is not None:
            usage.input_tokens = response.usage.input_tokens
            usage.output_tokens = response.usage.output_tokens
            usage.cached_tokens = response.usage.input_tokens_details.cached_tokens or 0
            usage.reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens or 0
        return draft, usage


def azure_openai_client(settings: Settings, token_provider: Callable[[], str]) -> OpenAI:
    if not settings.azure_openai_endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT is not set")
    return OpenAI(base_url=settings.azure_openai_endpoint.rstrip("/") + "/openai/v1/", api_key=token_provider)


def answer_model_from_settings(settings: Settings, token_provider: Callable[[], str] | None = None) -> AnswerModel:
    """Azure OpenAI when an endpoint and token provider are available, else OpenAI (local runs)."""
    if settings.azure_openai_endpoint and token_provider is not None:
        return ResponsesAnswerModel(
            azure_openai_client(settings, token_provider), settings.azure_openai_chat_deployment
        )
    if not settings.openai_api_key:
        raise ValueError("set AZURE_OPENAI_ENDPOINT (with az login) or OPENAI_API_KEY")
    return ResponsesAnswerModel(
        OpenAI(api_key=settings.openai_api_key.get_secret_value()), settings.openai_model, reasoning_effort=None
    )
