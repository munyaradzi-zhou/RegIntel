"""Writer selection and the Chat Completions writer (models from other families on Azure)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from regintel.answering.models import ChatAnswerModel, ResponsesAnswerModel, answer_model_from_settings
from regintel.answering.schema import DraftAnswer
from regintel.config import Settings

DRAFT = DraftAnswer(answerable=True, statements=[], decline_reason=None)


def _response(parsed: DraftAnswer | None, finish: str = "stop") -> SimpleNamespace:
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=20,
        prompt_tokens_details=SimpleNamespace(cached_tokens=40),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
    )
    message = SimpleNamespace(parsed=parsed)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)], usage=usage)


class FakeCompletions:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[no-any-return]


def _client(outcomes: list[Any]) -> tuple[Any, FakeCompletions]:
    completions = FakeCompletions(outcomes)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def _invalid_json() -> ValidationError:
    try:
        DraftAnswer.model_validate_json("not json")
    except ValidationError as exc:
        return exc
    raise AssertionError


def test_chat_writer_sends_the_same_prompt_and_schema_and_reports_usage() -> None:
    client, completions = _client([_response(DRAFT)])
    draft, usage = ChatAnswerModel(client, "Llama-3.3-70B-Instruct").draft("A question?", [])
    assert draft == DRAFT
    call = completions.calls[0]
    assert call["model"] == "Llama-3.3-70B-Instruct"
    assert call["response_format"] is DraftAnswer
    assert [m["role"] for m in call["messages"]] == ["system", "user"]
    assert "A question?" in call["messages"][1]["content"]
    assert (usage.input_tokens, usage.output_tokens, usage.cached_tokens, usage.reasoning_tokens) == (100, 20, 40, 5)


def test_chat_writer_retries_malformed_output_and_counts_it() -> None:
    client, completions = _client([_invalid_json(), _response(None, "length"), _response(DRAFT)])
    model = ChatAnswerModel(client, "m")
    assert model.draft("q", [])[0] == DRAFT
    assert model.malformed == 2
    assert len(completions.calls) == 3


def test_chat_writer_gives_up_after_its_attempts() -> None:
    client, _ = _client([_invalid_json(), _invalid_json()])
    model = ChatAnswerModel(client, "m", attempts=2)
    with pytest.raises(ValidationError):
        model.draft("q", [])
    assert model.malformed == 2


def _settings(**values: Any) -> Settings:
    base = {"azure_openai_endpoint": "https://oai.example", "writer_endpoint": None, "writer_deployment": None}
    return Settings(_env_file=None, **(base | values))  # type: ignore[call-arg]


def test_default_writer_is_the_azure_openai_deployment() -> None:
    model = answer_model_from_settings(_settings(), lambda: "token")
    assert isinstance(model, ResponsesAnswerModel)
    assert model.model == "gpt-5-mini"


def test_writer_settings_pick_api_endpoint_and_deployment() -> None:
    chat = answer_model_from_settings(
        _settings(writer_endpoint="https://ais.example", writer_deployment="Llama-3.3-70B-Instruct", writer_api="chat"),
        lambda: "token",
    )
    assert isinstance(chat, ChatAnswerModel)
    assert chat.model == "Llama-3.3-70B-Instruct"
    assert str(chat.client.base_url).startswith("https://ais.example/openai/v1")

    responses = answer_model_from_settings(
        _settings(writer_endpoint="https://ais.example", writer_deployment="grok-4.7", writer_reasoning_effort="none"),
        lambda: "token",
    )
    assert isinstance(responses, ResponsesAnswerModel)
    assert (responses.model, responses.reasoning_effort) == ("grok-4.7", None)
