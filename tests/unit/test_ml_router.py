"""The trained router: features, the served model file, fallback, and routing as a setting."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from regintel.agent.factory import AzureComponents
from regintel.answering.schema import Usage
from regintel.ml_router import FallbackRouter, MLRouter, RouterModel, features, load_model, raw_features
from regintel.router import RouteDecision, Router


def test_features_are_words_bigrams_and_character_grams() -> None:
    grams = raw_features("FAERS reports for semaglutide")
    assert grams["w:faers"] == 1 and grams["b:reports for"] == 1 and grams["c:<se"] == 1
    vocab = {g: i for i, g in enumerate(sorted(grams))}
    vector = features("FAERS reports for semaglutide", vocab, [1.0] * len(vocab))
    assert math.isclose(math.sqrt(sum(v * v for v in vector.values())), 1.0)  # L2-normalised
    assert features("nothing known here", {"w:faers": 0}, [1.0]) == {}


def test_served_model_loads_by_version_and_routes_data_questions_to_the_agent() -> None:
    model = load_model(1)
    assert model.version == 1 and len(model.sha256) == 64 and model.classes == ["agent", "guidance_multi",
                                                                                 "guidance_single"]  # fmt: skip
    assert model.meta["mlflow_run_id"] and set(model.thresholds) == set(model.classes)
    route, p = model.predict("How many FAERS reports list dupilumab as a suspect drug?")
    assert route == "agent" and p > 0.5
    assert math.isclose(sum(model.probabilities("What does ICH E6 say about monitoring?").values()), 1.0)
    with pytest.raises(FileNotFoundError):
        load_model(999)


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def route(self, question: str) -> tuple[RouteDecision, Usage]:
        self.calls += 1
        return RouteDecision(needs_data_tools=False, multi_part=True, reason="compares two"), Usage(calls=1)


def _model(**thresholds: float) -> RouterModel:
    return replace(load_model(1), thresholds={"agent": 0.0, "guidance_multi": 0.0, "guidance_single": 0.0} | thresholds)


def test_ml_router_decides_without_a_model_call() -> None:
    decision, usage = MLRouter(load_model(1)).route("What is the reporting odds ratio for nausea with semaglutide?")
    assert decision.path == "agent" and usage.calls == 0 and decision.reason.startswith("ml v1: agent")


def test_fallback_defers_only_below_the_threshold_of_the_predicted_route() -> None:
    llm = FakeLLM()
    confident = FallbackRouter(MLRouter(_model()), llm)  # type: ignore[arg-type]
    assert confident.route("How many FAERS reports mention fentanyl?")[0].path == "agent" and llm.calls == 0

    unsure = FallbackRouter(MLRouter(_model(agent=1.01)), llm)  # type: ignore[arg-type]
    decision, usage = unsure.route("How many FAERS reports mention fentanyl?")
    assert llm.calls == 1 and unsure.fallbacks == 1 and usage.calls == 1
    assert decision.path == "guidance_multi" and decision.reason.startswith("llm (ml agent p=")


@pytest.mark.parametrize(("routing", "kind"), [("llm", Router), ("ml", MLRouter), ("ml_with_fallback", FallbackRouter)])
def test_routing_is_a_setting(routing: str, kind: type[Any]) -> None:
    settings = SimpleNamespace(routing=routing, router_model_version=1, azure_openai_endpoint="https://oai.example",
                               azure_openai_chat_deployment="gpt-5-mini")  # fmt: skip
    components = SimpleNamespace(settings=settings, tokens=lambda: "token")
    assert isinstance(AzureComponents.router(components), kind)  # type: ignore[arg-type]
