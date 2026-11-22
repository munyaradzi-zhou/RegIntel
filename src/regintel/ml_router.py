"""A trained router: TF-IDF features into logistic regression, no model call.

The LLM router (regintel.router) spends one structured-output call, about 1.5 s, on every
question. This router classifies in well under a millisecond, offline. Below a confidence
threshold it can hand the question to the LLM router (ROUTING=ml_with_fallback).

The model is a JSON file (vocabulary, idf weights, coefficients), not a pickle: it can be read,
diffed and checksummed, and loading it executes no code. Features are computed by `features()`
below, which training (scripts/train_router.py) uses too, so serving and training cannot drift.
Versions live in `src/regintel/router_models/router-v<N>.json`, registered in MLflow at training.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from regintel.answering.schema import Usage
from regintel.router import RouteDecision

if TYPE_CHECKING:
    from regintel.router import Path as RoutePath
    from regintel.router import Router

MODELS = Path(__file__).parent / "router_models"
_WORD = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def raw_features(text: str) -> dict[str, int]:
    """Counts of word unigrams and bigrams, and character 3-5-grams inside words ("w:", "b:", "c:")."""
    words = tokens(text)
    counts: dict[str, int] = {}
    grams = [f"w:{w}" for w in words] + [f"b:{a} {b}" for a, b in itertools.pairwise(words)]
    for word in words:
        padded = f"<{word}>"
        grams += [f"c:{padded[i : i + n]}" for n in (3, 4, 5) for i in range(len(padded) - n + 1)]
    for gram in grams:
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def features(text: str, vocabulary: dict[str, int], idf: list[float]) -> dict[int, float]:
    """Sublinear TF-IDF over the vocabulary, L2-normalised, as a sparse {column: value}."""
    vector = {vocabulary[g]: (1.0 + math.log(n)) * idf[vocabulary[g]] for g, n in raw_features(text).items()
              if g in vocabulary}  # fmt: skip
    norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
    return {k: v / norm for k, v in vector.items()}


@dataclass(frozen=True)
class RouterModel:
    version: int
    classes: list[str]
    vocabulary: dict[str, int]
    idf: list[float]
    coef: list[list[float]]  # one row per class
    intercept: list[float]
    thresholds: dict[str, float]  # per predicted route: below it, defer to the LLM router
    meta: dict[str, Any]
    sha256: str

    @classmethod
    def load(cls, path: Path) -> RouterModel:
        raw = path.read_bytes()
        body = json.loads(raw)
        return cls(body["version"], body["classes"], body["vocabulary"], body["idf"], body["coef"],
                   body["intercept"], body["threshold"], body.get("meta", {}),
                   hashlib.sha256(raw).hexdigest())  # fmt: skip

    def probabilities(self, text: str) -> dict[str, float]:
        x = features(text, self.vocabulary, self.idf)
        scores = [b + sum(row[i] * v for i, v in x.items()) for row, b in zip(self.coef, self.intercept, strict=True)]
        top = max(scores)
        exps = [math.exp(s - top) for s in scores]
        total = sum(exps)
        return {c: e / total for c, e in zip(self.classes, exps, strict=True)}

    def predict(self, text: str) -> tuple[str, float]:
        probs = self.probabilities(text)
        route = max(probs, key=lambda c: probs[c])
        return route, probs[route]


@cache
def load_model(version: int) -> RouterModel:
    path = MODELS / f"router-v{version}.json"
    if not path.exists():
        raise FileNotFoundError(f"router model version {version} not found at {path}")
    model = RouterModel.load(path)
    if model.version != version:
        raise ValueError(f"{path.name} says version {model.version}")
    return model


def decision(route: RoutePath, reason: str) -> RouteDecision:
    return RouteDecision(needs_data_tools=route == "agent", multi_part=route == "guidance_multi", reason=reason)


class MLRouter:
    """Same interface as the LLM router: route(question) -> (decision, usage)."""

    def __init__(self, model: RouterModel) -> None:
        self.model = model

    def route(self, question: str) -> tuple[RouteDecision, Usage]:
        route, p = self.model.predict(question)
        return decision(route, f"ml v{self.model.version}: {route} p={p:.2f}"), Usage()  # type: ignore[arg-type]


class FallbackRouter:
    """The trained router when it is confident in its route; the LLM router otherwise. Thresholds
    are per predicted route (from training), since some routes are much easier to recognise."""

    def __init__(self, ml: MLRouter, llm: Router) -> None:
        self.ml = ml
        self.llm = llm
        self.fallbacks = 0

    def route(self, question: str) -> tuple[RouteDecision, Usage]:
        route, p = self.ml.model.predict(question)
        threshold = self.ml.model.thresholds.get(route, 1.0)
        if p >= threshold:
            return decision(route, f"ml v{self.ml.model.version}: {route} p={p:.2f}"), Usage()  # type: ignore[arg-type]
        self.fallbacks += 1
        llm_decision, usage = self.llm.route(question)
        llm_decision.reason = f"llm (ml {route} p={p:.2f} < {threshold:.2f}): {llm_decision.reason}"
        return llm_decision, usage
