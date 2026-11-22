"""Build the router dataset: real questions split once into train and test, plus generated
training variants.

    python scripts/build_router_data.py            # writes tests/evaluation/datasets/router_dataset.json

1. The 76 real questions (golden set + tool questions) are labelled with their route and split,
   stratified by route, 70/30 with a fixed seed. The test split is never shown to the generator
   and never used for training: it measures how the router does on questions it has not seen.
2. For each training question, gpt-5-mini writes paraphrases (same meaning, different wording)
   and new questions of the same kind (same route, different subject). Each variant records its
   parent, so a variant never crosses into the test split.

The output is committed, so training runs offline and is reproducible.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from regintel.answering.models import azure_openai_client
from regintel.config import get_settings

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "tests" / "evaluation" / "datasets"
OUT = DATASETS / "router_dataset.json"
SEED = 20261122
TEST_SHARE = 0.3
PARAPHRASES = 4
NEW_QUESTIONS = 3

ROUTES = {
    "agent": "needs a data tool: FAERS adverse-event report counts or reporting odds ratios, a CDISC "
    "Controlled Terminology code, codelist or submission value, what a named drug's US label says, or "
    "recalls of a named drug",
    "guidance_multi": "a question about regulatory guidance text (FDA, EMA/EU, ICH) that compares two or more "
    "regulators or documents, or asks for two or more distinct facts likely in different sections",
    "guidance_single": "one focused question about what regulatory guidance text (FDA, EMA/EU, ICH) says, "
    "answerable from one section; includes questions about safety reporting, terminology or data "
    "standards in general, which are guidance questions, not data questions",
}
GOLDEN_ROUTE = {"single": "guidance_single", "unanswerable": "guidance_single",
                "comparison": "guidance_multi", "multihop": "guidance_multi"}  # fmt: skip

PROMPT = """\
You write test questions for the router of a regulatory question-answering system for medicines. \
The route of the question below is "{route}": {definition}.

Question: {question}

Write {paraphrases} paraphrases: the same question in different words, as different users would ask \
it (vary length, formality and word order; keep every regulator, document, drug and number). Then \
write {new} new questions of the same route about different subjects, realistic for a regulatory \
affairs or pharmacovigilance professional. Every question must clearly belong to the route "{route}".\
"""


class Variants(BaseModel):
    paraphrases: list[str] = Field(description="Same meaning, different wording.")
    new_questions: list[str] = Field(description="Same route, different subjects.")


def real_questions() -> list[dict[str, Any]]:
    golden = json.loads((DATASETS / "golden_set.json").read_text(encoding="utf-8"))["questions"]
    tools = json.loads((DATASETS / "tool_questions.json").read_text(encoding="utf-8"))["questions"]
    rows = [{"id": q["id"], "question": q["question"], "route": GOLDEN_ROUTE[q["type"]], "kind": q["type"]}
            for q in golden]  # fmt: skip
    rows += [{"id": q["id"], "question": q["question"], "route": "agent", "kind": "data"} for q in tools]
    return rows


def split(rows: list[dict[str, Any]]) -> None:
    rng = random.Random(SEED)  # noqa: S311  (a reproducible split, not security)
    by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_route[row["route"]].append(row)
    for group in by_route.values():
        rng.shuffle(group)
        n_test = round(len(group) * TEST_SHARE)
        for i, row in enumerate(group):
            row["split"] = "test" if i < n_test else "train"


def main() -> None:
    settings = get_settings()
    from azure.identity import AzureCliCredential, get_bearer_token_provider

    tokens = get_bearer_token_provider(AzureCliCredential(), "https://cognitiveservices.azure.com/.default")
    client = azure_openai_client(settings, tokens)

    rows = real_questions()
    split(rows)
    for row in rows:
        row["source"] = "real"
    generated = []
    for parent in (r for r in rows if r["split"] == "train"):
        response = client.responses.parse(
            model=settings.azure_openai_chat_deployment,
            input=PROMPT.format(
                route=parent["route"],
                definition=ROUTES[parent["route"]],
                question=parent["question"],
                paraphrases=PARAPHRASES,
                new=NEW_QUESTIONS,
            ),
            text_format=Variants,
            reasoning={"effort": "low"},
            store=False,
        )
        variants = response.output_parsed or Variants(paraphrases=[], new_questions=[])
        for kind, texts in (("paraphrase", variants.paraphrases), ("new", variants.new_questions)):
            for n, text in enumerate(texts, start=1):
                generated.append({"id": f"{parent['id']}-{kind[0]}{n}", "question": text.strip(),
                                  "route": parent["route"], "kind": kind, "split": "train",
                                  "source": "generated", "parent": parent["id"]})  # fmt: skip
        added = f"+{len(variants.paraphrases)} paraphrases +{len(variants.new_questions)} new"
        print(f"{parent['id']:8} {parent['route']:16} {added}")

    OUT.write_text(
        json.dumps(
            {
                "description": "Router training and test data. Real questions from golden_set.json and "
                "tool_questions.json, split once (seed below); generated variants come only from training "
                "questions. See docs/router-data-card.md.",
                "seed": SEED,
                "generator": settings.azure_openai_chat_deployment,
                "questions": rows + generated,
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    counts = defaultdict(int)
    for q in rows + generated:
        counts[(q["split"], q["source"], q["route"])] += 1
    for key in sorted(counts):
        print(key, counts[key])


if __name__ == "__main__":
    main()
