# 0007. The LLM router stays the default over the trained router

**Status:** Accepted, 2026-11-22

## Context

A trained router (TF-IDF features into logistic regression) routes in under a millisecond with no model call, against 1.7 s for the LLM router.

## Decision

`ROUTING=llm` stays the default. `ml` and `ml_with_fallback` are available; the trained model is versioned (MLflow registry, JSON export) and its held-out scores are in the CI gate.

## Alternatives

- `ml`: free and instant, but 0.826 accuracy on held-out questions.
- `ml_with_fallback` (per-route thresholds at 0.95 out-of-fold precision): 0.913, with 61% of questions still going to the LLM.
- Embedding features (`text-embedding-3-large`): no better on held-out questions, and an API call per question.

## Consequences

- Routing costs about $0.20 per 1,000 questions and 1.7 s of a ~13 s answer: a small saving next to sending a multi-part question down the wrong path.
- More real labelled questions (23 held out today) are the way to revisit this.

## Evidence

Held-out real questions: LLM router 0.957 (macro-F1 0.938), trained 0.826 (0.722). The trained router got all 5 data questions right; its misses are multi-part questions sent down the single path.
