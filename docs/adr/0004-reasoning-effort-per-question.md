# 0004. Retrieval reasoning effort chosen per question shape

**Status:** Accepted, 2026-11-14

## Context

Azure AI Search knowledge bases plan sub-queries with a model at a chosen effort (`minimal`, `low`, `medium`). More planning finds more passages for multi-part questions but costs seconds on every query.

## Decision

A router classifies each question: multi-part guidance questions use `medium`, single lookups `low`, and data questions go to the agent, which searches at `minimal` and does its own decomposition.

## Alternatives

- One effort for everything: `medium` everywhere is slow; `minimal` everywhere misses multi-part evidence.
- Local hybrid search only: free, but finds every supporting passage on far fewer multi-part questions.

## Consequences

- One model call per question for routing (1.7 s, about $0.20 per 1,000 questions). A trained router ([0007](0007-llm-router-stays-default.md)) removes the call but misroutes multi-part questions.
- Routing a multi-part question to `low` can miss a source; the router's errors are measured (3 of 76 questions).

## Evidence

`medium` found every supporting passage on 9 of 10 multi-part questions, against 4 of 10 for local hybrid search, but cost about 11 s per query and did not help single lookups. Routing by shape kept answer quality and made single questions about 35% faster.
