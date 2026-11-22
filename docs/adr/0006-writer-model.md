# 0006. Writer model: gpt-5-mini, measured against other model families

**Status:** Accepted, 2026-11-21 (replaces the plan to compare with Claude)

## Context

The plan was to compare the writer with Claude. On an Azure free trial, Claude is sold through the Marketplace, which trial credit cannot pay for, so the comparison used Azure-sold models from other families on the same subscription.

## Decision

`gpt-5-mini` stays the writer. The writer is a setting (`WRITER_ENDPOINT`, `WRITER_DEPLOYMENT`, `WRITER_API`), with a Chat Completions path for models that ignore the answer schema on the Responses API.

## Alternatives

- `grok-4.7` (xAI): best facts and fewest misquotes, at twice the latency.
- `Llama-3.3-70B-Instruct` (Meta): malformed JSON 3 times, lowest fact recall.
- `DeepSeek-V3.2-Speciale`: most statements dropped by the verifier, and it answered one unanswerable question.

## Consequences

- Switching to grok-4.7 is one setting, if latency matters less than recall.
- The comparison held retrieval, prompt, verifier and reviewer fixed, so the differences are the writer's.

## Evidence

60 golden questions with identical retrieved passages. Fact recall: gpt-5-mini 0.942, grok-4.7 0.971, Llama 0.804, DeepSeek 0.926. Statements dropped by the verifier: 4.2%, 2.6%, 11.4%, 13.6%. Latency p50: 6.1, 12.1, 8.8, 13.2 s.
