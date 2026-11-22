# 0003. Reviewer from a different model family

**Status:** Accepted, 2026-11-14

## Context

An answer can quote real text and still conclude something the text does not say ("should" read as "must", a rule stretched to a subject the guidance never names). String checks cannot see this. A judge from the writer's own family tends to share its blind spots.

## Decision

A second model, `gpt-oss-120b` (the writer is `gpt-5-mini`), judges each verified statement against its quotes only: supported, overreach or unsupported. Anything but supported is dropped. Its "the question's subject is not addressed" verdict declines the answer only when code confirms that no passage names the subject.

## Alternatives

- No reviewer: overreach reaches users.
- A same-family judge: shared blind spots.
- A reviewer with full context or the writer's reasoning: it would judge the reasoning, not the quotes.

## Consequences

- One more model call per answer.
- The reviewer is a model and varies between runs (11 vs 13 drops on the same drafts), so its counts are compared with care.
- Its subject check can still over-decline: Azure AI Evaluation flagged 3 answerable questions declined although their passages scored 5/5 for retrieval.

## Evidence

Calibration set: 8 of 8 planted problems caught, 0 of 13 good statements rejected. Requiring code to confirm "not addressed" moved the answer rate from 0.942 to 1.0 on the cases the reviewer alone declined wrongly.
