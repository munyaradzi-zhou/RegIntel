# 0002. Verification in code, not by prompt

**Status:** Accepted, 2026-11-13

## Context

In regulatory work, a misquote or a wrong number in an answer is the worst failure. Prompting a model to "cite carefully" lowers the rate but cannot guarantee anything, and an LLM groundedness score can be argued with.

## Decision

The writer returns structured statements, each with verbatim quotes and passage numbers. A Python verifier checks every quote against its passage (tolerant of whitespace and typography, not of wording) and every number, by value, against the statement's quotes. Unverified statements are dropped; an answer with none left is withheld. The same verifier serves every path (guidance and agent).

## Alternatives

- Prompt-only grounding: no guarantee.
- An LLM groundedness judge as the gate: probabilistic, and blind to misquotes (see Evidence).

## Consequences

- Some true statements are dropped when the writer paraphrases inside a quote; the prompt asks for exact copies.
- The verifier is deterministic, so it is in the CI gate: 114 recorded and mutated drafts must be kept or dropped exactly as recorded.

## Evidence

- Statements verified rose 0.891 to 0.980 as the verifier and prompt improved.
- Azure AI Evaluation scored 12 of 13 drafts that RegIntel cut (18 misquoted, unsupported-number or overreaching statements) as 5/5 grounded: a semantic judge does not see what the verifier removes.
