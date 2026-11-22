# 0005. DuckDB with parameterised queries instead of text-to-SQL

**Status:** Accepted, 2026-11-14

## Context

Questions about FAERS need counts and reporting odds ratios. Numbers in safety answers must be reproducible and correct, and a model writing SQL is both an injection surface and a source of silently wrong queries.

## Decision

The agent never writes SQL. It calls typed tools (`faers_drug_summary(drug)`, `faers_disproportionality(drug, reaction)`); Python builds DuckDB expressions from those typed values over Parquet tables, computes the statistics, and returns sentences that the verifier then checks numbers against.

## Alternatives

- Text-to-SQL over the FAERS tables: flexible, but the query logic cannot be verified and it is an injection surface.
- Letting the model recall or estimate FAERS figures: not reproducible.

## Consequences

- Only the questions the tools support can be answered (counts, top reactions, reporting odds ratios with 95% CI); new kinds of question need new tools.
- Every number in an answer is in a tool record, so the verifier covers data answers as it covers guidance answers.

## Evidence

Tool questions answered with every expected fact (16/16 in the evaluation runs of 2026-11-21 and 2026-11-22), with FAERS ground truth computed from the data at evaluation time.
