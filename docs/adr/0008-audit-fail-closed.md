# 0008. Audit records are written before an answer is released

**Status:** Accepted, 2026-11-21

## Context

In a GxP setting an answer used as a record must be attributable and protected against undetected change (the ideas of 21 CFR Part 11 and EU GMP Annex 11). An answer without its record cannot be explained later.

## Decision

Each answer's record (caller key id, sources with sha256, models, prompt versions, raw draft, every decision) is hash-chained in Cosmos DB and written before the response is returned. If the write fails, the API returns 503 and withholds the answer. Appends are transactional batches conditional on the chain head's ETag, so two replicas cannot fork the chain.

## Alternatives

- Best-effort logging after the response: answers could exist with no record.
- Application Insights as the audit trail: sampled, retention-limited, no chain.

## Consequences

- Answers depend on Cosmos DB: an outage makes the API unavailable, by design.
- Removing the newest records together with the head pointer is not detectable from the chain alone; an external copy of the head (immutable Blob export) is the next step.

## Evidence

Live: 4 concurrent answers recorded with no fork; a hand-edited record reported at its exact sequence number; two live answers replayed with identical verifier decisions. The first deploy found a real bug (Cosmos DB returns 0.0 as 0, which changed the hash), fixed by canonicalising numbers.
