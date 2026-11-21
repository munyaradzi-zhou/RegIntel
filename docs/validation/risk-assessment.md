# Risk assessment

What can go wrong with an answer or its record, how bad it would be, and what covers it. Severity
and likelihood are high / medium / low; "residual" is what is left with the controls in place.

| # | Hazard | Severity | Control (requirement) | Residual |
|---|---|---|---|---|
| R1 | The writer misquotes guidance or invents a number | High | Verifier drops the statement (URS-01, URS-02); measured on every PR (URS-13) | Low |
| R2 | Real quotes, false conclusion ("should" read as "must", a rule stretched to a new subject) | High | Reviewer from another model family (URS-03) | Medium: the reviewer is a model and varies between runs |
| R3 | The system answers a question the guidance does not cover | High | Decline paths and the subject check (URS-04); 8/8 unanswerable declined live | Low |
| R4 | Prompt injection in a question or a retrieved passage | High | Screens before any model call and on every passage and tool result (URS-05) | Low |
| R5 | A poisoned document enters the corpus | High | Checksummed manifest (URS-15) | Low: the injection evaluation's one success was a forged passage, which the checksum blocks |
| R6 | A safety number is generated instead of computed | High | Data tools with parameterised queries; numbers verified against tool output (URS-06) | Low |
| R7 | openFDA is down or slow and the answer cites a label it never saw | High | No evidence on failure; bounded retries and timeouts (URS-07) | Low |
| R8 | An answer is released without a record | Medium | Record written first; fail closed (URS-08) | Low: answers become unavailable while Cosmos is down, which is the intended trade |
| R9 | A record is edited in the database (a label upgraded, a statement removed) | High | Hash chain; `/v1/audit/verify` reports the first broken record (URS-10) | Low |
| R10 | The newest records are deleted together with the head pointer | Medium | Day heads link the days; the head is reported by `/v1/audit/verify`, so a copy of it kept elsewhere exposes truncation | Medium: needs an external copy of the head (immutable Blob export is the next step) |
| R11 | Two replicas append at once and fork the chain | Medium | Transactional batch conditional on the head's ETag; the loser retries (URS-10) | Low |
| R12 | An answer cannot be explained later because a source changed | Medium | Sources recorded with sha256; tool records kept in full; replay reports changes (URS-11) | Low |
| R13 | A prompt is changed without anyone noticing | Medium | Versions and hashes in every answer and record; the gate fails an unversioned change (URS-12) | Low |
| R14 | A regression reaches production | Medium | Required checks and the evaluation gate (URS-13) | Medium: the gate is offline (BM25 and the verifier); answer quality is measured on demand |
| R15 | Someone without a key uses the API, or one client floods it | Low | API keys and rate limits (URS-14) | Low |
| R16 | A caller's identity leaks through the audit trail | Medium | Records hold a key id (hash prefix), never the key (URS-09) | Low |
