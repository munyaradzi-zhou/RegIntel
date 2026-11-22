# Evaluation results in detail

The README keeps the headline numbers; this page keeps the comparisons behind them.

What the evaluations changed along the way:

- **Chunking beat the pipeline.** With the same embeddings, planner and reranker, section-aware chunks beat Azure's integrated chunking on every retrieval metric and kept section-level citations.
- **Reasoning effort is not "more is better".** `medium` found every supporting passage on 9 of 10 multi-part questions, against 4 of 10 for local hybrid search, but cost 11 s per query and did not help single lookups. Routing by question shape kept answer quality and made single questions about 35% faster.
- **Verbatim checks catch misquotes, not overreach.** The reviewer was added after an answer in which every quote was real but the conclusion was not; on hand-built cases it caught 8 of 8 planted problems and rejected 0 of 13 good statements.
- **The first live evaluation caught a production bug.** Blocking model calls held the API's event loop for ~16 s, health probes timed out and the platform restarted the container mid-run (15 of 70 requests failed). The blocking work now runs in worker threads: 0 of 70 failed.
- **Prompt injection.** Of 13 attacks (direct, prompt extraction, and instructions planted in retrieved passages), 1 got through: a forged passage containing no instructions, which is data poisoning rather than injection; the corpus is pinned by checksum against exactly that. No benign question was blocked.

**Writers from four model families**, on the same 60 questions with identical retrieved passages, the same prompt and schema, and the same verifier and reviewer. Only the writer changes:

| Writer | Answered | Declined | Fact recall | All facts | Statements dropped by the verifier | Dropped by the reviewer | Latency p50 / p95 | Tokens in / out per answer |
|---|---|---|---|---|---|---|---|---|
| `gpt-5-mini` (OpenAI) | 52 / 52 | 8 / 8 | 0.942 | 0.904 | 11 of 264 (4.2%) | 11–13 | 6.1 / 11.8 s | 3,812 / 860 |
| `grok-4.7` (xAI) | 52 / 52 | 8 / 8 | **0.971** | **0.942** | **6 of 233 (2.6%)** | **6** | 12.1 / 26.5 s | 3,929 / 470 |
| `Llama-3.3-70B-Instruct` (Meta) | 50 / 52 | 8 / 8 | 0.804 | 0.692 | 15 of 132 (11.4%) | 8 | 8.8 / 64.5 s | 3,544 / 618 |
| `DeepSeek-V3.2-Speciale` | 52 / 52 | 7 / 8 | 0.926 | 0.885 | 42 of 309 (13.6%) | 18 | 13.2 / 24.0 s | 3,558 / 744 |

- Grok-4.7 recalled the most facts and misquoted the least, at twice gpt-5-mini's latency; gpt-5-mini stays the default for speed.
- DeepSeek wrote the most statements and had the most dropped, mostly numbers its quotes did not contain (33). It also answered one of the unanswerable questions, with a true but beside-the-point statement. Without the verifier, those would have been shown.
- Llama and DeepSeek ignore the answer schema on the Responses API but follow it through Chat Completions, so the writer supports both (`WRITER_API`). Llama still returned malformed JSON 3 times; the writer retries, and one question with no valid draft is scored as a failure, not skipped. Its retries account for the 64 s p95.
- The reviewer is a model too: re-reviewing the same gpt-5-mini drafts dropped 11 statements on one pass and 13 on the next, so review counts are compared with care. The verifier is deterministic.

The writer is chosen with `WRITER_ENDPOINT`, `WRITER_DEPLOYMENT` and `WRITER_API`; all four are pay-as-you-go deployments on Azure, so the comparison ran on the same subscription.

**A trained router against the LLM router.** The router is one gpt-5-mini call per question. The alternative is TF-IDF features into logistic regression, trained on 53 real questions plus 371 generated variants, tracked and registered in MLflow, and served from a JSON file. All three were scored on 23 real questions held out from training and from the generator ([data card](docs/router-data-card.md)):

| `ROUTING` | Held-out accuracy | Macro-F1 | LLM calls | Routing time per question | Cost per 1,000 questions |
|---|---|---|---|---|---|
| `llm` (default) | **0.957** | **0.938** | every question | 1.69 s | $0.20 |
| `ml_with_fallback` | 0.913 | | 61% | ≈1.0 s | $0.12 |
| `ml` | 0.826 | 0.722 | none | < 1 ms | $0 |

- The trained router recognises data questions perfectly (5 of 5, confidence 0.94 to 0.99). It struggles with "one section or several": it sent 3 of 4 multi-part questions down the single path. Embeddings (`text-embedding-3-large`) did no better on the held-out questions (0.83 to 0.87), so the router stays offline and free.
- The fallback threshold is set per predicted route, at 0.95 out-of-fold precision (about the LLM router's own accuracy), and was fixed before testing. A single global threshold sent 87% of questions to the LLM.
- Routing is a small part of the cost: $0.20 per 1,000 questions, and 1.7 s of a 13 s answer. The LLM router stays the default; `ml` suits latency-bound traffic where a multi-part miss is acceptable. The trained router's held-out scores are part of the evaluation gate.

**The research agent on Foundry Agent Service.** With `AGENT_HOSTING=foundry` the agent's definition (gpt-5-mini, the versioned instructions, the tool declarations, reasoning effort) is a versioned prompt agent in a Foundry project, published from code by `scripts/publish_foundry_agent.py`. Planning runs on Foundry; each tool call comes back to RegIntel, so the guards, evidence, writer, verifier and reviewer are unchanged. On the 16 tool questions:

| | In process | Foundry-hosted |
|---|---|---|
| Grounded answers · all expected facts · expected tools called | 16/16 · 1.000 · 16/16 | 16/16 · 1.000 · 16/16 |
| Tool calls per question | 1.19 | 1.19 |
| Research time p50 / p95 | 4.8 s / 7.9 s | 10.2 s / 12.9 s |
| Model cost per answer (research + writing) | $0.0020 | $0.0024 |

Hosting changed nothing about the answers. It adds versioned agent definitions with a playground in the Foundry portal, and traces in the project through its connected Application Insights. It costs about 5 s of planning time per data question. In-process stays the default.

**Azure AI Evaluation beside RegIntel's own checks** (`tests/evaluation/evaluate_azure_ai.py`, judge gpt-5-mini, 60 golden questions):

| | Result |
|---|---|
| Answered: groundedness · relevance (pass = 3 or more of 5) | 49/49 · 49/49 (means 5.00 · 4.61) |
| Drafts the verifier or reviewer cut (18 statements removed from 13 answers): SDK groundedness of the uncut draft | 5/5 for 12 of 13 |
| Answerable questions RegIntel declined whose passages scored retrieval 5 | 3 of 3 (be-01, sr-04, st-07) |
| Unanswerable questions correctly declined that pass relevance | 4/8 |

- The SDK and RegIntel agree on every answer RegIntel shows. The SDK judges whether a statement is supported in meaning, so a misquote, a number missing from its quote or a "should" turned into "must" still scores 5. The verifier and reviewer are the stricter checks; the SDK is no substitute for them.
- Where the SDK helps is the other direction: high retrieval scores on declined answers point to over-strict declines. All three were the reviewer's subject check declining a question its passages do answer.
- Relevance penalises a correct abstention, so it must not be used to score declines.
