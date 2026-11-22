# RegIntel

[![CI](https://github.com/munyaradzi-zhou/RegIntel/actions/workflows/ci.yml/badge.svg)](https://github.com/munyaradzi-zhou/RegIntel/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-14130C)
![Azure](https://img.shields.io/badge/Azure-AI%20Search%20%C2%B7%20OpenAI%20%C2%B7%20Container%20Apps-A14C3F)
[![License: MIT](https://img.shields.io/badge/license-MIT-5F594A)](LICENSE)

**Regulatory question answering for biopharma, with every quote and number checked against the source.**
RegIntel answers questions about FDA, ICH and EMA guidance, CDISC terminology, FAERS adverse-event data, and US drug labels and recalls (live from openFDA). A router sends each question to a retrieval pipeline or a tool-using research agent; the answer is then verified in code: a quote that is not verbatim in a retrieved passage, or a number that is not in a quote, is removed before the user sees it. When nothing survives, RegIntel declines instead of guessing.

![RegIntel answering a FAERS safety-signal question, every number quoted from the computed record](assets/ask-answer.png)

---

## What it does

| | |
|---|---|
| **Agentic retrieval** | Azure AI Search knowledge bases over 1,869 section-aware chunks of 23 public guidances (FDA, ICH, EMA, European Commission), each pinned by sha256. The planner decomposes multi-part questions into sub-queries; reasoning effort is chosen per question shape (`low` for single lookups, `medium` for comparisons), a choice made from measurements, not defaults. |
| **Grounding enforced in code** | The writer (`gpt-5-mini`, structured output) returns statements, each with verbatim quotes. A verifier checks every quote against the passage text and every number, by value, against the quotes. Unverified statements are dropped; an answer with none left is withheld. |
| **Independent reviewer** | `gpt-oss-120b`, a different model family from the writer, judges each verified statement as supported, overreach or unsupported. It catches what string matching cannot: a "should" written as "must", or a general rule stretched to a subject the source never mentions. |
| **Research agent with data tools** | A Microsoft Agent Framework agent for questions that need data: CDISC Controlled Terminology lookup, FAERS safety signals over 36,000 reports in DuckDB (counts and reporting odds ratios with 95% CI, computed by parameterised queries, never written by the model), and live openFDA drug label sections and recalls, verified like any passage. |
| **Guards** | Input screens for instruction overrides and prompt extraction, and a passage screen so retrieved text carrying instructions never reaches the writer. Declined questions get up to three pages from regulator websites, clearly marked as unverified pointers. |
| **GxP-style audit trail** | Every answer is recorded before it is released: who asked (key id), the route, each source with its sha256, models, prompt versions, the raw draft and every verifier and reviewer decision. Records are hash-chained in Cosmos DB; `GET /v1/audit/verify` finds the first edited or removed record, and `scripts/replay.py` re-verifies any answer. A [validation pack](docs/validation/) traces each requirement to its tests. |
| **Production plumbing** | FastAPI with SSE streaming, API keys and rate limits, follow-up questions in Cosmos DB sessions, OpenTelemetry to Application Insights, an MCP server, and a Streamlit UI. |

## Architecture

![RegIntel architecture](assets/architecture.png)

```mermaid
graph LR
    subgraph Clients
        UI["Streamlit UI"]
        MCP["MCP clients<br/>Claude · Copilot · Foundry"]
    end
    subgraph "Azure Container Apps"
        API["FastAPI<br/>keys · rate limit · SSE · tracing"]
        GUARD["Input guards"]
        ROUTER["Router"]
        PIPE["Retrieval pipeline<br/>single · multi-part"]
        AGENT["Research agent<br/>Agent Framework + tools"]
        VERIFY["Verifier → reviewer"]
    end
    subgraph "Azure AI"
        KB["AI Search<br/>knowledge bases"]
        OAI["Azure OpenAI<br/>gpt-5-mini · embeddings"]
        OSS["AI Services<br/>gpt-oss-120b"]
    end
    subgraph Data
        DUCK["FAERS · DuckDB"]
        CT["CDISC CT"]
        FDA["openFDA (live)<br/>labels · recalls"]
        COSMOS["Cosmos DB<br/>sessions"]
    end
    UI --> API
    MCP --> API
    API --> GUARD --> ROUTER
    ROUTER --> PIPE --> KB
    ROUTER --> AGENT
    AGENT --> KB
    AGENT --> DUCK
    AGENT --> CT
    AGENT --> FDA
    PIPE --> VERIFY
    AGENT --> VERIFY
    VERIFY --> OSS
    PIPE --> OAI
    AGENT --> OAI
    API --> COSMOS
```

Every service-to-service call uses Microsoft Entra ID through a user-assigned managed identity. The only secrets are the API client keys, which are generated at deploy time, stored in Key Vault and never printed.

| Azure service | Used for |
|---|---|
| Azure AI Search | Knowledge bases with agentic retrieval over the section index; a web source restricted to regulator domains for fallback pointers |
| Azure OpenAI | `gpt-5-mini` (router, planner, agent, writer), `text-embedding-3-large` |
| Azure AI Services | `gpt-oss-120b` answer reviewer |
| Container Apps + Container Registry | API and UI, scale to zero |
| Cosmos DB (serverless) | Conversation sessions, 7-day TTL |
| Key Vault · Log Analytics · Application Insights | API keys, logs, traces |

## Running on Azure

Deployed with Bicep and measured over HTTPS against the live API: 60 golden questions (8 deliberately unanswerable) and 16 data questions (FAERS, CDISC, openFDA labels and recalls), sent one at a time as a client would.

| Measured on the live deployment | |
|---|---|
| Answerable questions answered | 64 / 68 |
| Unanswerable questions declined | 8 / 8 |
| Expected facts in answers | 102 / 114 |
| Expected source document cited | 50 / 52 |
| Data questions routed to the agent / answered with tool evidence | 15 / 16 · 14 / 16 |
| Latency p50 / p95 | 13.4 s / 20.3 s (single-document 12.1 / 17.9 s, multi-part 18.2 / 23.6 s, agent 12.3 / 17.8 s) |

The models are not deterministic: an earlier run the same day, before one fix, scored 109 / 114 facts and 16 / 16 data questions routed. Single questions flip between runs in both directions, so differences of a few facts are noise.

![FAERS safety signals with reporting odds ratios](assets/safety-signals.png)

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

## Run it locally

Requirements: Python 3.11+, `az login` to a subscription with the Azure resources (or run `scripts/azure/deploy.sh` first).

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make dev                                             # pip install -e ".[dev,local,azure,serve,mcp,frontend]"
cp .env.example .env                                 # set the Azure endpoints

make corpus ingest  # download the 23 guidances (checksums verified) and build the section chunks
python -m regintel.faers.build   # FAERS JSON → Parquet safety tables
make run            # API on http://localhost:8000 (Swagger UI at /docs)
make run-ui         # UI  on http://localhost:8501
```

`make docker-up` runs the API and UI in containers. `make corpus` also downloads CDISC terminology and an openFDA FAERS extract (2025 Q1), which `python -m regintel.faers.build` converts to Parquet for DuckDB.

## Deploy to Azure

```bash
az login
scripts/azure/deploy.sh
```

[`scripts/azure/deploy.sh`](scripts/azure/deploy.sh) deploys the Bicep templates in [`infra/`](infra) (monitoring, managed identity, registry, Key Vault, Cosmos DB, role grants, Container Apps), generates the API keys straight into Key Vault, builds and pushes both images, rolls the apps onto them and runs [`smoke_test.sh`](scripts/azure/smoke_test.sh): health, readiness, a 401 without a key, and one answered question with its source.

**Continuous delivery.** After CI passes on `main`, [`deploy.yml`](.github/workflows/deploy.yml) builds the commit, rolls out a new revision of each app and smoke-tests it. It signs in with GitHub OIDC as an identity that can only push images, update the two apps and read the smoke-test key; no Azure credential is stored in GitHub. The runtime data is not in git, so the build pulls it from a content-addressed data image in the registry.

## API

```bash
# Ask a question (pass session_id back for follow-ups; /v1/ask/stream for Server-Sent Events)
curl -X POST localhost:8000/v1/ask -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"question": "What is the minimum number of subjects in an EMA bioequivalence study?"}'

# FAERS profile of a drug: counts and top reactions with reporting odds ratios
curl "localhost:8000/v1/faers/profile?drug=atorvastatin" -H "X-API-Key: $KEY"

# CDISC Controlled Terminology lookup
curl "localhost:8000/v1/cdisc?term=MILLIGRAM" -H "X-API-Key: $KEY"
```

Also `GET /v1/audit/verify` (recompute the audit trail's hash chain), `GET /v1/sessions/{id}`, `GET /healthz`, `GET /readyz`. Every answer carries a grounding label (`grounded`, `partially_grounded`, `not_grounded`, `declined`), its sources with the verbatim quotes, the path taken, per-stage timings and the version and text hash of each prompt that produced it (evaluation results record the same), and the id of its audit record (`python scripts/replay.py <audit_id>`). Full schemas at `/docs`; more in [`docs/api-and-mcp.md`](docs/api-and-mcp.md).

## Use it from other agents (MCP)

RegIntel is also a [Model Context Protocol](https://modelcontextprotocol.io) server, so Claude Code, Claude Desktop, GitHub Copilot or a Foundry agent can use it as a tool. All six tools are read-only; the two openFDA tools are the only ones that call out of the process.

| Tool | Does |
|---|---|
| `ask_regulatory_question` | Verified answer with sources; in-process or against a deployed API (`REGINTEL_API_URL`) |
| `lookup_cdisc_term` | Exact CDISC Controlled Terminology lookup |
| `faers_drug_summary` | Report counts and most frequent reactions for a drug |
| `faers_disproportionality` | Reporting odds ratio with 95% CI for a drug–reaction pair |
| `drug_label_section` | A section of a drug's current US label (boxed warning, contraindications, ...), live from openFDA, with set id, version and effective date |
| `recent_recalls` | FDA enforcement reports (recalls) for a drug in a date range, live from openFDA |

```bash
claude mcp add regintel -- python -m regintel.mcp_server
```

## Quality

`make check` runs the same gates as CI:

- **Ruff** lint and formatting, **mypy `--strict`** over the whole package
- **247 unit tests** with an 85% coverage gate: verifier, reviewer, router, agent loop (a fake agent calling the real tools), guards, chunking and glyph repair, FAERS statistics, the openFDA client against recorded responses (429, 5xx, timeouts, cache, an unreachable API adds no evidence), sessions and follow-up rewriting, the audit trail (hash chain, tampering, a Cosmos fake with ETags under contention, fail-closed recording, replay), the validation pack's links, API auth and rate limits, SSE, UI escaping
- **Bandit**, **pip-audit** and **gitleaks** over the full history
- Both **Docker** images built, and the API image started with no data or credentials (it must stay up and report why it is not ready), **Bicep** build, **shellcheck**

**Evaluation gate on every pull request** (`make eval-ci`, free, no model calls). It scores BM25 retrieval on the golden set over the committed chunk fixture, scores the trained router on its held-out questions, replays 114 recorded and deliberately mutated drafts through the verifier (each statement must be kept or dropped exactly as recorded), and checks that every prompt whose text changed has a new version. Results are compared with [`ci_baseline.json`](tests/evaluation/baselines/ci_baseline.json): a metric that drops more than 0.02 fails the build, and the report is posted on the pull request. `make eval-ci ARGS=--update-baseline` moves the bar deliberately, in the same pull request as the improvement.

The other evaluations call real models, so they run on demand: `make eval-retrieval`, `eval-answers`, `eval-agent`, `eval-injection`, `eval-review`, `eval-followups`, `eval-web`, and `python -m tests.evaluation.evaluate_live --url <api>` against a deployment.

## Project layout

```
src/regintel/
  corpus/         manifest, checksum-verified download, FAERS JSON → Parquet
  ingestion/      section-aware chunking, glyph repair, footnote handling, chunk store
  retrieval/      local BM25 / dense / hybrid baselines
  azure_search/   section index, knowledge sources and knowledge bases
  answering/      writer prompt, verifier, reviewer, answer schema
  agent/          research agent, its tools, and Foundry Agent Service hosting
  faers/ terminology/   DuckDB safety-signal queries, CDISC lookup
  openfda/        live openFDA client (timeouts, retries, TTL cache) and label / recall lookups
  audit/          hash-chained audit records, Cosmos store, chain verification, replay
  guard/          question and passage screens
  router.py service.py conversation.py   routing, end-to-end service, follow-ups
  ml_router.py router_models/   trained router (TF-IDF + logistic regression) and its versioned JSON models
  api/            FastAPI app, models, sessions in Cosmos DB
  frontend/       Streamlit UI
  mcp_server.py   MCP tools
infra/            Bicep: main.bicep, modules/, parameters/
scripts/          corpus download, Azure deploy and smoke test, UI screenshots
tests/            unit tests and evaluation suites with their datasets
```

## Write-up

A five-part series on what building and measuring RegIntel taught me: [RegIntel: regulatory answers you can check](https://munyaradzi-zhou.github.io/portfolio/blog/regintel/part1-a-citation-is-a-contract.html), and the [case study](https://munyaradzi-zhou.github.io/portfolio/work/regintel.html).

## About this project

I built RegIntel to show an end-to-end, evaluation-driven RAG and agent system in a regulated domain, where a confident wrong answer costs more than no answer. Every design decision above came from a measurement, and the measurements that went the wrong way are kept in the story.

The corpus is public guidance from FDA, ICH, EMA and the European Commission, plus CDISC terminology and openFDA FAERS data. FAERS reports are spontaneous and cannot establish that a drug caused a reaction.

RegIntel is a demonstration, not medical or regulatory advice. Its answers can be wrong and must be checked against the source documents by a qualified professional.

## License

[MIT](LICENSE) © Munyaradzi Zhou
