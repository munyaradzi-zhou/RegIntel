# Using RegIntel: HTTP API and MCP server

Both run the same service: input guard → follow-up rewriter → router → fixed pipeline or research
agent → quote and number verifier → answer reviewer → (on a decline) regulator-site pointers.
Local runs need `az login` (Azure AI Search, Azure OpenAI and the reviewer use Entra ID, no keys).

## HTTP API

```bash
regintel                          # or: python -m regintel.main   (HOST, PORT; default 127.0.0.1:8000)
```

| Method and path | What it does |
|---|---|
| `POST /v1/ask` | `{"question": "...", "session_id": "optional"}` → one answer as JSON |
| `POST /v1/ask/stream` | the same as server-sent events: `session`, `progress` (every 1.5 s), `answer`, `done` (or `error`) |
| `GET /v1/sessions/{id}` | the turns of a conversation: question as typed, the standalone question answered, label, answer |
| `GET /healthz` | the process is up |
| `GET /readyz` | the service is built and takes questions (503 while starting) |

An answer carries `label` (`grounded`, `partially_grounded`, `not_grounded`, `declined`, `blocked`),
`statements` (each with the numbers of its `sources`), `sources` (label, verbatim quote, document,
section, pages), `dropped` (statements removed by verification or review), `web_pointers` (only on
declines, unverified), `path`, `tool_calls`, `timings` and a disclaimer.

Follow-ups: send the `session_id` from the first answer. "And for narrow therapeutic index drugs?"
is rewritten into a standalone question from the earlier *questions* (never the answers, so no
answer text leaks into the next retrieval) and answered from fresh evidence.

Security: set `API_KEYS` (comma-separated) to require `X-API-Key`; keys are compared in constant
time and clients are rate limited per key (or per address when open), `RATE_LIMIT_PER_MINUTE`
(default 30). Every response has an `X-Request-ID`, also in the structured log.

## MCP server (Claude Desktop, IDEs, agents)

Tools, all read-only: `ask_regulatory_question` (with `session_id` for follow-ups),
`lookup_cdisc_term`, `faers_drug_summary`, `faers_disproportionality`.

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "regintel": {
      "command": "F:\\AI Ideas\\PhD Application\\Regulatory QandA System for Biopharma\\.venv\\Scripts\\python.exe",
      "args": ["-m", "regintel.mcp_server"],
      "cwd": "F:\\AI Ideas\\PhD Application\\Regulatory QandA System for Biopharma"
    }
  }
}
```

With `REGINTEL_API_URL` (and `REGINTEL_API_KEY`) set in that entry's `env`, `ask_regulatory_question`
calls the deployed API instead of running the pipeline in-process; the CDISC and FAERS tools
always run locally. The server writes logs to stderr: stdout carries the MCP protocol.
