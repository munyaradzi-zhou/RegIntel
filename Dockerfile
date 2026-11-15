# syntax=docker/dockerfile:1
# =============================================================================
# RegIntel API — multi-stage image
#   builder: install the package and the server extras into /app/venv
#   data:    the runtime data, from this build context (default) or from a data image in the
#            registry (CD runners do not have the git-ignored data):
#              docker build --target data-local -t <registry>/regintel-data:<tag> .   (push once)
#              docker build --build-arg DATA_SOURCE=<registry>/regintel-data:<tag> ...
#   runtime: slim, non-root, package + runtime data, uvicorn on :8000
# =============================================================================
ARG DATA_SOURCE=data-local

FROM scratch AS data-local
COPY data/ /data/

FROM ${DATA_SOURCE} AS data

FROM python:3.13-slim AS builder
# uv installs faster than pip and copes with HTTPS-scanning proxies that break pip's downloads.
COPY --from=ghcr.io/astral-sh/uv:0.7.2 /uv /bin/uv
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
# Behind a TLS-intercepting proxy or antivirus, pass its CA bundle:
#   docker build --secret id=ca_cert,src=<bundle.pem> ...   (used for the install only, not kept)
RUN --mount=type=secret,id=ca_cert,required=false \
    if [ -f /run/secrets/ca_cert ]; then export SSL_CERT_FILE=/run/secrets/ca_cert; fi && \
    uv venv /app/venv && \
    VIRTUAL_ENV=/app/venv uv pip install --no-cache ".[serve,mcp]"

FROM python:3.13-slim AS runtime
RUN groupadd -r regintel && useradd -r -g regintel -d /app -s /sbin/nologin regintel
WORKDIR /app
COPY --from=builder /app/venv /app/venv
# Runtime data. A CI build without the (git-ignored) data still produces a working image: the API
# starts, /healthz answers and /readyz reports what is missing.
COPY --from=data /data/ ./data/

ENV PATH="/app/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/app/data \
    HOST=0.0.0.0 \
    PORT=8000 \
    OTEL_SERVICE_NAME=regintel-api

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/healthz').raise_for_status()"
USER regintel
CMD ["regintel"]
