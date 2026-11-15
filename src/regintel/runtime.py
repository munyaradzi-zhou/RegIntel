"""Process setup shared by the entry points (API server, MCP server)."""

from __future__ import annotations

import logging
import sys
from functools import cache
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential


def use_os_certificates() -> None:
    """Verify TLS with the operating system's certificate store when `truststore` is installed.
    Needed on machines behind TLS-inspecting proxies or with a certificate store Python's bundle
    lacks (this development machine); a no-op in containers, where the bundle is enough."""
    try:
        import truststore
    except ImportError:
        return
    truststore.inject_into_ssl()


def log_to_stderr() -> None:
    """For stdio transports (MCP): stdout carries the protocol, so every log line goes to stderr."""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, force=True)
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))


@cache
def azure_credential() -> TokenCredential:
    """One Entra ID credential for the process: managed identity in Azure, environment variables
    (a service principal) in docker compose, the developer's `az login` on a workstation.
    Interactive browser login is excluded so a server never blocks waiting for a person."""
    from azure.identity import DefaultAzureCredential

    return DefaultAzureCredential(exclude_interactive_browser_credential=True)


def configure_telemetry(connection_string: str | None, service_name: str = "regintel-api") -> bool:
    """Send traces, metrics and logs to Application Insights when a connection string is set
    (in Azure; locally nothing is exported). The distro instruments FastAPI, httpx and requests,
    so every answer shows up as one request with its model and search calls underneath."""
    if not connection_string:
        return False
    import os

    from azure.monitor.opentelemetry import configure_azure_monitor

    os.environ.setdefault("OTEL_SERVICE_NAME", service_name)
    configure_azure_monitor(connection_string=connection_string)
    return True
