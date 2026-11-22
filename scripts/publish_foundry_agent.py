"""Publish a new version of the hosted research agent to the Foundry project.

    python scripts/publish_foundry_agent.py            # FOUNDRY_PROJECT_ENDPOINT from .env, az login
    python scripts/publish_foundry_agent.py --list     # versions published so far

The definition comes from the code: the versioned INSTRUCTIONS and the tool declarations of
regintel.agent.research and regintel.agent.tools, on the gpt-5-mini deployment of the project's
account. Set FOUNDRY_AGENT_VERSION to the printed version to use it (AGENT_HOSTING=foundry).
"""

from __future__ import annotations

import argparse
import asyncio
import warnings

from regintel.agent.foundry import publish
from regintel.config import get_settings

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--model", default=None, help="deployment on the project's account (default: chat deployment)")
    args = parser.parse_args()
    from azure.ai.projects.aio import AIProjectClient
    from azure.identity.aio import AzureCliCredential

    settings = get_settings()
    if not settings.foundry_project_endpoint:
        raise SystemExit("set FOUNDRY_PROJECT_ENDPOINT")
    async with (
        AzureCliCredential() as credential,
        AIProjectClient(endpoint=settings.foundry_project_endpoint, credential=credential) as project,
    ):
        if args.list:
            async for v in project.agents.list_versions(agent_name=settings.foundry_agent_name):
                print(v.version, v.metadata)
            return
        warnings.filterwarnings("ignore", message=".*experimental.*")
        version = await publish(
            project, settings.foundry_agent_name, args.model or settings.azure_openai_chat_deployment
        )
        print(f"published {settings.foundry_agent_name} version {version}: set FOUNDRY_AGENT_VERSION={version}")


if __name__ == "__main__":
    asyncio.run(main())
