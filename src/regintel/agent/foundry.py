"""The research agent hosted on Microsoft Foundry Agent Service (AGENT_HOSTING=foundry).

What moves to Foundry: the agent definition (model deployment, instructions, tool declarations,
reasoning effort) as a versioned *prompt agent* in the project, and the planning loop's model
calls, which then appear in the project's traces and conversation history.

What stays in RegIntel: the tools themselves. Foundry returns each function call to this process,
which runs it against the local evidence pool, so the guards, the evidence numbering, the writer,
the verifier and the reviewer are exactly the in-process ones. An agent version is published from
code (`python scripts/publish_foundry_agent.py`), so the instructions there are the versioned
INSTRUCTIONS in regintel.agent.research, and the version is recorded with every answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from regintel.agent.research import INSTRUCTIONS, INSTRUCTIONS_VERSION, MAX_TOOL_CALLS
from regintel.agent.tools import ResearchTools

if TYPE_CHECKING:
    from agent_framework import FunctionTool


class _DispatchOnly(logging.Filter):
    """The framework warns on every run that local tools are used for dispatch only. That is the
    design here (the declarations are on the published definition), so the warning is dropped."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "tool declarations cannot be sent" not in record.getMessage()


logging.getLogger("agent_framework.foundry").addFilter(_DispatchOnly())


def strict_schema(parameters: dict[str, Any]) -> dict[str, Any]:
    """Foundry's strict function schemas: no extra properties, every property required. Optional
    parameters stay optional by accepting null (pydantic already writes `anyOf [..., null]`)."""
    schema = dict(parameters)
    schema["additionalProperties"] = False
    schema["required"] = list(schema.get("properties", {}))
    for prop in schema.get("properties", {}).values():
        prop.pop("default", None)
    return schema


def all_tools() -> list[FunctionTool]:
    """Every tool the agent may use, for the published definition (sentinels make each one offered)."""
    sentinel: Any = object()
    return ResearchTools(retriever=sentinel, terminology=sentinel, faers=sentinel, openfda=sentinel).as_tools()


async def publish(project_client: Any, agent_name: str, model: str) -> str:
    """Create a new version of the hosted research agent from the code's instructions and tools."""
    from agent_framework import Agent
    from agent_framework_foundry import FoundryChatClient, to_prompt_agent

    local = Agent(
        FoundryChatClient(project_client=project_client, model=model),
        INSTRUCTIONS.format(max_calls=MAX_TOOL_CALLS),
        name=agent_name,
        description=f"RegIntel research step (instructions {INSTRUCTIONS_VERSION}); tools run in the RegIntel API.",
        tools=all_tools(),
        default_options=cast("Any", {"reasoning": {"effort": "low"}}),  # published with the definition
    )
    definition = to_prompt_agent(local)
    for declared in definition.tools or []:
        declared["parameters"] = strict_schema(declared["parameters"])
    version = await project_client.agents.create_version(
        agent_name=agent_name,
        definition=definition,
        metadata={"instructions_version": INSTRUCTIONS_VERSION, "source": "scripts/publish_foundry_agent.py"},
    )
    return str(version.version)


@dataclass(frozen=True)
class HostedAgent:
    project_endpoint: str
    agent_name: str
    agent_version: str
    credential: Any

    def agent(self, tools: list[FunctionTool], max_tool_calls: int = MAX_TOOL_CALLS) -> Any:
        """A client for one question, bound to that question's tools (and so its evidence pool)."""
        from agent_framework_foundry import FoundryAgent

        return FoundryAgent(
            project_endpoint=self.project_endpoint,
            agent_name=self.agent_name,
            agent_version=self.agent_version,
            credential=self.credential,
            tools=tools,
            function_invocation_configuration={"max_function_calls": max_tool_calls},
        )

    @property
    def label(self) -> str:
        return f"foundry:{self.agent_name}@{self.agent_version}"
