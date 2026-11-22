"""Build the Azure-backed components from settings (used by scripts/ask.py, the evaluations and the API).

Clients, the chunk store and the data tools are created once and shared by every path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import structlog

from regintel.agent.research import RegIntelAgent, azure_agent_client
from regintel.answering.models import answer_model_from_settings, azure_openai_client
from regintel.answering.pipeline import Answerer
from regintel.answering.review import reviewer_from_settings
from regintel.corpus.manifest import load_manifest
from regintel.faers.query import FaersStore
from regintel.ingestion.pipeline import read_chunks
from regintel.ml_router import FallbackRouter, MLRouter, load_model
from regintel.openfda.client import OpenFDAClient
from regintel.router import Router, Routes
from regintel.service import RegIntel
from regintel.terminology.cdisc import load_terminology

if TYPE_CHECKING:
    from regintel.answering.models import AnswerModel
    from regintel.answering.review import AnswerReviewer
    from regintel.azure_search.kb_retriever import Effort, KnowledgeBaseRetriever
    from regintel.config import Settings
    from regintel.corpus.manifest import Manifest
    from regintel.ingestion.chunking import Chunk


log = structlog.get_logger("regintel.factory")


@dataclass
class AzureComponents:
    settings: Settings
    tokens: Any  # Entra ID bearer token provider for Azure OpenAI
    search: Any  # KnowledgeBaseRetrievalClient for the sections knowledge base
    chunks: list[Chunk]
    manifest: Manifest
    writer: AnswerModel
    reviewer: AnswerReviewer | None

    @classmethod
    def connect(cls, settings: Settings) -> AzureComponents:
        from azure.identity import get_bearer_token_provider
        from azure.search.documents.knowledgebases import KnowledgeBaseRetrievalClient

        from regintel.azure_search.knowledge import AzureNames
        from regintel.runtime import azure_credential

        credential = azure_credential()
        tokens = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
        search = KnowledgeBaseRetrievalClient(
            settings.azure_search_endpoint or "",
            credential,
            knowledge_base_name=AzureNames(settings.azure_name_prefix).knowledge_base("sections"),
            api_version=settings.azure_search_api_version,
        )
        return cls(
            settings,
            tokens,
            search,
            read_chunks(settings.data_dir / "processed" / "chunks.jsonl"),
            load_manifest(settings.data_dir / "manifest.json"),
            answer_model_from_settings(settings, tokens),
            reviewer_from_settings(settings, tokens),
        )

    def retriever(self, effort: Effort) -> KnowledgeBaseRetriever:
        from regintel.azure_search.kb_retriever import KnowledgeBaseRetriever

        return KnowledgeBaseRetriever(f"kb-sections-{effort}", self.search, self.chunks, effort)

    def web(self) -> KnowledgeBaseRetriever:
        """The web-only knowledge base (regulator domains), for pointers when the guidance cannot answer."""
        from azure.search.documents.knowledgebases import KnowledgeBaseRetrievalClient

        from regintel.azure_search.kb_retriever import KnowledgeBaseRetriever
        from regintel.azure_search.knowledge import AzureNames
        from regintel.runtime import azure_credential

        client = KnowledgeBaseRetrievalClient(
            self.settings.azure_search_endpoint or "",
            azure_credential(),
            knowledge_base_name=AzureNames(self.settings.azure_name_prefix).knowledge_base("web"),
            api_version=self.settings.azure_search_api_version,
        )
        return KnowledgeBaseRetriever("kb-web", client, [], "minimal")

    @cached_property
    def openfda(self) -> OpenFDAClient:
        """One client (and cache) shared by every question; an API key raises the daily limit."""
        key = self.settings.openfda_api_key
        return OpenFDAClient(key.get_secret_value() if key else None)

    def agent(self) -> RegIntelAgent:
        """Guidance search runs at `minimal` effort: the agent itself splits the question into sub-questions."""
        cdisc_dir = self.settings.data_dir / "raw" / "cdisc"
        fetched = json.loads((cdisc_dir / "LOCK.json").read_text(encoding="utf-8"))["fetched_at"][:10]
        return RegIntelAgent(
            azure_agent_client(self.settings, self.tokens),
            self.writer,
            self.retriever("minimal"),
            terminology=load_terminology(cdisc_dir),
            terminology_version=f"NCI EVS, fetched {fetched}",
            faers=FaersStore(self.settings.data_dir / "processed" / "faers"),
            openfda=self.openfda,
            manifest=self.manifest,
            reviewer=self.reviewer,
        )

    def router(self) -> Routes:
        """ROUTING picks the router; the trained one is loaded by version and its checksum logged."""
        llm = Router(azure_openai_client(self.settings, self.tokens), self.settings.azure_openai_chat_deployment)
        if self.settings.routing == "llm":
            return llm
        model = load_model(self.settings.router_model_version)
        log.info("router_model", version=model.version, sha256=model.sha256[:12], routing=self.settings.routing)
        ml = MLRouter(model)
        return ml if self.settings.routing == "ml" else FallbackRouter(ml, llm)

    def service(self) -> RegIntel:
        return RegIntel(
            self.router(),
            Answerer(self.retriever("low"), self.writer, self.manifest, top_k=8, reviewer=self.reviewer),
            Answerer(self.retriever("medium"), self.writer, self.manifest, top_k=8, reviewer=self.reviewer),
            self.agent(),
            web=self.web(),
        )


def build_azure_agent(settings: Settings) -> RegIntelAgent:
    return AzureComponents.connect(settings).agent()


def build_azure_service(settings: Settings) -> RegIntel:
    return AzureComponents.connect(settings).service()
