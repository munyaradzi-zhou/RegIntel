"""Azure AI Search agentic retrieval objects: knowledge sources and the knowledge base.

Everything RegIntel creates on the (shared) search service is named `<prefix>-...`, so it never
collides with other projects' objects and can be removed as a set.

Knowledge sources:
    <prefix>-blob-ks      Azure Blob: the guidance PDFs. Azure AI Search generates the whole
                          ingestion pipeline (data source, skillset, index, indexer) and chunks
                          and embeds the text itself ("integrated vectorization").
    <prefix>-sections-ks  Search index: the same PDFs chunked by regintel.ingestion
                          (section-aware, with citation locators). Built by sections_index.py.
    <prefix>-web-ks       Web (Bing), restricted to regulator domains. A remote source: nothing
                          is indexed, and results are LLM *summaries* of web pages, not verbatim
                          text, so they can point to newer guidance but never be quoted as evidence.

Authentication is keyless throughout: the search service's managed identity reads the blobs
and calls Azure OpenAI; this code authenticates with the developer's Entra ID token.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    AzureBlobKnowledgeSource,
    AzureBlobKnowledgeSourceParameters,
    AzureOpenAIVectorizerParameters,
    KnowledgeBase,
    KnowledgeBaseAzureOpenAIModel,
    KnowledgeSourceContentExtractionMode,
    KnowledgeSourceReference,
    WebKnowledgeSource,
    WebKnowledgeSourceDomain,
    WebKnowledgeSourceDomains,
    WebKnowledgeSourceParameters,
)
from azure.search.documents.knowledgebases.models import (
    KnowledgeRetrievalLowReasoningEffort,
    KnowledgeRetrievalOutputMode,
    KnowledgeSourceAzureOpenAIVectorizer,
    KnowledgeSourceIngestionParameters,
)

if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

    from regintel.config import Settings

EMBEDDING_MODEL = "text-embedding-3-large"
# Primary regulator sources only: guidance pages, the CFR and ICH guidelines.
REGULATOR_DOMAINS = ("fda.gov", "ema.europa.eu", "ich.org", "ecfr.gov", "health.ec.europa.eu")


@dataclass(frozen=True)
class AzureNames:
    prefix: str

    @property
    def blob_ks(self) -> str:
        return f"{self.prefix}-blob-ks"

    @property
    def sections_index(self) -> str:
        return f"{self.prefix}-sections"

    @property
    def sections_ks(self) -> str:
        return f"{self.prefix}-sections-ks"

    @property
    def web_ks(self) -> str:
        return f"{self.prefix}-web-ks"

    def knowledge_base(self, variant: str) -> str:
        return f"{self.prefix}-kb-{variant}"


def index_client(settings: Settings, credential: TokenCredential) -> SearchIndexClient:
    if not settings.azure_search_endpoint:
        raise ValueError("AZURE_SEARCH_ENDPOINT is not set")
    return SearchIndexClient(settings.azure_search_endpoint, credential, api_version=settings.azure_search_api_version)


def _openai(settings: Settings, deployment: str, model: str) -> AzureOpenAIVectorizerParameters:
    if not settings.azure_openai_endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT is not set")
    return AzureOpenAIVectorizerParameters(
        resource_url=settings.azure_openai_endpoint, deployment_name=deployment, model_name=model
    )


def storage_resource_id(subscription_id: str, resource_group: str, account: str) -> str:
    """Keyless blob access: a ResourceId 'connection string' makes the search identity authenticate."""
    return (
        f"ResourceId=/subscriptions/{subscription_id}/resourceGroups/{resource_group}"
        f"/providers/Microsoft.Storage/storageAccounts/{account};"
    )


def blob_knowledge_source(settings: Settings, names: AzureNames, connection: str) -> AzureBlobKnowledgeSource:
    ingestion: Any = KnowledgeSourceIngestionParameters(
        identity=None,  # None = the search service's system-assigned identity
        disable_image_verbalization=True,  # text-only corpus; skip paying an LLM to describe images
        embedding_model=KnowledgeSourceAzureOpenAIVectorizer(
            azure_open_ai_parameters=_openai(settings, settings.azure_openai_embedding_deployment, EMBEDDING_MODEL)
        ),
        content_extraction_mode=KnowledgeSourceContentExtractionMode.MINIMAL,
        ingestion_schedule=None,  # the corpus is pinned by checksum: ingest once, on demand
    )
    return AzureBlobKnowledgeSource(
        name=names.blob_ks,
        description=(
            "FDA, ICH and EMA/EU regulatory guidance documents (final versions) on clinical trial design, "
            "statistics, safety reporting, bioequivalence, good clinical practice and electronic records."
        ),
        azure_blob_parameters=AzureBlobKnowledgeSourceParameters(
            connection_string=connection,
            container_name=settings.azure_storage_container,
            folder_path=None,
            ingestion_parameters=ingestion,
        ),
    )


def web_knowledge_source(names: AzureNames, domains: tuple[str, ...] = REGULATOR_DOMAINS) -> WebKnowledgeSource:
    return WebKnowledgeSource(
        name=names.web_ks,
        description="Current guidance pages from regulators (FDA, EMA, European Commission, ICH, eCFR).",
        web_parameters=WebKnowledgeSourceParameters(
            domains=WebKnowledgeSourceDomains(
                allowed_domains=[WebKnowledgeSourceDomain(address=d, include_subpages=True) for d in domains]
            )
        ),
    )


def knowledge_base(settings: Settings, names: AzureNames, variant: str, knowledge_sources: list[str]) -> KnowledgeBase:
    """Extractive retrieval with LLM query planning. Answer synthesis stays off on purpose:
    RegIntel's own agent writes the answer and its grounding verifier checks every quote."""
    model = KnowledgeBaseAzureOpenAIModel(
        azure_open_ai_parameters=_openai(settings, settings.azure_openai_chat_deployment, "gpt-5-mini")
    )
    instructions = (
        "Questions comparing jurisdictions (FDA vs EMA/EU vs ICH) need passages from each jurisdiction named. "
        "Prefer the most specific section over introductions, tables of contents and reference lists."
    )
    if names.web_ks in knowledge_sources:
        instructions += (
            f" Search {names.web_ks} only when the question concerns guidance that the indexed guidance does not"
            " cover, or asks what is new or current."
        )
    return KnowledgeBase(
        name=names.knowledge_base(variant),
        description="Regulatory guidance for biopharma: FDA, ICH and EMA/EU.",
        retrieval_instructions=instructions,
        output_mode=KnowledgeRetrievalOutputMode.EXTRACTIVE_DATA,
        knowledge_sources=[KnowledgeSourceReference(name=ks) for ks in knowledge_sources],
        models=[model],
        retrieval_reasoning_effort=KnowledgeRetrievalLowReasoningEffort(),
    )
