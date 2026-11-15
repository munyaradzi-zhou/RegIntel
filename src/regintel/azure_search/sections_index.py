"""The challenger index: RegIntel's section-aware chunks in Azure AI Search.

Same embedding model as the Blob knowledge source (text-embedding-3-large, 3,072 dimensions),
so the comparison isolates one variable: who chunks the documents. Each document keeps the
fields the evaluation and citations need (doc_id, section locator, pages), and the vectorizer
lets the knowledge base embed queries with the search service's own identity.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizer,
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchIndexFieldReference,
    SearchIndexKnowledgeSource,
    SearchIndexKnowledgeSourceParameters,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

from regintel.azure_search.knowledge import EMBEDDING_MODEL, _openai

if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

    from regintel.azure_search.knowledge import AzureNames
    from regintel.config import Settings
    from regintel.ingestion.chunking import Chunk

DIMENSIONS = 3072
SEMANTIC_CONFIG = "sections-semantic"
Embed = Callable[[list[str]], list[list[float]]]


def build_index(settings: Settings, names: AzureNames) -> SearchIndex:
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="jurisdiction", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchableField(name="section", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="section_title", type=SearchFieldDataType.String),
        SimpleField(name="page_start", type=SearchFieldDataType.Int32),
        SimpleField(name="page_end", type=SearchFieldDataType.Int32),
        SearchableField(name="header", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=DIMENSIONS,
            vector_search_profile_name="hnsw-aoai",
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[VectorSearchProfile(name="hnsw-aoai", algorithm_configuration_name="hnsw", vectorizer_name="aoai")],
        vectorizers=[
            AzureOpenAIVectorizer(
                vectorizer_name="aoai",
                parameters=_openai(settings, settings.azure_openai_embedding_deployment, EMBEDDING_MODEL),
            )
        ],
    )
    semantic = SemanticSearch(
        default_configuration_name=SEMANTIC_CONFIG,
        configurations=[
            SemanticConfiguration(
                name=SEMANTIC_CONFIG,
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="header"),
                    content_fields=[SemanticField(field_name="content")],
                    keywords_fields=[SemanticField(field_name="section_title")],
                ),
            )
        ],
    )
    return SearchIndex(
        name=names.sections_index,
        description="RegIntel section-aware chunks of FDA, ICH and EMA/EU guidance, with citation locators.",
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic,
    )


def to_document(chunk: Chunk, vector: list[float]) -> dict[str, Any]:
    return {
        "id": chunk.chunk_id.replace("#", "_"),  # keys allow letters, digits, _ - = only
        "doc_id": chunk.doc_id,
        "jurisdiction": chunk.jurisdiction,
        "section": chunk.section,
        "section_title": chunk.section_title,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "header": chunk.header,
        "content": chunk.text,
        "content_vector": vector,
    }


def batches(items: list[Chunk], size: int) -> Iterator[list[Chunk]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def upload_chunks(search: SearchClient, chunks: list[Chunk], embed: Embed, batch_size: int = 64) -> int:
    uploaded = 0
    for batch in batches(chunks, batch_size):
        vectors = embed([c.search_text for c in batch])
        results = search.upload_documents([to_document(c, v) for c, v in zip(batch, vectors, strict=True)])
        failed = [r.key for r in results if not r.succeeded]
        if failed:
            raise RuntimeError(f"upload failed for {failed[:5]}")
        uploaded += len(batch)
    return uploaded


def delete_stale(search: SearchClient, chunks: list[Chunk]) -> int:
    """Remove documents whose key is no longer a chunk (ids shift when the chunker changes)."""
    current = {c.chunk_id.replace("#", "_") for c in chunks}
    stale = [d["id"] for d in search.search(search_text="*", select=["id"], top=100_000) if d["id"] not in current]
    if stale:
        search.delete_documents([{"id": key} for key in stale])
    return len(stale)


def sections_knowledge_source(names: AzureNames) -> SearchIndexKnowledgeSource:
    return SearchIndexKnowledgeSource(
        name=names.sections_ks,
        description=(
            "FDA, ICH and EMA/EU regulatory guidance, chunked by document section; each passage carries "
            "its section number for citation."
        ),
        search_index_parameters=SearchIndexKnowledgeSourceParameters(
            search_index_name=names.sections_index,
            semantic_configuration_name=SEMANTIC_CONFIG,
            source_data_fields=[
                SearchIndexFieldReference(name=f)
                for f in ("doc_id", "jurisdiction", "section", "section_title", "page_start", "page_end", "header")
            ],
        ),
    )


def search_client(settings: Settings, names: AzureNames, credential: TokenCredential) -> SearchClient:
    if not settings.azure_search_endpoint:
        raise ValueError("AZURE_SEARCH_ENDPOINT is not set")
    return SearchClient(settings.azure_search_endpoint, names.sections_index, credential)
