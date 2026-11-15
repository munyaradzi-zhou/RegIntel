"""Create the knowledge base variants and run one retrieval to show the response shape.

    python scripts/azure/build_knowledge_bases.py
    python scripts/azure/build_knowledge_bases.py --query "EMA bioequivalence acceptance range" --variant sections

Variants (one knowledge source each, so the evaluation compares chunking strategies cleanly):
    regintel-kb-auto       Blob knowledge source (Azure's integrated chunking)
    regintel-kb-sections   section-index knowledge source (RegIntel's chunking)
    regintel-kb-sections-web  the section index plus the Web (Bing) source on regulator domains
    regintel-kb-web        the Web source alone: pointers offered when the guidance cannot answer

Creating the Web knowledge source sends queries to Bing, outside the Azure compliance boundary,
and is billed separately ("Grounding with Bing"). The queries are regulatory questions, no PII.
"""

from __future__ import annotations

import argparse
import json

from azure.identity import AzureCliCredential
from azure.search.documents.knowledgebases import KnowledgeBaseRetrievalClient
from azure.search.documents.knowledgebases.models import (
    KnowledgeBaseMessage,
    KnowledgeBaseMessageTextContent,
    KnowledgeBaseRetrievalRequest,
)

from regintel.azure_search.knowledge import AzureNames, index_client, knowledge_base, web_knowledge_source
from regintel.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query", default="What acceptance range must the 90% confidence interval meet for EU bioequivalence?"
    )
    parser.add_argument("--variant", choices=["auto", "sections", "sections-web", "web"], default="sections")
    args = parser.parse_args()

    settings = get_settings()
    names = AzureNames(settings.azure_name_prefix)
    credential = AzureCliCredential()
    indexes = index_client(settings, credential)
    indexes.create_or_update_knowledge_source(web_knowledge_source(names))
    print("knowledge source:", names.web_ks)
    variants = {
        "auto": [names.blob_ks],
        "sections": [names.sections_ks],
        "sections-web": [names.sections_ks, names.web_ks],
        "web": [names.web_ks],  # fallback pointers when the guidance has no answer
    }
    for variant, sources in variants.items():
        kb = indexes.create_or_update_knowledge_base(knowledge_base(settings, names, variant, sources))
        print("knowledge base:", kb.name)

    client = KnowledgeBaseRetrievalClient(
        settings.azure_search_endpoint or "",
        credential,
        knowledge_base_name=names.knowledge_base(args.variant),
        api_version=settings.azure_search_api_version,
    )
    request = KnowledgeBaseRetrievalRequest(
        messages=[KnowledgeBaseMessage(role="user", content=[KnowledgeBaseMessageTextContent(text=args.query)])],
        include_activity=True,
    )
    response = client.retrieve(request).as_dict()
    print(json.dumps(response, indent=2, default=str)[:6000])


if __name__ == "__main__":
    main()
