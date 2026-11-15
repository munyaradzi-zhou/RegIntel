"""Create the sections index, upload the embedded chunks, and register it as a knowledge source.

python scripts/azure/build_sections_knowledge_source.py
"""

from __future__ import annotations

import time

from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI, RateLimitError

from regintel.azure_search.knowledge import AzureNames, index_client
from regintel.azure_search.sections_index import (
    build_index,
    delete_stale,
    search_client,
    sections_knowledge_source,
    upload_chunks,
)
from regintel.config import get_settings
from regintel.ingestion.pipeline import read_chunks


def main() -> None:
    settings = get_settings()
    names = AzureNames(settings.azure_name_prefix)
    credential = AzureCliCredential()
    openai = AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint or "",
        azure_ad_token_provider=get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default"),
        api_version="2024-10-21",
    )

    def embed(texts: list[str]) -> list[list[float]]:
        for attempt in range(6):
            try:
                result = openai.embeddings.create(model=settings.azure_openai_embedding_deployment, input=texts)
                return [item.embedding for item in result.data]
            except RateLimitError:
                time.sleep(10 * (attempt + 1))
        raise RuntimeError("embedding kept hitting the rate limit")

    indexes = index_client(settings, credential)
    indexes.create_or_update_index(build_index(settings, names))
    print("index:", names.sections_index)

    chunks = read_chunks(settings.data_dir / "processed" / "chunks.jsonl")
    started = time.time()
    search = search_client(settings, names, credential)
    count = upload_chunks(search, chunks, embed)
    print(f"uploaded {count} chunks in {time.time() - started:.0f} s; removed {delete_stale(search, chunks)} stale")

    indexes.create_or_update_knowledge_source(sections_knowledge_source(names))
    print("knowledge source:", names.sections_ks)


if __name__ == "__main__":
    main()
