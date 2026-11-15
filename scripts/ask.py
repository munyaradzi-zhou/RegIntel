"""Ask one question end to end and print the verified answer with its sources.

    python scripts/ask.py "Which acceptance range applies to the 90% CI for EU bioequivalence?"
    python scripts/ask.py --effort medium "Compare FDA and EMA ... "
    python scripts/ask.py --local "..."          # BM25 + OpenAI key, no Azure

Needs `az login` for the Azure path (knowledge base retrieval and gpt-5-mini via Entra ID).
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from regintel.answering.models import answer_model_from_settings
from regintel.answering.pipeline import Answerer
from regintel.config import get_settings
from regintel.corpus.manifest import load_manifest
from regintel.ingestion.pipeline import read_chunks

if TYPE_CHECKING:
    from regintel.answering.schema import AnswerResult
    from regintel.config import Settings

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--effort", choices=["minimal", "low", "medium"], default="minimal")
    parser.add_argument("--local", action="store_true", help="BM25 retrieval and the OpenAI API")
    parser.add_argument("--agent", action="store_true", help="research agent with guidance, CDISC and FAERS tools")
    args = parser.parse_args()

    settings = get_settings()
    chunks = read_chunks(settings.data_dir / "processed" / "chunks.jsonl")
    manifest = load_manifest(settings.data_dir / "manifest.json")
    if args.local:
        from regintel.retrieval.bm25 import BM25Retriever

        answerer = Answerer(BM25Retriever(chunks), answer_model_from_settings(settings), manifest)
    else:
        from azure.identity import AzureCliCredential, get_bearer_token_provider
        from azure.search.documents.knowledgebases import KnowledgeBaseRetrievalClient

        from regintel.azure_search.kb_retriever import KnowledgeBaseRetriever
        from regintel.azure_search.knowledge import AzureNames

        credential = AzureCliCredential()
        client = KnowledgeBaseRetrievalClient(
            settings.azure_search_endpoint or "",
            credential,
            knowledge_base_name=AzureNames(settings.azure_name_prefix).knowledge_base("sections"),
            api_version=settings.azure_search_api_version,
        )
        retriever = KnowledgeBaseRetriever("kb-sections", client, chunks, args.effort)
        tokens = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
        answerer = Answerer(retriever, answer_model_from_settings(settings, tokens), manifest)
        if args.agent:
            run_agent(args.question, settings)
            return

    show(answerer.answer(args.question))


def run_agent(question: str, settings: Settings) -> None:
    import asyncio

    from regintel.agent.factory import build_azure_agent

    answer = asyncio.run(build_azure_agent(settings).answer(question))
    print(f"research {answer.research_seconds:.1f}s, {len(answer.calls)} tool calls, usage {answer.research_usage}")
    for call in answer.calls:
        print(f"  {call.name}({call.arguments}) -> E{call.evidence} in {call.seconds}s")
    print(f"""
research note:
{answer.research_note}
""")
    show(answer.result)


def show(result: AnswerResult) -> None:
    print(f"[{result.label}]  retrieval {result.retrieval_seconds:.1f}s, generation {result.generation_seconds:.1f}s")
    print(f"tokens: {result.usage}\n")
    print(result.text, "\n")
    for number, source in enumerate(result.sources, start=1):
        print(f"[{number}] {source.source_label}\n    > {source.quote}")
    for statement in result.statements:
        if not statement.verified:
            print("\nDROPPED:", statement.text, "->", "; ".join(statement.problems))


if __name__ == "__main__":
    main()
