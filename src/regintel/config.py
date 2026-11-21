"""Settings, read from the environment and .env.

Every external service is optional except the LLM: with only OPENAI_API_KEY set,
the system runs locally on disk (BM25 + local embeddings, in-memory sessions).
Development runs on gpt-4o-mini to keep costs down; the release configuration is Claude.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_provider: Literal["openai", "anthropic", "foundry"] = "openai"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: SecretStr | None = None
    claude_model: str = "claude-opus-5-5"
    foundry_resource: str | None = None

    # Retrieval
    search_backend: Literal["local", "azure"] = "local"
    azure_search_endpoint: str | None = None
    azure_search_api_version: str = "2026-08-01-preview"  # query planning, reasoning effort, web KS
    azure_openai_endpoint: str | None = None
    azure_openai_chat_deployment: str = "gpt-5-mini"
    azure_openai_embedding_deployment: str = "text-embedding-3-large"
    # Answer writer, when it is not the Azure OpenAI chat deployment above: any model on an Azure
    # endpoint with the /openai/v1/ API (e.g. grok-4.7 or Llama on the AI Services resource).
    # "responses" uses the Responses API; "chat" uses Chat Completions with a JSON schema, for models
    # whose Responses API ignores the schema.
    writer_api: Literal["responses", "chat"] = "responses"
    writer_endpoint: str | None = None  # default: azure_openai_endpoint
    writer_deployment: str | None = None  # default: azure_openai_chat_deployment
    writer_reasoning_effort: Literal["minimal", "low", "medium", "high", "none"] = "low"  # responses only
    # Answer reviewer: a different model family from the writer, on an AI Services (Foundry) resource
    azure_judge_endpoint: str | None = None
    azure_judge_deployment: str = "gpt-oss-120b"
    azure_storage_account: str | None = None
    azure_storage_container: str = "regintel-guidance"
    azure_name_prefix: str = "regintel"  # every search object RegIntel creates starts with this
    retrieval_top_k: int = Field(default=8, ge=1, le=50)

    # Data
    data_dir: Path = Path("data")
    openfda_api_key: SecretStr | None = None
    # API: comma-separated client keys (none = open, for local development only) and a per-client limit
    api_keys: SecretStr | None = None
    rate_limit_per_minute: int = 30

    # Azure state
    cosmos_endpoint: str | None = None
    applicationinsights_connection_string: SecretStr | None = None

    @model_validator(mode="after")
    def _check_backends(self) -> Settings:
        if self.llm_provider == "foundry" and not self.foundry_resource:
            raise ValueError("LLM_PROVIDER=foundry needs FOUNDRY_RESOURCE")
        if self.search_backend == "azure" and not (self.azure_search_endpoint and self.azure_openai_endpoint):
            raise ValueError("SEARCH_BACKEND=azure needs AZURE_SEARCH_ENDPOINT and AZURE_OPENAI_ENDPOINT")
        return self

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "raw" / "guidance"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"


@lru_cache
def get_settings() -> Settings:
    return Settings()
