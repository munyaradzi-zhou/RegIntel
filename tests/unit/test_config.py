from pathlib import Path

import pytest
from pydantic import ValidationError

from regintel.config import Settings


def test_defaults_run_locally() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "openai"
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.search_backend == "local"
    assert settings.claude_model == "claude-opus-5-5"


def test_foundry_requires_resource() -> None:
    with pytest.raises(ValidationError, match="FOUNDRY_RESOURCE"):
        Settings(_env_file=None, llm_provider="foundry")


def test_azure_search_requires_endpoints() -> None:
    with pytest.raises(ValidationError, match="AZURE_SEARCH_ENDPOINT"):
        Settings(_env_file=None, search_backend="azure", azure_search_endpoint="https://x")


def test_derived_paths() -> None:
    settings = Settings(_env_file=None, data_dir=Path("d"))
    assert settings.corpus_dir == Path("d/raw/guidance")
    assert settings.index_dir == Path("d/index")
