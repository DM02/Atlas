import pytest

import app.ai.embeddings.factory as embedding_factory_module
import app.ai.llm.factory as llm_factory_module
from app.ai.embeddings.factory import get_embedding_provider
from app.ai.llm.factory import get_llm_provider
from app.core.config import Settings


@pytest.fixture(autouse=True)
def _clear_provider_caches():
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()
    yield
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()


def test_get_llm_provider_uses_openrouter_when_base_url_set(monkeypatch) -> None:
    settings = Settings(
        llm_base_url="https://openrouter.ai/api/v1",
        openrouter_api_key="or-test-key",
        llm_model="some/model:free",
    )
    monkeypatch.setattr(llm_factory_module, "get_settings", lambda: settings)

    provider = get_llm_provider()

    assert provider.model_name == "some/model:free"
    assert str(provider._client.base_url).startswith("https://openrouter.ai")


def test_get_llm_provider_raises_when_openrouter_key_missing(monkeypatch) -> None:
    settings = Settings(llm_base_url="https://openrouter.ai/api/v1", openrouter_api_key=None)
    monkeypatch.setattr(llm_factory_module, "get_settings", lambda: settings)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        get_llm_provider()


def test_get_llm_provider_falls_back_to_openai_when_base_url_unset(monkeypatch) -> None:
    settings = Settings(llm_base_url=None, openai_api_key="oa-test-key", llm_model="gpt-4o-mini")
    monkeypatch.setattr(llm_factory_module, "get_settings", lambda: settings)

    provider = get_llm_provider()

    assert provider.model_name == "gpt-4o-mini"
    assert "openrouter" not in str(provider._client.base_url)


def test_get_llm_provider_raises_when_openai_key_missing_and_no_base_url(monkeypatch) -> None:
    settings = Settings(llm_base_url=None, openai_api_key=None)
    monkeypatch.setattr(llm_factory_module, "get_settings", lambda: settings)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_llm_provider()


def test_get_embedding_provider_uses_openrouter_when_base_url_set(monkeypatch) -> None:
    settings = Settings(
        embedding_base_url="https://openrouter.ai/api/v1",
        openrouter_api_key="or-test-key",
        embedding_model="openai/text-embedding-3-small",
        openai_embedding_dimension=1536,
    )
    monkeypatch.setattr(embedding_factory_module, "get_settings", lambda: settings)

    provider = get_embedding_provider()

    assert provider.model_name == "openai/text-embedding-3-small"
    assert provider.dimension == 1536
    assert str(provider._client.base_url).startswith("https://openrouter.ai")


def test_get_embedding_provider_raises_when_openrouter_key_missing(monkeypatch) -> None:
    settings = Settings(embedding_base_url="https://openrouter.ai/api/v1", openrouter_api_key=None)
    monkeypatch.setattr(embedding_factory_module, "get_settings", lambda: settings)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        get_embedding_provider()


def test_get_embedding_provider_falls_back_to_openai_when_base_url_unset(monkeypatch) -> None:
    settings = Settings(
        embedding_base_url=None,
        openai_api_key="oa-test-key",
        embedding_model="text-embedding-3-small",
    )
    monkeypatch.setattr(embedding_factory_module, "get_settings", lambda: settings)

    provider = get_embedding_provider()

    assert provider.model_name == "text-embedding-3-small"
    assert "openrouter" not in str(provider._client.base_url)


def test_get_embedding_provider_raises_when_openai_key_missing_and_no_base_url(monkeypatch) -> None:
    settings = Settings(embedding_base_url=None, openai_api_key=None)
    monkeypatch.setattr(embedding_factory_module, "get_settings", lambda: settings)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_embedding_provider()
