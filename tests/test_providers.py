from esdc.providers.base import Provider, ProviderConfig


def test_provider_config_validation():
    config = ProviderConfig(
        name="openai", provider_type="openai", api_key="sk-test", model="gpt-4o"
    )
    assert config.name == "openai"
    assert config.api_key == "sk-test"
    assert config.provider_type == "openai"
    assert config.model == "gpt-4o"


def test_openai_provider_config():
    from esdc.providers.base import ProviderConfig

    config = ProviderConfig(
        name="test_openai",
        provider_type="openai",
        model="gpt-4o-mini",
    )
    assert config.name == "test_openai"
    assert config.provider_type == "openai"
    assert config.model == "gpt-4o-mini"
