import os

import pytest


@pytest.fixture(autouse=True)
def clean_env():
    phoenix_keys = [
        "PHOENIX_ENABLED",
        "PHOENIX_COLLECTOR_ENDPOINT",
        "PHOENIX_PROJECT_NAME",
    ]
    old = {k: os.environ.get(k) for k in phoenix_keys}
    for k in phoenix_keys:
        os.environ.pop(k, None)
    yield
    for k, v in old.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


def test_phoenix_config_defaults():
    from esdc.chat.phoenix_config import PhoenixConfig

    config = PhoenixConfig.from_env()
    assert config.enabled is False
    assert config.collector_endpoint == "http://localhost:4317"
    assert config.project_name == "esdc-agent"


def test_phoenix_config_enabled_via_env():
    os.environ["PHOENIX_ENABLED"] = "true"
    from esdc.chat.phoenix_config import PhoenixConfig

    config = PhoenixConfig.from_env()
    assert config.enabled is True


def test_phoenix_config_custom_endpoint():
    os.environ["PHOENIX_ENABLED"] = "true"
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://phoenix:4317"
    os.environ["PHOENIX_PROJECT_NAME"] = "esdc-prod"
    from esdc.chat.phoenix_config import PhoenixConfig

    config = PhoenixConfig.from_env()
    assert config.collector_endpoint == "http://phoenix:4317"
    assert config.project_name == "esdc-prod"
