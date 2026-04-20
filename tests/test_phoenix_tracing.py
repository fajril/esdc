import os
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.allow_provider_config


@pytest.fixture(autouse=True)
def clean_env_and_state():
    phoenix_keys = [
        "PHOENIX_ENABLED",
        "PHOENIX_COLLECTOR_ENDPOINT",
        "PHOENIX_PROJECT_NAME",
    ]
    old = {k: os.environ.get(k) for k in phoenix_keys}
    for k in phoenix_keys:
        os.environ.pop(k, None)

    import esdc.phoenix.phoenix_tracing as pt

    pt._initialized = False
    yield
    pt._initialized = False

    for k, v in old.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


def test_setup_phoenix_tracing_disabled():
    with patch("esdc.configs.Config._load_config") as mock_load:
        mock_load.return_value = {}
        import esdc.phoenix.phoenix_tracing as pt

        result = pt.setup_phoenix_tracing()
        assert result is False


def test_setup_phoenix_tracing_enabled():
    with patch("esdc.configs.Config._load_config") as mock_load:
        mock_load.return_value = {
            "phoenix": {
                "enabled": True,
                "collector_endpoint": "http://localhost:4317",
                "project_name": "iris",
            },
        }
        with patch("phoenix.otel.register") as mock_register:
            mock_register.return_value = MagicMock()

            import esdc.phoenix.phoenix_tracing as pt

            result = pt.setup_phoenix_tracing()
            assert result is True
            mock_register.assert_called_once_with(
                endpoint="http://localhost:4317",
                project_name="iris",
                auto_instrument=True,
            )


def test_setup_phoenix_tracing_idempotent():
    with patch("esdc.configs.Config._load_config") as mock_load:
        mock_load.return_value = {
            "phoenix": {
                "enabled": True,
                "collector_endpoint": "http://localhost:4317",
                "project_name": "iris",
            },
        }
        with patch("phoenix.otel.register") as mock_register:
            mock_register.return_value = MagicMock()

            from esdc.phoenix.phoenix_tracing import setup_phoenix_tracing

            result1 = setup_phoenix_tracing()
            result2 = setup_phoenix_tracing()
            assert result1 is True
            assert result2 is True
            assert mock_register.call_count == 1


def test_setup_phoenix_tracing_custom_endpoint():
    with patch("esdc.configs.Config._load_config") as mock_load:
        mock_load.return_value = {
            "phoenix": {
                "enabled": True,
                "collector_endpoint": "http://phoenix-server:4317",
                "project_name": "iris-prod",
            },
        }
        with patch("phoenix.otel.register") as mock_register:
            mock_register.return_value = MagicMock()

            import esdc.phoenix.phoenix_tracing as pt

            result = pt.setup_phoenix_tracing()
            assert result is True
            mock_register.assert_called_once_with(
                endpoint="http://phoenix-server:4317",
                project_name="iris-prod",
                auto_instrument=True,
            )


def test_setup_phoenix_tracing_env_overrides_yaml():
    os.environ["PHOENIX_ENABLED"] = "true"
    os.environ["PHOENIX_PROJECT_NAME"] = "iris-from-env"
    with patch("esdc.configs.Config._load_config") as mock_load:
        mock_load.return_value = {
            "phoenix": {
                "enabled": True,
                "collector_endpoint": "http://yaml:4317",
                "project_name": "iris-from-yaml",
            },
        }
        with patch("phoenix.otel.register") as mock_register:
            mock_register.return_value = MagicMock()

            import esdc.phoenix.phoenix_tracing as pt

            result = pt.setup_phoenix_tracing()
            assert result is True
            mock_register.assert_called_once_with(
                endpoint="http://yaml:4317",
                project_name="iris-from-env",
                auto_instrument=True,
            )
