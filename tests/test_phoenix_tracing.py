import os
import sys
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.allow_provider_config


def test_setup_phoenix_tracing_disabled():
    import esdc.phoenix.phoenix_tracing as pt

    pt._initialized = False

    result = pt.setup_phoenix_tracing()
    assert result is False


def test_setup_phoenix_tracing_enabled():
    os.environ["PHOENIX_ENABLED"] = "true"

    import esdc.phoenix.phoenix_tracing as pt

    pt._initialized = False

    with patch("phoenix.otel.register") as mock_register:
        mock_register.return_value = MagicMock()

        result = pt.setup_phoenix_tracing()
        assert result is True
        mock_register.assert_called_once_with(
            project_name="esdc-agent",
            auto_instrument=True,
        )


def test_setup_phoenix_tracing_idempotent():
    os.environ["PHOENIX_ENABLED"] = "true"

    import esdc.phoenix.phoenix_tracing as pt

    pt._initialized = False

    with patch("phoenix.otel.register") as mock_register:
        mock_register.return_value = MagicMock()

        from esdc.phoenix.phoenix_tracing import setup_phoenix_tracing

        result1 = setup_phoenix_tracing()
        result2 = setup_phoenix_tracing()
        assert result1 is True
        assert result2 is True
        assert mock_register.call_count == 1


def test_setup_phoenix_tracing_custom_endpoint():
    os.environ["PHOENIX_ENABLED"] = "true"
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://phoenix-server:4317"
    os.environ["PHOENIX_PROJECT_NAME"] = "esdc-prod"

    import esdc.phoenix.phoenix_tracing as pt

    pt._initialized = False

    with patch("phoenix.otel.register") as mock_register:
        mock_register.return_value = MagicMock()

        result = pt.setup_phoenix_tracing()
        assert result is True
        mock_register.assert_called_once_with(
            project_name="esdc-prod",
            auto_instrument=True,
        )
