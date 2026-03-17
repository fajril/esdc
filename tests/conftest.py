import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner():
    """Create a fresh CliRunner for each test."""
    return CliRunner()


@pytest.fixture(autouse=True)
def reset_config_cache():
    """Reset Config cache before each test."""
    from esdc.configs import Config

    Config._config_cache = None
    yield
