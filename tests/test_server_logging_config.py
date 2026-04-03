"""Test logging configuration module."""

import pytest
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

from esdc.server.logging_config import (
    parse_size,
    setup_server_logging,
    get_request_logger,
)


class TestParseSize:
    """Test size parsing function."""

    def test_parse_kb(self):
        """Parse KB sizes."""
        assert parse_size("10KB") == 10 * 1024
        assert parse_size("100KB") == 100 * 1024

    def test_parse_mb(self):
        """Parse MB sizes."""
        assert parse_size("10MB") == 10 * 1024 * 1024
        assert parse_size("10mb") == 10 * 1024 * 1024

    def test_parse_gb(self):
        """Parse GB sizes."""
        assert parse_size("1GB") == 1024**3

    def test_parse_bytes(self):
        """Parse bytes without suffix."""
        assert parse_size("1024") == 1024


class TestSetupServerLogging:
    """Test server logging setup."""

    def test_setup_creates_logger(self, tmp_path):
        """Verify setup_server_logging creates logger."""
        config = {
            "level": "INFO",
            "server": {
                "level": "DEBUG",
                "file": {
                    "enabled": True,
                    "path": "test_server.log",
                    "max_size": "1MB",
                    "backup_count": 2,
                },
                "console": {"enabled": False},
            },
        }

        # Create mock Config class
        mock_config_class = MagicMock()
        mock_config_class.get_config_dir.return_value = tmp_path

        # Patch the import inside the function
        with patch.dict(
            "sys.modules", {"esdc.configs": MagicMock(Config=mock_config_class)}
        ):
            # This should create logger
            logger = setup_server_logging(config)

            assert logger is not None
            assert logger.name == "esdc.server"

    def test_get_request_logger(self):
        """Verify get_request_logger returns logger."""
        logger = get_request_logger()
        assert logger.name == "esdc.server.middleware"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
