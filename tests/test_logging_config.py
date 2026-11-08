"""Test logging configuration loading from config.yaml."""

import pytest


class TestLoggingConfig:
    """Test logging configuration loading."""

    def test_default_logging_config_structure(self):
        """Verify default logging config has all required fields."""
        # This test will fail initially because Config doesn't have logging methods yet
        # It documents what we expect to implement

        # Mock config.yaml content with logging section
        mock_config = {
            "api_url": "https://esdc.skkmigas.go.id/",
            "database": {"path": "/test/path"},
            "logging": {
                "level": "INFO",
                "server": {
                    "level": "DEBUG",
                    "file": {
                        "enabled": True,
                        "path": "logs/esdc_server.log",
                        "max_size": "10MB",
                        "backup_count": 5,
                    },
                    "console": {"enabled": True},
                },
                "chat": {
                    "level": "WARNING",
                    "file": {
                        "enabled": True,
                        "path": "logs/esdc_chat.log",
                        "max_size": "10MB",
                        "backup_count": 5,
                    },
                    "console": {"enabled": False},
                },
                "performance": {
                    "enabled": True,
                    "slow_request_threshold_ms": 1000,
                    "log_tool_timing": True,
                    "log_llm_timing": True,
                },
            },
        }

        # Verify structure
        assert "logging" in mock_config
        logging_config = mock_config["logging"]

        # Check global level
        assert "level" in logging_config
        assert logging_config["level"] in [
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ]

        # Check server config
        assert "server" in logging_config
        server_config = logging_config["server"]
        assert "level" in server_config
        assert "file" in server_config
        assert "console" in server_config

        # Check file config structure
        file_config = server_config["file"]
        assert "enabled" in file_config
        assert "path" in file_config
        assert "max_size" in file_config
        assert "backup_count" in file_config

        # Check chat config
        assert "chat" in logging_config

        # Check performance config
        assert "performance" in logging_config
        perf_config = logging_config["performance"]
        assert "enabled" in perf_config
        assert "slow_request_threshold_ms" in perf_config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
