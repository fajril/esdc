"""Tests for OpenTerminal tool integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from esdc.chat.openterminal import (
    _DEFAULT_PACKAGES,
    _build_file_url,
    get_openterminal_tools,
    run_command,
    run_python,
)


class TestGetOpenterminalTools:
    """Tests for conditional tool registration."""

    def test_returns_none_when_not_configured(self):
        """Should return None when OpenTerminal is not configured."""
        with patch("esdc.configs.Config.get_openterminal_config", return_value=None):
            result = get_openterminal_tools()
            assert result is None

    def test_returns_tools_when_configured(self):
        """Should return Compute Engine and Code Interpreter when configured."""
        mock_config = {
            "url": "http://open-terminal:8000",
            "api_key": "test-key",
            "packages": "matplotlib, seaborn, pandas",
            "timeout": 120,
            "write_timeout": 30,
        }
        with (
            patch(
                "esdc.configs.Config.get_openterminal_config",
                return_value=mock_config,
            ),
            patch("esdc.configs.Config.get_openwebui_config", return_value=None),
        ):
            result = get_openterminal_tools()

            assert result is not None
            assert len(result) == 2
            tool_names = {t.name for t in result}
            assert tool_names == {"Compute Engine", "Code Interpreter"}

    def test_updates_run_command_description_with_packages(self):
        """run_command description should include the configured packages."""
        mock_config = {
            "url": "http://open-terminal:8000",
            "api_key": "test-key",
            "packages": "matplotlib, seaborn, pandas",
            "timeout": 120,
            "write_timeout": 30,
        }
        with (
            patch(
                "esdc.configs.Config.get_openterminal_config",
                return_value=mock_config,
            ),
            patch("esdc.configs.Config.get_openwebui_config", return_value=None),
        ):
            result = get_openterminal_tools()

            assert result is not None
            run_cmd = [t for t in result if t.name == "Compute Engine"][0]
            assert "matplotlib, seaborn, pandas" in run_cmd.description

    def test_uses_default_packages_when_not_specified(self):
        """Should use default packages when config omits packages."""
        mock_config = {
            "url": "http://open-terminal:8000",
            "api_key": "test-key",
            "packages": _DEFAULT_PACKAGES,
            "timeout": 120,
            "write_timeout": 30,
        }
        with (
            patch(
                "esdc.configs.Config.get_openterminal_config",
                return_value=mock_config,
            ),
            patch("esdc.configs.Config.get_openwebui_config", return_value=None),
        ):
            result = get_openterminal_tools()

            assert result is not None
            run_cmd = [t for t in result if t.name == "Compute Engine"][0]
            assert "matplotlib" in run_cmd.description
            assert "seaborn" in run_cmd.description
            assert "scikit-learn" in run_cmd.description

    def test_tool_display_names(self):
        """Tools should have branded display names."""
        assert run_command.name == "Compute Engine"
        assert run_python.name == "Code Interpreter"


class TestRunCommandTool:
    """Tests for run_command tool."""

    @pytest.fixture(autouse=True)
    def setup_config(self):
        """Set up mock config for each test."""
        self.config = {
            "url": "http://open-terminal:8000",
            "api_key": "test-key",
            "packages": "matplotlib",
            "timeout": 120,
            "write_timeout": 30,
        }
        with patch("esdc.chat.openterminal._get_config", return_value=self.config):
            yield

    @pytest.mark.asyncio
    async def test_successful_command_v2_sync(self):
        """Should return output on successful command (v2 API with wait param)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "2026-test-12345",
            "command": "echo hello",
            "status": "done",
            "exit_code": 0,
            "output": [
                {"type": "output", "data": "Plot saved to /home/user/output/test.png"}
            ],
            "truncated": False,
            "next_offset": 1,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("esdc.chat.openterminal.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.post = AsyncMock(return_value=mock_response)

            result = await run_command.ainvoke({"command": "echo hello"})
            assert "Plot saved" in result

    @pytest.mark.asyncio
    async def test_failed_command_v2(self):
        """Should return error message for non-zero exit code (v2 API)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "2026-test-12345",
            "command": "invalid_cmd",
            "status": "done",
            "exit_code": 1,
            "output": [{"type": "output", "data": ""}],
        }
        mock_response.raise_for_status = MagicMock()

        with patch("esdc.chat.openterminal.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.post = AsyncMock(return_value=mock_response)

            result = await run_command.ainvoke({"command": "invalid_cmd"})
            assert "exited with code 1" in result

    @pytest.mark.asyncio
    async def test_polling_when_still_running(self):
        """Should poll status endpoint when command still running after wait."""
        # First response: still running
        mock_post_response = MagicMock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {
            "id": "2026-test-12345",
            "command": "sleep 2",
            "status": "running",
            "exit_code": None,
            "output": [],
        }
        mock_post_response.raise_for_status = MagicMock()

        # Second response: done
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            "id": "2026-test-12345",
            "command": "sleep 2",
            "status": "done",
            "exit_code": 0,
            "output": [{"type": "output", "data": "finished"}],
            "next_offset": 1,
        }
        mock_get_response.raise_for_status = MagicMock()

        with patch("esdc.chat.openterminal.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.post = AsyncMock(return_value=mock_post_response)
            mock_instance.get = AsyncMock(return_value=mock_get_response)
            mock_instance.delete = AsyncMock()

            result = await run_command.ainvoke({"command": "sleep 2"})
            assert "finished" in result

    @pytest.mark.asyncio
    async def test_connection_error(self):
        """Should return error when Compute Engine is unreachable."""
        import httpx

        with patch("esdc.chat.openterminal.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.post = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            result = await run_command.ainvoke({"command": "echo hello"})
            assert "Cannot connect to Compute Engine" in result

    @pytest.mark.asyncio
    async def test_timeout_error(self):
        """Should return error on timeout."""
        import httpx

        with patch("esdc.chat.openterminal.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.post = AsyncMock(
                side_effect=httpx.TimeoutException("Timed out")
            )

            result = await run_command.ainvoke({"command": "sleep 200"})
            assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_no_auth_header_without_api_key(self):
        """Should not send auth header when api_key is empty."""
        self.config["api_key"] = ""

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"output": "ok", "exit_code": 0}
        mock_response.raise_for_status = MagicMock()

        with patch("esdc.chat.openterminal.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.post = AsyncMock(return_value=mock_response)

            result = await run_command.ainvoke({"command": "echo hello"})
            assert result == "ok"


class TestConfig:
    """Tests for OpenTerminal and OpenWebUI configuration."""

    def test_get_openterminal_config_returns_none_when_missing(self):
        """Should return None when openterminal section is not in config."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "esdc.configs.Config._load_config",
                return_value={},
            ),
        ):
            from esdc.configs import Config

            Config._config_cache = {}
            result = Config.get_openterminal_config()
            assert result is None

    def test_get_openterminal_config_returns_config_when_set(self):
        """Should return config dict when openterminal is configured."""
        config_data = {
            "openterminal": {
                "url": "http://open-terminal:8000",
                "api_key": "my-secret-key",
                "packages": "matplotlib, pandas",
                "timeout": 60,
                "write_timeout": 10,
            }
        }
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "esdc.configs.Config._load_config",
                return_value=config_data,
            ),
        ):
            from esdc.configs import Config

            Config._config_cache = config_data
            result = Config.get_openterminal_config()

            assert result is not None
            assert result["url"] == "http://open-terminal:8000"
            assert result["api_key"] == "my-secret-key"
            assert result["packages"] == "matplotlib, pandas"
            assert result["timeout"] == 60
            assert result["write_timeout"] == 10

    def test_env_vars_override_config(self):
        """Environment variables should take priority over config file."""
        config_data = {
            "openterminal": {
                "url": "http://config-url:8000",
                "api_key": "config-key",
            }
        }
        env = {
            "OPEN_TERMINAL_URL": "http://env-url:9000",
            "OPEN_TERMINAL_API_KEY": "env-key",
        }
        with (
            patch.dict("os.environ", env, clear=True),
            patch(
                "esdc.configs.Config._load_config",
                return_value=config_data,
            ),
        ):
            from esdc.configs import Config

            Config._config_cache = config_data
            result = Config.get_openterminal_config()

            assert result is not None
            assert result["url"] == "http://env-url:9000"
            assert result["api_key"] == "env-key"

    def test_default_packages_when_not_specified(self):
        """Should use default packages when not specified in config."""
        config_data = {
            "openterminal": {
                "url": "http://open-terminal:8000",
            }
        }
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "esdc.configs.Config._load_config",
                return_value=config_data,
            ),
        ):
            from esdc.configs import Config

            Config._config_cache = config_data
            result = Config.get_openterminal_config()

            assert result is not None
            assert "matplotlib" in result["packages"]
            assert "seaborn" in result["packages"]

    def test_get_openwebui_config_returns_none_when_missing(self):
        """Should return None when openwebui section is not in config."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "esdc.configs.Config._load_config",
                return_value={},
            ),
        ):
            from esdc.configs import Config

            Config._config_cache = {}
            result = Config.get_openwebui_config()
            assert result is None

    def test_get_openwebui_config_returns_config_when_set(self):
        """Should return config dict when openwebui is configured."""
        config_data = {
            "openwebui": {
                "url": "http://localhost:3000",
                "terminal_server_id": "my-terminal",
            }
        }
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "esdc.configs.Config._load_config",
                return_value=config_data,
            ),
        ):
            from esdc.configs import Config

            Config._config_cache = config_data
            result = Config.get_openwebui_config()

            assert result is not None
            assert result["url"] == "http://localhost:3000"
            assert result["terminal_server_id"] == "my-terminal"

    def test_openwebui_env_vars_override_config(self):
        """Environment variables should override config for openwebui."""
        config_data = {
            "openwebui": {
                "url": "http://config-url:3000",
                "terminal_server_id": "config-id",
            }
        }
        env = {
            "OPENWEBUI_URL": "http://env-url:4000",
            "OPENWEBUI_TERMINAL_SERVER_ID": "env-id",
        }
        with (
            patch.dict("os.environ", env, clear=True),
            patch(
                "esdc.configs.Config._load_config",
                return_value=config_data,
            ),
        ):
            from esdc.configs import Config

            Config._config_cache = config_data
            result = Config.get_openwebui_config()

            assert result is not None
            assert result["url"] == "http://env-url:4000"
            assert result["terminal_server_id"] == "env-id"

    def test_openwebui_default_terminal_server_id(self):
        """Should use 'openterminal' as default terminal_server_id."""
        config_data = {
            "openwebui": {
                "url": "http://localhost:3000",
            }
        }
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "esdc.configs.Config._load_config",
                return_value=config_data,
            ),
        ):
            from esdc.configs import Config

            Config._config_cache = config_data
            result = Config.get_openwebui_config()

            assert result is not None
            assert result["terminal_server_id"] == "openterminal"

    def test_get_openwebui_config_includes_proxy_url(self):
        """Should include proxy_url in config when set."""
        config_data = {
            "openwebui": {
                "url": "http://localhost:3000",
                "proxy_url": "https://iris.skkmigas.go.id",
                "terminal_server_id": "my-terminal",
            }
        }
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("esdc.configs.Config._load_config", return_value=config_data),
        ):
            from esdc.configs import Config

            Config._config_cache = config_data
            result = Config.get_openwebui_config()
            assert result is not None
            assert result["url"] == "http://localhost:3000"
            assert result["proxy_url"] == "https://iris.skkmigas.go.id"
            assert result["terminal_server_id"] == "my-terminal"

    def test_proxy_url_defaults_to_url(self):
        """proxy_url should fall back to url when not set."""
        config_data = {
            "openwebui": {
                "url": "http://localhost:3000",
                "terminal_server_id": "my-terminal",
            }
        }
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("esdc.configs.Config._load_config", return_value=config_data),
        ):
            from esdc.configs import Config

            Config._config_cache = config_data
            result = Config.get_openwebui_config()
            assert result is not None
            assert result["proxy_url"] == "http://localhost:3000"

    def test_proxy_url_env_overrides_config(self):
        """OPENWEBUI_PROXY_URL env var should override config."""
        config_data = {
            "openwebui": {
                "url": "http://localhost:3000",
                "proxy_url": "https://config-url.example.com",
            }
        }
        env = {"OPENWEBUI_PROXY_URL": "https://env-url.example.com"}
        with (
            patch.dict("os.environ", env, clear=True),
            patch("esdc.configs.Config._load_config", return_value=config_data),
        ):
            from esdc.configs import Config

            Config._config_cache = config_data
            result = Config.get_openwebui_config()
            assert result is not None
            assert result["proxy_url"] == "https://env-url.example.com"

    def test_proxy_url_strips_trailing_slash(self):
        """proxy_url should strip trailing slash."""
        config_data = {
            "openwebui": {
                "url": "http://localhost:3000/",
                "proxy_url": "https://iris.skkmigas.go.id/",
            }
        }
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("esdc.configs.Config._load_config", return_value=config_data),
        ):
            from esdc.configs import Config

            Config._config_cache = config_data
            result = Config.get_openwebui_config()
            assert result is not None
            assert not result["url"].endswith("/")
            assert not result["proxy_url"].endswith("/")

    def test_openwebui_strips_trailing_slash(self):
        """Should strip trailing slash from openwebui URL."""
        config_data = {
            "openwebui": {
                "url": "http://localhost:3000/",
                "terminal_server_id": "my-terminal",
            }
        }
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "esdc.configs.Config._load_config",
                return_value=config_data,
            ),
        ):
            from esdc.configs import Config

            Config._config_cache = config_data
            result = Config.get_openwebui_config()

            assert result is not None
            assert result["url"] == "http://localhost:3000"


class TestBuildFileUrl:
    """Tests for _build_file_url helper."""

    def test_builds_url_with_proxy_url(self):
        """Should use proxy_url when available."""
        ow_config = {
            "url": "http://localhost:3000",
            "proxy_url": "https://iris.skkmigas.go.id",
            "terminal_server_id": "my-terminal",
        }
        with patch("esdc.chat.openterminal._get_ow_config", return_value=ow_config):
            url = _build_file_url("/home/user/output/test.png", "Test Image")
            assert (
                url
                == "![Test Image](https://iris.skkmigas.go.id/api/v1/terminals/my-terminal/files/read?path=/home/user/output/test.png)"
            )

    def test_falls_back_to_url_when_no_proxy_url(self):
        """Should fall back to url when proxy_url is not set."""
        ow_config = {
            "url": "http://localhost:3000",
            "terminal_server_id": "my-terminal",
        }
        with patch("esdc.chat.openterminal._get_ow_config", return_value=ow_config):
            url = _build_file_url("/home/user/output/test.png", "Test Image")
            assert (
                url
                == "![Test Image](http://localhost:3000/api/v1/terminals/my-terminal/files/read?path=/home/user/output/test.png)"
            )

    def test_uses_filename_as_default_description(self):
        """Should use filename when description is empty."""
        ow_config = {
            "url": "http://localhost:3000",
            "proxy_url": "https://iris.skkmigas.go.id",
            "terminal_server_id": "openterminal",
        }
        with patch("esdc.chat.openterminal._get_ow_config", return_value=ow_config):
            url = _build_file_url("/home/user/output/forecast.png")
            assert "![forecast.png]" in url
            assert "https://iris.skkmigas.go.id" in url

    def test_strips_trailing_slash_from_proxy_url(self):
        """Should strip trailing slash from proxy_url."""
        ow_config = {
            "url": "http://localhost:3000",
            "proxy_url": "https://iris.skkmigas.go.id/",
            "terminal_server_id": "openterminal",
        }
        with patch("esdc.chat.openterminal._get_ow_config", return_value=ow_config):
            url = _build_file_url("/home/user/output/test.png", "Test")
            assert url.startswith("![Test](https://iris.skkmigas.go.id/api/")

    def test_url_encodes_paths_with_special_characters(self):
        """Should URL-encode file paths with special characters."""
        ow_config = {
            "url": "http://localhost:3000",
            "proxy_url": "https://iris.skkmigas.go.id",
            "terminal_server_id": "openterminal",
        }
        with patch("esdc.chat.openterminal._get_ow_config", return_value=ow_config):
            url = _build_file_url("/home/user/output/forecast 2024.png", "Chart")
            assert "forecast%202024.png" in url
            assert "![Chart]" in url

    def test_raises_when_ow_config_missing(self):
        """Should raise ValueError when OpenWebUI not configured."""
        with (
            patch(
                "esdc.chat.openterminal._get_ow_config",
                side_effect=ValueError("OpenWebUI not configured"),
            ),
            pytest.raises(ValueError, match="OpenWebUI not configured"),
        ):
            _build_file_url("/home/user/test.png")
