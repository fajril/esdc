"""Tests for OpenTerminal tool integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from esdc.chat.openterminal import (
    _DEFAULT_PACKAGES,
    _build_file_url,
    get_openterminal_tools,
    run_command,
    view_file,
    write_file,
)


class TestGetOpenterminalTools:
    """Tests for conditional tool registration."""

    def test_returns_none_when_not_configured(self):
        """Should return None when OpenTerminal is not configured."""
        with patch("esdc.configs.Config.get_openterminal_config", return_value=None):
            result = get_openterminal_tools()
            assert result is None

    def test_returns_tools_when_configured_without_openwebui(self):
        """Should return Compute Engine and File Processing when no openwebui config."""
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
            assert tool_names == {"Compute Engine", "File Processing"}

    def test_returns_three_tools_with_openwebui(self):
        """Should include View File when openwebui config is set."""
        mock_ot_config = {
            "url": "http://open-terminal:8000",
            "api_key": "test-key",
            "packages": "matplotlib, seaborn, pandas",
            "timeout": 120,
            "write_timeout": 30,
        }
        mock_ow_config = {
            "url": "http://localhost:3000",
            "terminal_server_id": "openterminal",
        }
        with (
            patch(
                "esdc.configs.Config.get_openterminal_config",
                return_value=mock_ot_config,
            ),
            patch(
                "esdc.configs.Config.get_openwebui_config",
                return_value=mock_ow_config,
            ),
        ):
            result = get_openterminal_tools()

            assert result is not None
            assert len(result) == 3
            tool_names = {t.name for t in result}
            assert tool_names == {"Compute Engine", "File Processing", "View File"}

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
        assert write_file.name == "File Processing"
        assert view_file.name == "View File"


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
    async def test_successful_command(self):
        """Should return output on successful command."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": "Plot saved to /home/user/output/test.png",
            "exit_code": 0,
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
    async def test_failed_command(self):
        """Should return error message for non-zero exit code."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": "",
            "exit_code": 1,
            "error": "command not found",
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


class TestWriteFileTool:
    """Tests for write_file tool."""

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
    async def test_successful_write(self):
        """Should return success message on successful write."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("esdc.chat.openterminal.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.post = AsyncMock(return_value=mock_response)

            result = await write_file.ainvoke(
                {"path": "/home/user/plot.py", "content": "print('hello')"}
            )
            assert "success" in result.lower()
            assert "/home/user/plot.py" in result

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

            result = await write_file.ainvoke(
                {"path": "/home/user/test.py", "content": "test"}
            )
            assert "Cannot connect to Compute Engine" in result


class TestViewFileTool:
    """Tests for view_file tool."""

    @pytest.fixture(autouse=True)
    def setup_config(self):
        """Set up mock configs for each test."""
        self.ot_config = {
            "url": "http://open-terminal:8000",
            "api_key": "test-key",
            "packages": "matplotlib",
            "timeout": 120,
            "write_timeout": 30,
        }
        self.ow_config = {
            "url": "http://localhost:3000",
            "terminal_server_id": "my-terminal",
        }

    @pytest.mark.asyncio
    async def test_returns_markdown_image_with_proxy_url(self):
        """Should return markdown image with OpenWebUI proxy URL."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with (
            patch("esdc.chat.openterminal._get_config", return_value=self.ot_config),
            patch("esdc.chat.openterminal._get_ow_config", return_value=self.ow_config),
            patch("esdc.chat.openterminal.httpx.AsyncClient") as mock_client,
        ):
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.head = AsyncMock(return_value=mock_response)

            result = await view_file.ainvoke(
                {"path": "/home/user/output/forecast.png", "description": "Forecast"}
            )
            assert "![Forecast]" in result
            assert "/api/v1/terminals/my-terminal/files/read" in result
            assert "forecast.png" in result

    @pytest.mark.asyncio
    async def test_file_not_found(self):
        """Should return error when file does not exist."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with (
            patch("esdc.chat.openterminal._get_config", return_value=self.ot_config),
            patch("esdc.chat.openterminal._get_ow_config", return_value=self.ow_config),
            patch("esdc.chat.openterminal.httpx.AsyncClient") as mock_client,
        ):
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.head = AsyncMock(return_value=mock_response)

            result = await view_file.ainvoke({"path": "/home/user/output/missing.png"})
            assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_default_description_from_filename(self):
        """Should use filename as description when not provided."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with (
            patch("esdc.chat.openterminal._get_config", return_value=self.ot_config),
            patch("esdc.chat.openterminal._get_ow_config", return_value=self.ow_config),
            patch("esdc.chat.openterminal.httpx.AsyncClient") as mock_client,
        ):
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.head = AsyncMock(return_value=mock_response)

            result = await view_file.ainvoke({"path": "/home/user/output/forecast.png"})
            assert "![forecast.png]" in result

    @pytest.mark.asyncio
    async def test_connection_error(self):
        """Should return error when Compute Engine is unreachable."""
        import httpx

        with (
            patch("esdc.chat.openterminal._get_config", return_value=self.ot_config),
            patch("esdc.chat.openterminal._get_ow_config", return_value=self.ow_config),
            patch("esdc.chat.openterminal.httpx.AsyncClient") as mock_client,
        ):
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.head = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            result = await view_file.ainvoke({"path": "/home/user/output/test.png"})
            assert "Cannot connect to Compute Engine" in result

    @pytest.mark.asyncio
    async def test_url_encoding_for_paths(self):
        """Should URL-encode file paths with special characters."""
        from esdc.chat.openterminal import _build_file_url

        ow_config = {
            "url": "http://localhost:3000",
            "terminal_server_id": "openterminal",
        }
        with patch("esdc.chat.openterminal._get_ow_config", return_value=ow_config):
            url = _build_file_url("/home/user/output/forecast 2024.png", "Chart")
            assert "forecast%202024.png" in url
            assert "![Chart]" in url


class TestBuildFileUrl:
    """Tests for _build_file_url helper."""

    def test_builds_proxy_url(self):
        """Should build correct proxy URL."""
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
            "terminal_server_id": "openterminal",
        }
        with patch("esdc.chat.openterminal._get_ow_config", return_value=ow_config):
            url = _build_file_url("/home/user/output/forecast.png")
            assert "![forecast.png]" in url

    def test_strips_trailing_slash_from_url(self):
        """Should strip trailing slash from OpenWebUI URL."""
        ow_config = {
            "url": "http://localhost:3000/",
            "terminal_server_id": "openterminal",
        }
        with patch("esdc.chat.openterminal._get_ow_config", return_value=ow_config):
            url = _build_file_url("/home/user/output/test.png", "Test")
            assert url.startswith("![Test](http://localhost:3000/api/")

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
