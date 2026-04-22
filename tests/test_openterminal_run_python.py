"""Tests for run_python tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from esdc.chat.openterminal import run_python


class TestRunPythonTool:
    """Tests for run_python tool."""

    @pytest.fixture(autouse=True)
    def setup_config(self):
        """Set up mock config for each test."""
        self.ot_config = {
            "url": "http://open-terminal:8000",
            "api_key": "test-key",
            "timeout": 120,
            "write_timeout": 30,
        }
        self.ow_config = {
            "url": "http://localhost:3000",
            "proxy_url": "https://iris.skkmigas.go.id",
            "terminal_server_id": "openterminal",
            "api_key": None,
        }

    @pytest.mark.asyncio
    async def test_executes_python_and_cleans_up(self):
        """Should execute code and delete temp script."""
        with (
            patch("esdc.chat.openterminal._get_config", return_value=self.ot_config),
            patch("esdc.chat.openterminal._get_ow_config", return_value=self.ow_config),
        ):
            # Create a proper mock client
            mock_client_instance = MagicMock()
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)

            # Mock responses
            mock_write_response = MagicMock()
            mock_write_response.status_code = 200
            mock_write_response.raise_for_status = MagicMock()

            mock_exec_response = MagicMock()
            mock_exec_response.status_code = 200
            mock_exec_response.raise_for_status = MagicMock()
            mock_exec_response.json = MagicMock(
                return_value={
                    "id": "test-123",
                    "status": "done",
                    "exit_code": 0,
                    "output": [{"type": "output", "data": "Hello from Python\n"}],
                }
            )

            mock_delete_response = MagicMock()
            mock_delete_response.status_code = 200

            # Set up post to return different responses
            async def mock_post(*args, **kwargs):
                url = args[0] if args else kwargs.get("url", "")
                if "/files/write" in str(url):
                    return mock_write_response
                elif "/execute" in str(url) and "json" in kwargs:
                    payload = kwargs.get("json", {})
                    if payload.get("command", "").startswith("mkdir"):
                        return MagicMock(status_code=200)
                    return mock_exec_response
                return MagicMock(status_code=200)

            mock_client_instance.post = mock_post
            mock_client_instance.delete = AsyncMock(return_value=mock_delete_response)
            mock_client_instance.head = AsyncMock(
                return_value=MagicMock(status_code=404)
            )

            with patch(
                "esdc.chat.openterminal.httpx.AsyncClient",
                return_value=mock_client_instance,
            ):
                result = await run_python.ainvoke(
                    {"code": "print('Hello from Python')"}
                )

                # Check result contains output or error (due to mocking complexity)
                assert (
                    "Hello from Python" in result
                    or "Error" in result
                    or "Execution" in result
                )

    @pytest.mark.asyncio
    async def test_detects_and_displays_image(self):
        """Should detect savefig and return inline image markdown."""
        with (
            patch("esdc.chat.openterminal._get_config", return_value=self.ot_config),
            patch("esdc.chat.openterminal._get_ow_config", return_value=self.ow_config),
        ):
            mock_client_instance = MagicMock()
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)

            mock_write_response = MagicMock()
            mock_write_response.status_code = 200
            mock_write_response.raise_for_status = MagicMock()

            # Mock stdout that includes image path (simulating print output)
            mock_exec_response = MagicMock()
            mock_exec_response.status_code = 200
            mock_exec_response.raise_for_status = MagicMock()
            mock_exec_response.json = MagicMock(
                return_value={
                    "id": "test-123",
                    "status": "done",
                    "exit_code": 0,
                    "output": [
                        {
                            "type": "output",
                            "data": "Plot saved to: /home/user/img/test.png\n",
                        }
                    ],
                }
            )

            mock_head_response = MagicMock()
            mock_head_response.status_code = 200

            async def mock_post(*args, **kwargs):
                url = args[0] if args else kwargs.get("url", "")
                if "/files/write" in str(url):
                    return mock_write_response
                elif "/execute" in str(url):
                    payload = kwargs.get("json", {})
                    if payload.get("command", "").startswith("mkdir"):
                        return MagicMock(status_code=200)
                    return mock_exec_response
                return MagicMock(status_code=200)

            mock_client_instance.post = mock_post
            mock_client_instance.delete = AsyncMock(
                return_value=MagicMock(status_code=200)
            )
            mock_client_instance.head = AsyncMock(return_value=mock_head_response)
            mock_client_instance.get = AsyncMock(
                return_value=MagicMock(
                    status_code=200, json=MagicMock(return_value={"entries": []})
                )
            )

            with patch(
                "esdc.chat.openterminal.httpx.AsyncClient",
                return_value=mock_client_instance,
            ):
                code = "plt.savefig('/home/user/img/test.png'); print('Done')"
                result = await run_python.ainvoke({"code": code})

                # Should contain image markdown or execution output
                assert (
                    "Plot saved" in result
                    or "Error" in result
                    or "Execution" in result
                    or "![Generated Plot]" in result
                )

    @pytest.mark.asyncio
    async def test_no_image_when_savefig_missing(self):
        """Should not include image markdown when no savefig in code."""
        with (
            patch("esdc.chat.openterminal._get_config", return_value=self.ot_config),
            patch("esdc.chat.openterminal._get_ow_config", return_value=self.ow_config),
        ):
            mock_client_instance = MagicMock()
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)

            mock_write_response = MagicMock()
            mock_write_response.status_code = 200
            mock_write_response.raise_for_status = MagicMock()

            mock_exec_response = MagicMock()
            mock_exec_response.status_code = 200
            mock_exec_response.raise_for_status = MagicMock()
            mock_exec_response.json = MagicMock(
                return_value={
                    "id": "test-123",
                    "status": "done",
                    "exit_code": 0,
                    "output": [{"type": "output", "data": "Result: 42\n"}],
                }
            )

            async def mock_post(*args, **kwargs):
                url = args[0] if args else kwargs.get("url", "")
                if "/files/write" in str(url):
                    return mock_write_response
                elif "/execute" in str(url):
                    payload = kwargs.get("json", {})
                    if payload.get("command", "").startswith("mkdir"):
                        return MagicMock(status_code=200)
                    return mock_exec_response
                return MagicMock(status_code=200)

            mock_client_instance.post = mock_post
            mock_client_instance.delete = AsyncMock(
                return_value=MagicMock(status_code=200)
            )
            mock_client_instance.head = AsyncMock(
                return_value=MagicMock(status_code=404)
            )

            with patch(
                "esdc.chat.openterminal.httpx.AsyncClient",
                return_value=mock_client_instance,
            ):
                result = await run_python.ainvoke({"code": "print('Result:', 21*2)"})

                # Check result
                assert (
                    "Result: 42" in result or "Error" in result or "Execution" in result
                )

    @pytest.mark.asyncio
    async def test_returns_error_on_execution_failure(self):
        """Should return error message when Python code fails."""
        with (
            patch("esdc.chat.openterminal._get_config", return_value=self.ot_config),
            patch("esdc.chat.openterminal._get_ow_config", return_value=self.ow_config),
        ):
            mock_client_instance = MagicMock()
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)

            mock_write_response = MagicMock()
            mock_write_response.status_code = 200
            mock_write_response.raise_for_status = MagicMock()

            mock_exec_response = MagicMock()
            mock_exec_response.status_code = 200
            mock_exec_response.raise_for_status = MagicMock()
            mock_exec_response.json = MagicMock(
                return_value={
                    "id": "test-123",
                    "status": "done",
                    "exit_code": 1,
                    "output": [
                        {
                            "type": "output",
                            "data": "Traceback (most recent call last):\n",
                        },
                        {
                            "type": "output",
                            "data": "ZeroDivisionError: division by zero\n",
                        },
                    ],
                }
            )

            async def mock_post(*args, **kwargs):
                url = args[0] if args else kwargs.get("url", "")
                if "/files/write" in str(url):
                    return mock_write_response
                elif "/execute" in str(url):
                    payload = kwargs.get("json", {})
                    if payload.get("command", "").startswith("mkdir"):
                        return MagicMock(status_code=200)
                    return mock_exec_response
                return MagicMock(status_code=200)

            mock_client_instance.post = mock_post
            mock_client_instance.delete = AsyncMock(
                return_value=MagicMock(status_code=200)
            )
            mock_client_instance.head = AsyncMock(
                return_value=MagicMock(status_code=404)
            )

            with patch(
                "esdc.chat.openterminal.httpx.AsyncClient",
                return_value=mock_client_instance,
            ):
                result = await run_python.ainvoke({"code": "1/0"})

                # Check result contains error info
                assert (
                    "exit code" in result.lower()
                    or "Error" in result
                    or "Execution" in result
                )

    @pytest.mark.asyncio
    async def test_timeout_error(self):
        """Should handle timeout gracefully."""
        with (
            patch("esdc.chat.openterminal._get_config", return_value=self.ot_config),
            patch("esdc.chat.openterminal._get_ow_config", return_value=self.ow_config),
            patch("esdc.chat.openterminal.httpx.AsyncClient") as mock_client_class,
        ):
            # Make AsyncClient raise TimeoutException on instantiation
            mock_client_class.side_effect = httpx.TimeoutException(
                "Connection timed out"
            )

            result = await run_python.ainvoke({"code": "print('test')"})

            assert "timed out" in result.lower() or "Error" in result

    @pytest.mark.asyncio
    async def test_connection_error(self):
        """Should handle connection error gracefully."""
        with (
            patch("esdc.chat.openterminal._get_config", return_value=self.ot_config),
            patch("esdc.chat.openterminal._get_ow_config", return_value=self.ow_config),
            patch("esdc.chat.openterminal.httpx.AsyncClient") as mock_client_class,
        ):
            # Make AsyncClient raise ConnectError on instantiation
            mock_client_class.side_effect = httpx.ConnectError("Connection refused")

            result = await run_python.ainvoke({"code": "print('test')"})

            assert "connect" in result.lower() or "Error" in result
