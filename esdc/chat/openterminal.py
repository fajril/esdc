"""OpenTerminal integration for sandboxed code execution.

Provides Compute Engine (run_command) and Code Interpreter (run_python)
tools for data visualization and file operations.
Compatible with Open Terminal v2 (procman) API.
"""

import asyncio
import logging
import uuid
from typing import Annotated, Any
from urllib.parse import quote

import httpx
from langchain.tools import tool

logger = logging.getLogger("esdc.chat.openterminal")

_OT_CONFIG: dict[str, Any] | None = None
_OW_CONFIG: dict[str, Any] | None = None

_DEFAULT_PACKAGES = (
    "matplotlib, seaborn, pandas, numpy, scipy, statsmodels, scikit-learn, plotly"
)


def get_openterminal_tools() -> list[Any] | None:
    """Return OpenTerminal tool instances if configured, else None.

    Called by agent.py during agent creation to conditionally register
    Compute Engine, File Processing, and View File tools.
    """
    from esdc.configs import Config

    global _OT_CONFIG, _OW_CONFIG
    _OT_CONFIG = Config.get_openterminal_config()
    if not _OT_CONFIG:
        logger.info("[OPENTERM] Not configured, tools not registered")
        return None

    _OW_CONFIG = Config.get_openwebui_config()

    packages = _OT_CONFIG.get("packages", _DEFAULT_PACKAGES)

    run_command.description = (
        "Execute a shell command in a sandboxed Linux environment.\n\n"
        "Use this tool for data visualization, file operations, "
        "and running Python scripts.\n"
        f"The environment has Python with these packages available: {packages}.\n\n"
        "Key guidelines:\n"
        "- For creating plots, use matplotlib/seaborn and save to /home/user/output/\n"
        "- Use timestamped filenames to avoid collisions: "
        "forecast_20260418_143052.png\n"
        "- For multi-step operations, chain commands with &&\n"
        "- If a package is not available, install it first: pip install <package>\n"
        "- Always check the working directory before making changes\n"
        "- Commands run in a sandboxed container — no access to ESDC database\n"
        "- To use data from queries, embed the data directly in the script\n\n"
        "Example workflow for creating a plot:\n"
        "1. Query data using execute_sql\n"
        "2. Construct a Python script with the data embedded\n"
        "3. Save via File Processing, then Compute Engine to execute\n"
        "4. Use View File to display the plot inline in the chat"
    )

    logger.info(
        "[OPENTERM] Configured | url=%s packages=%s openwebui=%s",
        _OT_CONFIG["url"],
        packages[:50],
        "yes" if _OW_CONFIG else "no",
    )

    tools = [run_command, run_python]
    return tools


def _get_config() -> dict[str, Any]:
    """Get cached OpenTerminal config (set during get_openterminal_tools)."""
    if _OT_CONFIG is None:
        msg = "OpenTerminal not configured. Add 'openterminal' section to config.yaml."
        raise ValueError(msg)
    return _OT_CONFIG


def _get_headers() -> dict[str, str]:
    """Build authorization headers from config."""
    config = _get_config()
    headers = {"Content-Type": "application/json"}
    if config.get("api_key"):
        headers["Authorization"] = f"Bearer {config['api_key']}"
    return headers


def _extract_output(result: dict[str, Any]) -> str:
    """Extract output text from v2 API response.

    Open Terminal v2 returns output as a list of entries:
    [{"type": "output", "data": "..."}, ...]

    Args:
        result: Response JSON from /execute endpoint.

    Returns:
        Concatenated output string.
    """
    raw_output = result.get("output", "")
    if isinstance(raw_output, list):
        parts = []
        for entry in raw_output:
            if isinstance(entry, dict) and entry.get("type") == "output":
                parts.append(entry.get("data", ""))
        return "".join(parts).rstrip()
    if isinstance(raw_output, str):
        # v1 format fallback
        return raw_output
    return str(raw_output)


async def _poll_until_done(
    client: httpx.AsyncClient,
    config: dict[str, Any],
    process_id: str,
    max_wait: int,
) -> dict[str, Any]:
    """Poll /execute/{id}/status until command completes or timeout.

    Args:
        client: HTTPX client instance.
        config: OpenTerminal configuration dict.
        process_id: Process ID from /execute response.
        max_wait: Maximum seconds to wait.

    Returns:
        Final status response from API.
    """
    url = f"{config['url']}/execute/{process_id}/status"
    poll_interval = 0.5
    elapsed = 0.0

    while elapsed < max_wait:
        # Use wait param to reduce polling frequency
        params = {"wait": min(poll_interval * 2, 5)}
        try:
            response = await client.get(url, params=params, headers=_get_headers())
            response.raise_for_status()
            result = response.json()

            status = result.get("status", "unknown")
            if status in ("done", "killed", "error"):
                return result

        except httpx.HTTPStatusError as e:
            logger.error(
                "[OPENTERM] poll HTTP_ERROR | status=%d process_id=%s",
                e.response.status_code,
                process_id,
            )
            return {
                "status": "error",
                "exit_code": -1,
                "output": [{"type": "output", "data": f"HTTP error: {e}"}],
            }

        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    # Timeout: try to kill the process
    logger.warning(
        "[OPENTERM] poll TIMEOUT | process_id=%s elapsed=%.1fs", process_id, elapsed
    )
    try:
        await client.delete(
            f"{config['url']}/execute/{process_id}",
            headers=_get_headers(),
        )
    except Exception as e:
        logger.warning("[OPENTERM] kill process failed: %s", e)

    return {
        "status": "timeout",
        "exit_code": -1,
        "output": [{"type": "output", "data": f"Command timed out after {max_wait}s"}],
    }


@tool("Compute Engine")
async def run_command(
    command: Annotated[
        str,
        (
            "Shell command to execute. Supports chaining (&&, ||, ;), "
            "pipes (|), and redirections."
        ),
    ],
    cwd: Annotated[
        str | None,
        "Working directory for the command. Defaults to /home/user.",
    ] = None,
) -> str:
    """Execute a shell command in a sandboxed Linux environment.

    Use this tool for data visualization, file operations, and running Python scripts.
    The environment has Python available. For creating plots, save to /home/user/.
    """
    config = _get_config()
    timeout = config.get("timeout", 120)
    url = f"{config['url']}/execute"

    payload: dict[str, Any] = {"command": command}
    if cwd:
        payload["cwd"] = cwd

    # v2 API: use wait parameter for synchronous behavior
    # If command finishes within wait period, output is included inline
    params = {"wait": timeout}

    logger.info("[OPENTERM] run_command | cmd=%s", command[:100])

    try:
        async with httpx.AsyncClient(timeout=timeout + 30) as client:
            response = await client.post(
                url,
                json=payload,
                params=params,
                headers=_get_headers(),
            )
            response.raise_for_status()
            result = response.json()

        status = result.get("status", "unknown")
        exit_code = result.get("exit_code", -1)

        # If still running after wait, poll for completion
        if status == "running" and result.get("id"):
            process_id = result["id"]
            logger.debug(
                "[OPENTERM] Command still running, polling | process_id=%s", process_id
            )
            result = await _poll_until_done(client, config, process_id, timeout)
            status = result.get("status", "unknown")
            exit_code = result.get("exit_code", -1)

        output = _extract_output(result)

        if exit_code != 0:
            logger.warning(
                "[OPENTERM] run_command FAILED | exit_code=%d cmd=%s",
                exit_code,
                command[:80],
            )
            return f"Command exited with code {exit_code}.\nOutput:\n{output}"

        logger.info(
            "[OPENTERM] run_command OK | output_len=%d",
            len(output),
        )
        return output

    except httpx.TimeoutException:
        logger.error("[OPENTERM] run_command TIMEOUT | cmd=%s", command[:80])
        return (
            f"Error: Command timed out after {timeout}s. Try simplifying the command."
        )
    except httpx.ConnectError:
        logger.error("[OPENTERM] run_command CONNECT_ERROR | url=%s", url)
        return (
            f"Error: Cannot connect to Compute Engine at {config['url']}. "
            "The service may not be running."
        )
    except httpx.HTTPStatusError as e:
        logger.error(
            "[OPENTERM] run_command HTTP_ERROR | status=%d url=%s",
            e.response.status_code,
            url,
        )
        return (
            f"Error: Compute Engine returned HTTP "
            f"{e.response.status_code}: {e.response.text[:500]}"
        )
    except Exception as e:
        logger.error("[OPENTERM] run_command ERROR | error=%s", e)
        return f"Error: {e}"


@tool("Code Interpreter")
async def run_python(
    code: Annotated[
        str,
        (
            "Python code to execute. Visible in OpenWebUI tool panel. "
            "For visualizations, use the pre-defined `output_image_path` variable "
            "to save images. When this tool returns image markdown, copy it "
            "into your final response so the user can see the image inline."
        ),
    ],
) -> str:
    """Execute Python code with automatic script cleanup and image display.

    The code is written to a temporary file, executed, and the file is
    automatically deleted. Images saved to `output_image_path` are displayed
    inline via OpenWebUI proxy.

    IMPORTANT: When this tool result contains "![Generated Plot](http://...)",
    include that image markdown directly in your final response so the image
    renders inline for the user.

    Example:
        code="import matplotlib.pyplot as plt; plt.plot([1,2,3]); "
             "plt.savefig(output_image_path); print('Done')"
    """
    from urllib.parse import quote

    config = _get_config()

    # Generate UUID for this execution
    img_uuid = str(uuid.uuid4())
    output_path = f"/home/user/img/{img_uuid}.png"

    # Inject output_image_path variable into the code
    injected_code = f'output_image_path = "{output_path}"\n{code}'

    # Generate UUID for temp script
    script_uuid = str(uuid.uuid4())
    script_path = f"/tmp/script_{script_uuid}.py"

    # Ensure /home/user/img directory exists
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"{config['url']}/execute",
                json={"command": "mkdir -p /home/user/img"},
                params={"wait": 5},
                headers=_get_headers(),
            )
    except Exception:
        pass  # Directory might already exist

    try:
        async with httpx.AsyncClient(timeout=150) as client:
            # Step 1: Write code to temp file
            write_payload = {"path": script_path, "content": injected_code}
            write_response = await client.post(
                f"{config['url']}/files/write",
                json=write_payload,
                headers=_get_headers(),
            )
            write_response.raise_for_status()

            # Step 2: Execute the script
            exec_payload = {"command": f"python {script_path}"}
            exec_params = {"wait": 120}

            exec_response = await client.post(
                f"{config['url']}/execute",
                json=exec_payload,
                params=exec_params,
                headers=_get_headers(),
            )
            exec_response.raise_for_status()
            result_data = exec_response.json()

            # Poll if still running
            status = result_data.get("status", "unknown")
            exit_code = result_data.get("exit_code", -1)

            if status == "running" and result_data.get("id"):
                process_id = result_data["id"]
                poll_result = await _poll_until_done(client, config, process_id, 120)
                status = poll_result.get("status", "unknown")
                exit_code = poll_result.get("exit_code", -1)
                result_data = poll_result

            output = _extract_output(result_data)

            # Step 3: Always delete temp script
            try:
                await client.delete(
                    f"{config['url']}/execute/{result_data.get('id', '')}",
                    headers=_get_headers(),
                )
            except Exception:
                # Try alternative cleanup
                try:
                    await client.post(
                        f"{config['url']}/execute",
                        json={"command": f"rm -f {script_path}"},
                        params={"wait": 5},
                        headers=_get_headers(),
                    )
                except Exception:
                    logger.warning(
                        "[OPENTERM] Failed to cleanup temp script: %s", script_path
                    )

            # Step 4: Check if image was generated and display it
            if output_path in output:
                # Image was saved — build proxy URL
                try:
                    ow_config = _get_ow_config()
                    proxy_url = ow_config.get("proxy_url", ow_config["url"]).rstrip("/")
                    server_id = ow_config["terminal_server_id"]
                except ValueError:
                    # OWUI not configured — fall back to direct OT URL
                    proxy_url = config["url"].rstrip("/")
                    server_id = ""

                encoded_path = quote(output_path, safe="/")
                if server_id:
                    file_url = (
                        f"{proxy_url}/api/v1/terminals/{server_id}"
                        f"/files/read?path={encoded_path}"
                    )
                else:
                    file_url = f"{proxy_url}/files/read?path={encoded_path}"

                return (
                    f"Execution complete (exit code: {exit_code}):\n\n"
                    f"```\n{output}\n```\n\n"
                    f"![Generated Plot]({file_url})"
                )

            # Return without image if output_path not found in stdout
            return f"Execution complete (exit code: {exit_code}):\n\n```\n{output}\n```"

    except httpx.TimeoutException:
        logger.error("[OPENTERM] run_python TIMEOUT")
        return "Error: Python execution timed out after 120s."
    except httpx.ConnectError:
        logger.error("[OPENTERM] run_python CONNECT_ERROR")
        return "Error: Cannot connect to Compute Engine."
    except Exception as e:
        logger.error("[OPENTERM] run_python ERROR | %s", e)
        return f"Error: {e}"


def _get_ow_config() -> dict[str, Any]:
    """Get cached OpenWebUI config (set during get_openterminal_tools)."""
    if _OW_CONFIG is None:
        msg = "OpenWebUI not configured. Add 'openwebui' section to config.yaml."
        raise ValueError(msg)
    return _OW_CONFIG


def _build_file_url(filepath: str, description: str = "") -> str:
    """Build an OpenWebUI proxy URL for a file on the Compute Engine.

    Constructs a URL that routes through OpenWebUI's reverse proxy to
    access files on the Compute Engine sandbox. Uses proxy_url (the
    browser-facing URL) rather than the server-side url.

    Args:
        filepath: Absolute path to the file on the Compute Engine.
        description: Optional alt text for the image.

    Returns:
        Markdown image string: ![description](proxy_url)
    """
    ow_config = _get_ow_config()
    proxy_url = ow_config.get("proxy_url", ow_config["url"]).rstrip("/")
    server_id = ow_config["terminal_server_id"]
    encoded_path = quote(filepath, safe="/")
    file_url = (
        f"{proxy_url}/api/v1/terminals/{server_id}/files/read?path={encoded_path}"
    )
    desc = description or filepath.split("/")[-1]
    return f"![{desc}]({file_url})"
