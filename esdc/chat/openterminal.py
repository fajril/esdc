import logging
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

    tools = [run_command, write_file]
    if _OW_CONFIG:
        tools.append(view_file)
        view_file.description = (
            "Display a file from the Compute Engine sandbox inline in the chat.\n\n"
            "Use this tool after creating plots or other visual output to show them "
            "to the user. Returns a markdown image link that renders inline.\n\n"
            "Supported file types:\n"
            "- Images: .png, .jpg, .jpeg, .gif, .svg, .webp\n"
            "- Always use this after Compute Engine produces a plot file\n\n"
            "The file must already exist on the Compute Engine filesystem.\n"
            "Typical paths: /home/user/output/forecast_20260418_143052.png"
        )
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
    url = f"{config['url']}/execute"
    timeout = config.get("timeout", 120)

    payload: dict[str, Any] = {"command": command}
    if cwd:
        payload["cwd"] = cwd

    logger.info("[OPENTERM] run_command | cmd=%s", command[:100])

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                json=payload,
                headers=_get_headers(),
            )
            response.raise_for_status()
            result = response.json()

        output = result.get("output", "")
        exit_code = result.get("exit_code", -1)

        if exit_code != 0:
            logger.warning(
                "[OPENTERM] run_command FAILED | exit_code=%d cmd=%s",
                exit_code,
                command[:80],
            )
            return (
                f"Command exited with code {exit_code}.\n"
                f"Output:\n{output}\n"
                f"Error output:\n{result.get('error', '')}"
            )

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


@tool("File Processing")
async def write_file(
    path: Annotated[
        str,
        (
            "Absolute or relative path to write to. "
            "Parent directories are created automatically."
        ),
    ],
    content: Annotated[
        str,
        (
            "Text content to write to the file. "
            "For Python scripts, include proper indentation."
        ),
    ],
) -> str:
    """Write text content to a file in the sandboxed environment.

    Use this tool to save Python scripts, data files, or configuration files
    before executing them with run_command. Parent directories are created
    automatically.

    Common patterns:
    - Save a Python plot script, then run it with run_command
    - Save data as CSV/JSON for processing
    - Save intermediate results for multi-step analysis
    """
    config = _get_config()
    url = f"{config['url']}/files/write"
    timeout = config.get("write_timeout", 30)

    payload = {"path": path, "content": content}

    logger.info("[OPENTERM] write_file | path=%s len=%d", path, len(content))

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                json=payload,
                headers=_get_headers(),
            )
            response.raise_for_status()

        logger.info("[OPENTERM] write_file OK | path=%s", path)
        return f"File written successfully: {path}"

    except httpx.TimeoutException:
        logger.error("[OPENTERM] write_file TIMEOUT | path=%s", path)
        return f"Error: Write timed out after {timeout}s."
    except httpx.ConnectError:
        logger.error("[OPENTERM] write_file CONNECT_ERROR | url=%s", config["url"])
        return (
            f"Error: Cannot connect to Compute Engine at {config['url']}. "
            "The service may not be running."
        )
    except httpx.HTTPStatusError as e:
        logger.error(
            "[OPENTERM] write_file HTTP_ERROR | status=%d path=%s",
            e.response.status_code,
            path,
        )
        return (
            f"Error: Compute Engine returned HTTP "
            f"{e.response.status_code}: {e.response.text[:500]}"
        )
    except Exception as e:
        logger.error("[OPENTERM] write_file ERROR | error=%s", e)
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
    access files on the Compute Engine sandbox. The browser renders
    these URLs inline because OpenWebUI proxies the request with
    the user's auth cookie.

    Args:
        filepath: Absolute path to the file on the Compute Engine.
        description: Optional alt text for the image.

    Returns:
        Markdown image string: ![description](proxy_url)
    """
    ow_config = _get_ow_config()
    ow_url = ow_config["url"].rstrip("/")
    server_id = ow_config["terminal_server_id"]
    encoded_path = quote(filepath, safe="/")
    proxy_url = f"{ow_url}/api/v1/terminals/{server_id}/files/read?path={encoded_path}"
    desc = description or filepath.split("/")[-1]
    return f"![{desc}]({proxy_url})"


@tool("View File")
async def view_file(
    path: Annotated[
        str,
        (
            "Absolute path to the file on the Compute Engine sandbox. "
            "Typically /home/user/output/filename.png"
        ),
    ],
    description: Annotated[
        str,
        (
            "Short description of the file, used as alt text for images. "
            "E.g. 'Sales Forecast Plot for Abadi Field'"
        ),
    ] = "",
) -> str:
    """Display a file from the Compute Engine sandbox inline in the chat.

    Use this tool after creating plots or other visual output with Compute Engine
    to show them to the user. Returns a markdown image link that renders inline.

    The file must already exist on the Compute Engine filesystem.
    Typical paths: /home/user/output/forecast_20260418_143052.png
    """
    config = _get_config()

    logger.info("[OPENTERM] view_file | path=%s desc=%s", path, description[:50])

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.head(
                f"{config['url']}/files/read",
                params={"path": path},
                headers=_get_headers(),
            )

        if response.status_code == 404:
            logger.warning("[OPENTERM] view_file NOT_FOUND | path=%s", path)
            return f"Error: File not found at {path}. Check the path and try again."

        url = _build_file_url(path, description)
        logger.info("[OPENTERM] view_file OK | path=%s url_len=%d", path, len(url))
        return url

    except httpx.ConnectError:
        logger.error("[OPENTERM] view_file CONNECT_ERROR | url=%s", config["url"])
        return (
            f"Error: Cannot connect to Compute Engine at {config['url']}. "
            "The service may not be running."
        )
    except httpx.TimeoutException:
        logger.error("[OPENTERM] view_file TIMEOUT | path=%s", path)
        return f"Error: File check timed out for {path}."
    except ValueError as e:
        logger.error("[OPENTERM] view_file CONFIG_ERROR | error=%s", e)
        return f"Error: {e}"
    except Exception as e:
        logger.error("[OPENTERM] view_file ERROR | error=%s", e)
        return f"Error: {e}"
