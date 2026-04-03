"""Logging configuration and setup for ESDC server."""

# Standard library
import logging
import logging.handlers
import sys
from typing import Any


def parse_size(size_str: str) -> int:
    """Parse size string like '10MB' to bytes.

    Args:
        size_str: Size string (e.g., '10MB', '1GB', '500KB')

    Returns:
        Size in bytes
    """
    size_str = size_str.upper().strip()

    multipliers = {
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
    }

    for suffix, multiplier in multipliers.items():
        if size_str.endswith(suffix):
            return int(size_str[: -len(suffix)]) * multiplier

    # Assume bytes if no suffix
    return int(size_str)


def setup_server_logging(config: dict[str, Any]) -> logging.Logger:
    """Setup logging for ESDC server.

    Args:
        config: Logging configuration from Config.get_logging_config()

    Returns:
        Logger instance
    """
    from esdc.configs import Config

    server_config = config.get("server", {})
    global_level = config.get("level", "INFO")

    # Get log directory
    log_dir = Config.get_config_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("esdc.server")
    logger.setLevel(getattr(logging, global_level.upper(), logging.INFO))

    # Remove existing handlers
    logger.handlers.clear()

    # File handler
    file_config = server_config.get("file", {})
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    if file_config.get("enabled", True):
        log_path = file_config.get("path", "logs/esdc_server.log")
        log_file = Config.get_config_dir() / log_path
        log_file.parent.mkdir(parents=True, exist_ok=True)

        max_size = parse_size(file_config.get("max_size", "10MB"))
        backup_count = file_config.get("backup_count", 5)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count,
        )
        file_handler.setLevel(
            getattr(logging, server_config.get("level", "INFO").upper(), logging.INFO)
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

    # Console handler
    console_config = server_config.get("console", {})
    if console_config.get("enabled", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(
            getattr(logging, server_config.get("level", "INFO").upper(), logging.INFO)
        )
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)

    # Suppress verbose library logs
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("markdown_it").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger.info("ESDC Server logging initialized")

    return logger


def get_request_logger() -> logging.Logger:
    """Get request logger for middleware."""
    return logging.getLogger("esdc.server.middleware")


def get_agent_logger() -> logging.Logger:
    """Get agent logger for streaming responses."""
    return logging.getLogger("esdc.server.agent")
