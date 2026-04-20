from __future__ import annotations

import logging

from esdc.phoenix.phoenix_config import PhoenixConfig

logger = logging.getLogger("esdc.phoenix.tracing")

_initialized = False


def setup_phoenix_tracing() -> bool:
    global _initialized

    config = PhoenixConfig.from_config()
    if not config.enabled:
        logger.info("Phoenix tracing disabled (phoenix.enabled is false)")
        return False

    if _initialized:
        logger.debug("Phoenix tracing already initialized, skipping")
        return True

    from phoenix.otel import register

    register(
        endpoint=config.collector_endpoint,
        project_name=config.project_name,
        auto_instrument=True,
    )

    _initialized = True
    logger.info(
        "Phoenix tracing initialized | project=%s endpoint=%s",
        config.project_name,
        config.collector_endpoint,
    )
    return True
