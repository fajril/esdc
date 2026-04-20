from __future__ import annotations

import logging

from esdc.phoenix.phoenix_config import PhoenixConfig

logger = logging.getLogger("esdc.phoenix.tracing")

_initialized = False


def setup_phoenix_tracing() -> bool:
    global _initialized

    config = PhoenixConfig.from_env()
    if not config.enabled:
        logger.info("Phoenix tracing disabled (PHOENIX_ENABLED not set)")
        return False

    if _initialized:
        logger.debug("Phoenix tracing already initialized, skipping")
        return True

    from phoenix.otel import register

    register(
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
