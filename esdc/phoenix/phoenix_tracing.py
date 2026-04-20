from __future__ import annotations

import atexit
import logging

from esdc.phoenix.phoenix_config import PhoenixConfig

logger = logging.getLogger("esdc.phoenix.tracing")

_initialized = False
_tracer_provider = None


def setup_phoenix_tracing() -> bool:
    global _initialized, _tracer_provider

    config = PhoenixConfig.from_config()
    if not config.enabled:
        logger.info("Phoenix tracing disabled (phoenix.enabled is false)")
        return False

    if _initialized:
        logger.debug("Phoenix tracing already initialized, skipping")
        return True

    from opentelemetry import trace as trace_api
    from phoenix.otel import register

    tracer_provider = register(
        endpoint=config.collector_endpoint,
        project_name=config.project_name,
        auto_instrument=True,
        batch=True,
        set_global_tracer_provider=False,
    )

    trace_api.set_tracer_provider(tracer_provider)

    _tracer_provider = tracer_provider
    atexit.register(tracer_provider.shutdown)

    _initialized = True
    logger.info(
        "Phoenix tracing initialized | project=%s endpoint=%s batch=True",
        config.project_name,
        config.collector_endpoint,
    )
    return True
