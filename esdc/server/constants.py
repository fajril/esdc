"""SSE and streaming constants shared across server modules.

Defined here to avoid circular imports between routes, agent_wrapper,
and responses_wrapper.
"""

SSE_KEEPALIVE_INTERVAL = 15
SSE_STREAM_TIMEOUT = 300
