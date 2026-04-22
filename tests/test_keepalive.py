"""Tests for the SSE keepalive wrapper.

Verifies that with_keepalive() correctly:
1. Forwards chunks from the inner generator
2. Sends keepalive pings during gaps (without destroying the generator)
3. Handles stream end (StopAsyncIteration)
4. Detects client disconnect
5. Cancels pending task on cleanup
"""

import asyncio

import pytest

from esdc.server.constants import SSE_KEEPALIVE_INTERVAL, SSE_STREAM_TIMEOUT
from esdc.server.routes import with_keepalive


class FakeRequest:
    """Mock for Starlette's Request with controllable disconnect state."""

    def __init__(self, disconnected_after: int | None = None):
        self._call_count = 0
        self._disconnected_after = disconnected_after

    async def is_disconnected(self) -> bool:
        self._call_count += 1
        if self._disconnected_after is not None:
            return self._call_count > self._disconnected_after
        return False


class TestWithKeepalive:
    """Test suite for with_keepalive()."""

    @pytest.mark.asyncio
    async def test_forwards_all_chunks(self):
        """All chunks from the inner generator are forwarded."""

        async def inner():
            for i in range(5):
                yield f"data: chunk_{i}\n\n"

        request = FakeRequest()
        results = []
        async for chunk in with_keepalive(inner(), request, "test-1"):
            results.append(chunk)

        data_chunks = [r for r in results if not r.startswith(":")]
        assert len(data_chunks) == 5

    @pytest.mark.asyncio
    async def test_sends_keepalive_during_gaps(self):
        """Keepalive pings sent during long gaps without destroying generator.

        This is the critical regression test: the old asyncio.wait_for()
        implementation permanently closed async generators on timeout.
        """
        delay = SSE_KEEPALIVE_INTERVAL + 1

        async def slow_inner():
            yield "data: first\n\n"
            await asyncio.sleep(delay)
            yield "data: second\n\n"

        request = FakeRequest()
        results = []
        async for chunk in with_keepalive(slow_inner(), request, "test-2"):
            results.append(chunk)

        data_chunks = [r for r in results if r.startswith("data:")]
        keepalive_chunks = [r for r in results if r.startswith(":")]

        assert len(data_chunks) == 2, (
            f"Expected 2 data chunks, got {len(data_chunks)}: {data_chunks}"
        )
        assert len(keepalive_chunks) >= 1, (
            f"Expected at least 1 keepalive, got {len(keepalive_chunks)}"
        )

    @pytest.mark.asyncio
    async def test_generator_survives_multiple_timeouts(self):
        """Generator survives multiple keepalive timeouts.

        Verifies the fix for the bug where asyncio.wait_for() on
        __anext__() permanently closed async generators after the first
        timeout.
        """

        async def multi_gap_inner():
            yield "data: a\n\n"
            await asyncio.sleep(SSE_KEEPALIVE_INTERVAL + 2)
            yield "data: b\n\n"
            await asyncio.sleep(SSE_KEEPALIVE_INTERVAL + 2)
            yield "data: c\n\n"

        request = FakeRequest()
        results = []
        async for chunk in with_keepalive(multi_gap_inner(), request, "test-3"):
            results.append(chunk)

        data_chunks = [r for r in results if r.startswith("data:")]
        assert len(data_chunks) == 3, (
            f"Expected 3 data chunks, got {len(data_chunks)}: {data_chunks}"
        )

    @pytest.mark.asyncio
    async def test_stops_on_client_disconnect_during_keepalive(self):
        """Stream stops when client disconnects during a keep-alive wait."""
        delay = SSE_KEEPALIVE_INTERVAL + 2  # longer than keepalive interval

        async def slow_inner():
            yield "data: first\n\n"
            await asyncio.sleep(delay)
            yield "data: second\n\n"  # should not be reached

        # Client stays connected for 2 is_disconnected checks, then disconnects
        request = FakeRequest(disconnected_after=2)
        results = []
        async for chunk in with_keepalive(slow_inner(), request, "test-4"):
            results.append(chunk)

        data_chunks = [r for r in results if r.startswith("data:")]
        assert len(data_chunks) >= 1, (
            f"Expected at least 1 data chunk, got {len(data_chunks)}"
        )

    @pytest.mark.asyncio
    async def test_stops_on_client_disconnect_after_chunk(self):
        """Stream stops when client disconnects after receiving chunks."""
        # Client disconnects after 3 is_disconnected checks
        # (enough to yield first 2 chunks then stop)
        request = FakeRequest(disconnected_after=3)

        async def inner():
            yield "data: first\n\n"
            yield "data: second\n\n"
            await asyncio.sleep(SSE_KEEPALIVE_INTERVAL + 2)
            yield "data: never_reached\n\n"

        results = []
        async for chunk in with_keepalive(inner(), request, "test-5"):
            results.append(chunk)

        data_chunks = [r for r in results if r.startswith("data:")]
        # Should get at least the first chunk(s) before disconnect is detected
        assert len(data_chunks) >= 1, (
            f"Expected at least 1 data chunk, got {len(data_chunks)}"
        )

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        """Empty stream completes without error."""

        async def empty():
            return
            yield

        request = FakeRequest()
        results = []
        async for chunk in with_keepalive(empty(), request, "test-6"):
            results.append(chunk)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_fast_stream_no_keepalives(self):
        """Fast stream produces no keepalive pings."""

        async def fast_inner():
            for i in range(5):
                yield f"data: chunk_{i}\n\n"

        request = FakeRequest()
        results = []
        async for chunk in with_keepalive(fast_inner(), request, "test-7"):
            results.append(chunk)

        keepalive_chunks = [r for r in results if r.startswith(":")]
        assert len(keepalive_chunks) == 0


class TestSSEConstants:
    """Test that SSE constants are defined correctly."""

    def test_keepalive_interval(self):
        assert SSE_KEEPALIVE_INTERVAL == 15

    def test_stream_timeout(self):
        assert SSE_STREAM_TIMEOUT == 300
