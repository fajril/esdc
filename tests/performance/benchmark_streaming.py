"""Benchmark streaming response generation."""

import uuid

import pytest


class TestStreamingPerformance:
    """Benchmark streaming utilities."""

    def test_uuid_generation_baseline(self, benchmark):
        """Benchmark: UUID generation per chunk (baseline)."""

        def generate_100_uuids():
            chunk_ids = []
            for _ in range(100):
                chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                chunk_ids.append(chunk_id)
            return chunk_ids

        result = benchmark(generate_100_uuids)
        assert len(result) == 100

    def test_uuid_generation_cached(self, benchmark):
        """Benchmark: UUID generation with stream-level caching."""

        def generate_100_uuids_cached():
            stream_uuid = uuid.uuid4().hex[:12]
            chunk_ids = []
            for i in range(100):
                chunk_id = f"chatcmpl-{stream_uuid}-{i:04d}"
                chunk_ids.append(chunk_id)
            return chunk_ids

        result = benchmark(generate_100_uuids_cached)
        assert len(result) == 100


if __name__ == "__main__":
    pytest.main([__file__, "--benchmark-only", "-v"])
