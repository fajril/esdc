# tests/corpus/test_llama_teardown.py
"""A llama.cpp model that outlives Python must not abort the process.

ggml-metal's dylib destructor runs at exit() and asserts every residency
set was released (ggml-metal-device.m: GGML_ASSERT([rsets->data count] ==
0)). Any Llama still alive when CPython finishes finalizing trips it and
the process dies with SIGABRT -- exit 134 -- after the work already
succeeded. Reported symptom: `esdc chat` and the full pytest run both
exit 134 with every result already printed.

Each case runs in a subprocess on purpose: the abort is a native SIGABRT,
so an in-process check would kill the test runner instead of failing.
The child parks the model in a daemon thread's frame, which is the
cheapest reliable way to keep a reference past interpreter finalization
(pytest's retained tracebacks do the same thing by accident).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from esdc.corpus.llama_backend import (
    EMBED_FILE,
    EMBED_REPO,
    RERANK_FILE,
    RERANK_REPO,
)

pytestmark = pytest.mark.integration


def _cached(repo: str, filename: str) -> bool:
    """True when the GGUF is already in the local cache (no download)."""
    from huggingface_hub import hf_hub_download

    from esdc.configs import Config

    try:
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            cache_dir=str(Config.get_config_dir() / "models"),
            local_files_only=True,
        )
    except Exception:
        return False
    return True


_HOLD = """
import threading, time
{load}
threading.Thread(
    target=lambda held: time.sleep(3600), args=(model,), daemon=True
).start()
time.sleep(0.2)
"""

_EMBED_CHILD = _HOLD.format(
    load=(
        "from esdc.embedders import InternalEmbedder, _get_model\n"
        "InternalEmbedder().generate_embedding('teardown probe')\n"
        "model = _get_model()"
    )
)

_RERANK_CHILD = _HOLD.format(
    load=(
        "from esdc.corpus.reranker import Reranker\n"
        "rr = Reranker.get()\n"
        "assert rr is not None, 'reranker failed to load'\n"
        "rr.rerank('teardown probe', ['a chunk of text'])\n"
        "model = rr"
    )
)


def _run_child(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
    )


def _assert_clean_exit(proc: subprocess.CompletedProcess[str]) -> None:
    assert proc.returncode == 0, (
        f"child exited {proc.returncode} "
        "(134 = SIGABRT from the ggml-metal teardown assert)\n"
        f"{proc.stderr[-2000:]}"
    )


@pytest.mark.skipif(
    not _cached(EMBED_REPO, EMBED_FILE), reason="embedding GGUF not cached"
)
def test_embedding_model_held_past_shutdown_exits_cleanly():
    _assert_clean_exit(_run_child(_EMBED_CHILD))


@pytest.mark.skipif(
    not _cached(RERANK_REPO, RERANK_FILE), reason="reranker GGUF not cached"
)
def test_reranker_held_past_shutdown_exits_cleanly():
    _assert_clean_exit(_run_child(_RERANK_CHILD))
