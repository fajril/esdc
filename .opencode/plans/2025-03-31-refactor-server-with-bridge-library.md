# Refactor ESDC Server dengan langchain-openai-api-bridge

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor implementasi manual OpenAI-compatible API server menggunakan library `langchain-openai-api-bridge` yang sudah teruji dan memiliki support streaming yang baik.

**Architecture:** Gunakan `langchain-openai-api-bridge` untuk mengekspos ESDC agent sebagai OpenAI-compatible API. Library ini menyediakan FastAPI extension yang menangani streaming, threading, dan format response OpenAI secara otomatis.

**Tech Stack:** 
- `langchain-openai-api-bridge` (baru)
- FastAPI (existing)
- LangGraph Agent (existing)
- uvicorn (existing)

---

## Overview

Implementasi manual saat ini (`esdc/server/routes.py`, `esdc/server/agent_wrapper.py`) bermasalah dengan streaming response di OpenWebUI. Library `langchain-openai-api-bridge` menyediakan solusi yang sudah teruji dengan 91+ stars dan maintenance aktif.

**Keuntungan menggunakan library:**
1. Streaming SSE yang sudah teruji dengan OpenWebUI
2. Error handling robust
3. Format OpenAI 100% compatible
4. Kurangi kode dari ~400 lines menjadi ~100 lines
5. Support Chat Completions dan Assistant API

---

## Prerequisites

**Pastikan:**
- `esdc chat` pernah berjalan dan provider sudah dikonfigurasi
- Database ESDC sudah ada (hasil `esdc fetch --save`)

---

## Task 1: Add langchain-openai-api-bridge Dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add dependency**

Tambahkan ke `[project]dependencies`:

```toml
dependencies = [
    # ... existing dependencies ...
    "langchain-openai-api-bridge[langchain,langchain-serve]>=1.0.0",
    # Web server
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
    "sse-starlette>=1.6.0",
]
```

**Step 2: Sync dependencies**

Run: `uv sync`

Expected: Dependencies terinstall tanpa error

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(deps): add langchain-openai-api-bridge dependency"
```

---

## Task 2: Create ESDC Agent Factory

**Files:**
- Create: `esdc/server/agent_factory.py`

**Step 1: Create AgentFactory class**

```python
# esdc/server/agent_factory.py
"""Agent Factory untuk langchain-openai-api-bridge."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai_api_bridge.assistant import AgentFactory, CreateLLMDto

from esdc.chat.agent import create_agent
from esdc.configs import Config
from esdc.providers import create_llm_from_config


class ESDCAgentFactory(AgentFactory):
    """Factory untuk membuat ESDC agent."""

    def create_agent(self, dto: CreateLLMDto) -> Runnable:
        """Create ESDC agent dengan tools."""
        llm = self.create_llm(dto=dto)
        
        # Create agent tanpa checkpointer (stateless untuk API)
        agent = create_agent(llm, checkpointer=None)
        
        return agent

    def create_llm(self, dto: CreateLLMDto) -> BaseChatModel:
        """Create LLM dari config."""
        # Get provider config
        provider_config = Config.get_provider_config()
        
        if not provider_config:
            raise ValueError(
                "No provider configured. Please run 'esdc provider add' first."
            )
        
        # Use model from request if provided, otherwise from config
        model = dto.model or provider_config.get("model")
        
        # Build config dict
        config = {
            "provider_type": provider_config.get("provider", "ollama"),
            "model": model,
            "base_url": provider_config.get("base_url"),
            "api_key": provider_config.get("api_key"),
        }
        
        return create_llm_from_config(config)
```

**Step 2: Test import**

Run: `uv run python -c "from esdc.server.agent_factory import ESDCAgentFactory; print('Import successful')"`

Expected: "Import successful"

**Step 3: Commit**

```bash
git add esdc/server/agent_factory.py
git commit -m "feat(server): add ESDCAgentFactory for langchain-openai-api-bridge"
```

---

## Task 3: Refactor Server App

**Files:**
- Modify: `esdc/server/app.py` (rewrite)
- Delete: `esdc/server/routes.py`
- Delete: `esdc/server/agent_wrapper.py`

**Step 1: Rewrite app.py**

```python
# esdc/server/app.py
"""FastAPI application using langchain-openai-api-bridge."""

# Standard library
import logging

# Third-party
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai_api_bridge.fastapi import LangchainOpenaiApiBridgeFastAPI
from langchain_openai_api_bridge.assistant import (
    InMemoryMessageRepository,
    InMemoryRunRepository,
    InMemoryThreadRepository,
)
import uvicorn

# Local
from esdc.configs import Config
from esdc.server.agent_factory import ESDCAgentFactory

# Logger
logger = logging.getLogger("esdc.server")


def create_app() -> FastAPI:
    """Create FastAPI application with langchain-openai-api-bridge."""
    app = FastAPI(
        title="ESDC API",
        description="OpenAI-compatible API for ESDC agent",
        version="0.4.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Setup repositories
    thread_repository = InMemoryThreadRepository()
    message_repository = InMemoryMessageRepository()
    run_repository = InMemoryRunRepository()

    # Setup bridge
    bridge = LangchainOpenaiApiBridgeFastAPI(
        app=app,
        agent_factory_provider=lambda: ESDCAgentFactory(),
    )

    # Bind OpenAI Assistant API
    bridge.bind_openai_assistant_api(
        thread_repository_provider=thread_repository,
        message_repository_provider=message_repository,
        run_repository_provider=run_repository,
        prefix="/v1",
    )

    return app


def run_server(
    host: str = "0.0.0.0", port: int = 3334, log_level: str = "info"
) -> None:
    """Run the web server.

    Args:
        host: Server host address
        port: Server port
        log_level: Uvicorn log level
    """
    # Initialize configuration
    Config.init_config()

    # Create FastAPI app
    app = create_app()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    logger.info(f"Starting ESDC server on http://{host}:{port}")
    logger.info(f"API documentation available at http://{host}:{port}/docs")

    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
    )
```

**Step 2: Delete old files**

Run: `rm esdc/server/routes.py esdc/server/agent_wrapper.py`

**Step 3: Test import**

Run: `uv run python -c "from esdc.server.app import create_app, run_server; print('Import successful')"`

Expected: "Import successful"

**Step 4: Commit**

```bash
git add esdc/server/app.py
git rm esdc/server/routes.py esdc/server/agent_wrapper.py
git commit -m "refactor(server): use langchain-openai-api-bridge for OpenAI-compatible API

- Replace manual implementation with langchain-openai-api-bridge
- Add ESDCAgentFactory for agent creation
- Remove routes.py and agent_wrapper.py
- Support streaming via bridge library"
```

---

## Task 4: Update __init__.py

**Files:**
- Modify: `esdc/server/__init__.py`

**Step 1: Update exports**

```python
# esdc/server/__init__.py
"""ESDC Server module for OpenAI-compatible API."""

from esdc.server.app import create_app, run_server
from esdc.server.agent_factory import ESDCAgentFactory

__all__ = ["create_app", "run_server", "ESDCAgentFactory"]
```

**Step 2: Commit**

```bash
git add esdc/server/__init__.py
git commit -m "chore(server): update __init__ exports"
```

---

## Task 5: Run Linting and Type Checking

**Files:**
- All modified files

**Step 1: Run ruff**

Run: `uv run ruff check esdc/server/`

Expected: All checks passed

**Step 2: Run ruff format**

Run: `uv run ruff format esdc/server/`

Expected: Files formatted

**Step 3: Run basedpyright**

Run: `uv run basedpyright esdc/server/`

Expected: 0 errors, 0 warnings

**Step 4: Test imports**

Run: `uv run python -c "from esdc.server import create_app, run_server, ESDCAgentFactory; print('All imports successful')"`

Expected: "All imports successful"

**Step 5: Commit**

```bash
git add -A
git commit -m "style(server): fix linting and type checking"
```

---

## Task 6: Test Server

**Files:**
- Manual testing

**Step 1: Start server**

Run: `uv run esdc serve --port 3334 --log-level info`

Server akan start di http://0.0.0.0:3334

**Step 2: Test /v1/models**

Di terminal lain:
```bash
curl http://localhost:3334/v1/models
```

Expected: JSON response dengan daftar models

**Step 3: Test chat completion**
```bash
curl -X POST http://localhost:3334/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "esdc-agent",
    "messages": [{"role": "user", "content": "berapa cadangan lapangan abadi?"}]
  }'
```

Expected: Response dari agent (bisa memakan waktu beberapa detik)

**Step 4: Test dengan OpenWebUI**
1. Buka OpenWebUI
2. Settings → Connections → OpenAI
3. Base URL: `http://localhost:3334/v1`
4. Model: `esdc-agent`
5. Test chat

---

## Summary

Setelah implementasi:
1. Server menggunakan library yang teruji
2. Streaming response berfungsi dengan baik
3. Format OpenAI 100% compatible
4. Error handling robust
5. Code lebih sedikit dan maintainable

**Next Steps:**
- Jika testing berhasil, commit dan push
- Jika ada issue, debug dengan log file

---

## Troubleshooting

**Issue: Provider not configured**
Solution: Jalankan `esdc provider add` terlebih dahulu

**Issue: Database not found**
Solution: Jalankan `esdc fetch --save` untuk mengisi database

**Issue: Model not responding**
- Cek log di console
- Pastikan LLM provider berjalan (e.g., Ollama server)
- Test dengan `esdc chat` terlebih dahulu
