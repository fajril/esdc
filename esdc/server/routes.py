# Standard library
import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator

# Third-party
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

# Local
from esdc.server.agent_wrapper import generate_response, generate_streaming_response
from esdc.server.constants import SSE_KEEPALIVE_INTERVAL
from esdc.server.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    ModelInfo,
    ModelList,
)
from esdc.server.responses_models import ResponsesRequest
from esdc.server.responses_wrapper import (
    generate_responses_stream,
    generate_responses_sync,
)

router = APIRouter()
logger = logging.getLogger("esdc.server.routes")


async def with_keepalive(
    stream: AsyncGenerator[str, None],
    request_obj: Request,
    request_id: str,
) -> AsyncGenerator[str, None]:
    r"""Wrap an SSE stream with keep-alive comments and disconnect detection.

    Sends ``: keep-alive\n\n`` SSE comments every
    ``SSE_KEEPALIVE_INTERVAL`` seconds when no data is flowing,
    preventing cloudflared tunnel timeouts (100s default).
    Also checks for client disconnection.

    Uses ``asyncio.wait()`` with a persistent pending task instead of
    ``asyncio.wait_for()``.  This is critical because ``wait_for()``
    cancels the underlying coroutine on timeout, which permanently
    closes async generators — destroying the stream after the first
    keep-alive ping.  The ``wait()`` approach keeps the pending
    ``__anext__()`` task alive across timeouts.
    """
    _stream_end = object()

    async def get_next():
        try:
            return await stream_aiter.__anext__()
        except StopAsyncIteration:
            return _stream_end

    stream_aiter = stream.__aiter__()
    yield_counter = 0
    keepalive_counter = 0
    pending_task = asyncio.ensure_future(get_next())

    try:
        while True:
            done, _ = await asyncio.wait(
                {pending_task},
                timeout=SSE_KEEPALIVE_INTERVAL,
            )

            if done:
                result = pending_task.result()
                if result is _stream_end:
                    return

                if await request_obj.is_disconnected():
                    logger.info(
                        "[REQUEST %s] Client disconnected after"
                        " %d yields, stopping stream",
                        request_id,
                        yield_counter,
                    )
                    return

                yield_counter += 1
                assert isinstance(result, str)
                yield result
                pending_task = asyncio.ensure_future(get_next())
                keepalive_counter = 0
            else:
                if await request_obj.is_disconnected():
                    logger.info(
                        "[REQUEST %s] Client disconnected during"
                        " keep-alive wait, stopping stream",
                        request_id,
                    )
                    return

                keepalive_counter += 1
                logger.debug(
                    "[REQUEST %s] Keep-alive #%d sent (%d yields so far)",
                    request_id,
                    keepalive_counter,
                    yield_counter,
                )
                yield ": keep-alive\n\n"
    finally:
        pending_task.cancel()


@router.get("/models")
async def list_models() -> ModelList:
    """List available models.

    Returns:
        List of available models in OpenAI-compatible format.
    """
    return ModelList(
        data=[
            ModelInfo(
                id="iris",
                created=int(time.time()),
                owned_by="IRIS",
            )
        ]
    )


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    request_obj: Request,
):
    """Create chat completion.

    Handles both streaming and non-streaming chat completion requests
    in OpenAI-compatible format.

    Args:
        request: Chat completion request
        request_obj: FastAPI request object for header access

    Returns:
        StreamingResponse for streaming requests, ChatCompletionResponse otherwise
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # Calculate message hash for deduplication detection
    messages_hash = hashlib.md5(
        json.dumps([m.model_dump() for m in request.messages], sort_keys=True).encode()
    ).hexdigest()[:8]

    # Log request details
    conv_id = getattr(request, "conversation_id", "none")
    msg_count = len(request.messages) if request.messages else 0

    # Safe access to messages
    first_msg_preview = "empty"
    last_msg_preview = "empty"
    if request.messages and len(request.messages) > 0:
        first_msg_preview = (
            request.messages[0].content[:30].replace("\n", " ") + "..."
            if request.messages[0].content
            else "empty"
        )
        last_msg_preview = (
            request.messages[-1].content[:30].replace("\n", " ") + "..."
            if request.messages[-1].content
            else "empty"
        )

    logger.info(
        f"[REQUEST {request_id}] START - conv={conv_id}, stream={request.stream}, "
        f"messages={msg_count}, hash={messages_hash}, "
        f"first='{first_msg_preview}', last='{last_msg_preview}', "
        f"time={time.time():.3f}"
    )

    try:
        if request.stream:
            # Return streaming response
            async def generate_stream_inner() -> AsyncGenerator[str, None]:
                """Generate SSE data chunks."""
                logger.debug(f"[REQUEST {request_id}] Starting streaming response")
                try:
                    async for chunk in generate_streaming_response(
                        messages=request.messages,
                        model=request.model,
                        temperature=request.temperature or 0.7,
                        request_id=request_id,
                    ):
                        yield f"data: {chunk}\n\n"

                    yield "data: [DONE]\n\n"
                    logger.info(
                        f"[REQUEST {request_id}] Streaming completed successfully"
                    )

                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    logger.exception(
                        f"[REQUEST {request_id}] ERROR during streaming: {error_msg}"
                    )
                    error_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": f"Error: {str(e)}"},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                with_keepalive(generate_stream_inner(), request_obj, request_id),
                media_type="text/event-stream",
            )

        else:
            # Return non-streaming response
            logger.debug(f"[REQUEST {request_id}] Processing non-streaming response")
            try:
                result = await generate_response(
                    messages=request.messages,
                    model=request.model,
                    temperature=request.temperature or 0.7,
                )

                logger.info(f"[REQUEST {request_id}] Non-streaming response completed")

                # Result is always OpenAI format
                return ChatCompletionResponse(
                    id=result["id"],
                    created=result["created"],
                    model=result["model"],
                    choices=[
                        Choice(
                            message=Message(
                                role=result["choices"][0]["message"].get(
                                    "role", "assistant"
                                ),
                                content=result["choices"][0]["message"].get(
                                    "content", ""
                                ),
                            ),
                            finish_reason=result["choices"][0].get(
                                "finish_reason", "stop"
                            ),
                        )
                    ],
                )

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.exception(
                    f"[REQUEST {request_id}] ERROR in non-streaming: {error_msg}"
                )
                raise HTTPException(status_code=500, detail=str(e)) from e

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.exception(f"[REQUEST {request_id}] FATAL ERROR: {error_msg}")
        raise HTTPException(status_code=500, detail=str(e)) from None
    finally:
        logger.info(f"[REQUEST {request_id}] END")


@router.post("/responses", response_model=None)
async def create_response(
    request: ResponsesRequest,
    request_obj: Request,
):
    """Create response using Open Responses API format.

    Handles both streaming and non-streaming responses.
    Stateless mode only (no previous_response_id support).

    Args:
        request: Responses API request
        request_obj: FastAPI request object

    Returns:
        StreamingResponse for streaming requests, Response object otherwise
    """
    request_id = f"resp_{uuid.uuid4().hex[:24]}"

    # Log request details
    input_type = (
        "string" if isinstance(request.input, str) else f"list({len(request.input)})"
    )
    logger.info(
        f"[RESPONSES {request_id}] START - stream={request.stream}, "
        f"input={input_type}, model={request.model}"
    )

    try:
        if request.stream:
            # Return streaming response
            async def generate_stream_inner() -> AsyncGenerator[str, None]:
                """Generate SSE stream for Responses API."""
                logger.debug(f"[RESPONSES {request_id}] Starting streaming response")
                try:
                    async for event in generate_responses_stream(
                        input_messages=request.input,
                        model=request.model,
                        instructions=request.instructions,
                        tools=request.tools,
                        temperature=request.temperature or 0.7,
                    ):
                        yield event

                    yield "data: [DONE]\n\n"
                    logger.info(
                        f"[RESPONSES {request_id}] Streaming completed successfully"
                    )

                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    logger.exception(
                        f"[RESPONSES {request_id}] ERROR during streaming: {error_msg}"
                    )
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                with_keepalive(generate_stream_inner(), request_obj, request_id),
                media_type="text/event-stream",
            )

        else:
            # Return non-streaming response
            logger.debug(f"[RESPONSES {request_id}] Processing non-streaming response")
            try:
                result = await generate_responses_sync(
                    input_messages=request.input,
                    model=request.model,
                    instructions=request.instructions,
                    tools=request.tools,
                    temperature=request.temperature or 0.7,
                )

                logger.info(
                    f"[RESPONSES {request_id}] Non-streaming response completed"
                )
                return result

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.exception(
                    f"[RESPONSES {request_id}] ERROR in non-streaming: {error_msg}"
                )
                raise HTTPException(status_code=500, detail=str(e)) from e

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.exception(f"[RESPONSES {request_id}] FATAL ERROR: {error_msg}")
        raise HTTPException(status_code=500, detail=str(e)) from None
    finally:
        logger.info(f"[RESPONSES {request_id}] END")
