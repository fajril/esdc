# Standard library
import logging
import time
import uuid
from collections.abc import AsyncGenerator

# Third-party
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

# Local
from esdc.server.agent_wrapper import generate_response, generate_streaming_response
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
from esdc.server.tool_formatter import should_use_native_format

router = APIRouter()
logger = logging.getLogger("esdc.server.routes")


@router.get("/models")
async def list_models() -> ModelList:
    """List available models.

    Returns:
        List of available models in OpenAI-compatible format.
    """
    return ModelList(
        data=[
            ModelInfo(
                id="esdc-agent",
                created=int(time.time()),
                owned_by="esdc",
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

    # Log request details
    conv_id = getattr(request, "conversation_id", "none")
    msg_count = len(request.messages) if request.messages else 0
    logger.info(
        f"[REQUEST {request_id}] START - conv={conv_id}, "
        f"stream={request.stream}, messages={msg_count}"
    )

    # Detect format preference from headers
    headers = dict(request_obj.headers)
    use_native = should_use_native_format(headers, request.stream)
    logger.debug(f"[REQUEST {request_id}] use_native_format={use_native}")

    try:
        if request.stream:
            # Return streaming response
            async def generate_stream() -> AsyncGenerator[str, None]:
                """Generate SSE stream."""
                logger.debug(f"[REQUEST {request_id}] Starting streaming response")
                try:
                    async for chunk in generate_streaming_response(
                        messages=request.messages,
                        model=request.model,
                        temperature=request.temperature or 0.7,
                        use_native_format=use_native,
                    ):
                        # Format as SSE data
                        yield f"data: {chunk}\n\n"

                    # Send final [DONE] marker
                    yield "data: [DONE]\n\n"
                    logger.info(
                        f"[REQUEST {request_id}] Streaming completed successfully"
                    )

                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    logger.exception(
                        f"[REQUEST {request_id}] ERROR during streaming: {error_msg}"
                    )
                    # Send error as final chunk
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
                    yield f"data: {error_chunk}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate_stream(),
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
                    use_native_format=use_native,
                )

                logger.info(f"[REQUEST {request_id}] Non-streaming response completed")

                # Handle both native OpenAI format and legacy format
                if "choices" in result:
                    # Native OpenAI format - result is already ChatCompletion-like
                    return ChatCompletionResponse(
                        id=result.get("id", request_id),
                        created=result.get("created", created),
                        model=result.get("model", request.model),
                        choices=[
                            Choice(
                                message=Message(
                                    role=result["choices"][0]["message"].get(
                                        "role", "assistant"
                                    ),
                                    content=result["choices"][0]["message"].get(
                                        "content", ""
                                    ),
                                    tool_calls=result["choices"][0]["message"].get(
                                        "tool_calls"
                                    ),
                                ),
                                finish_reason=result["choices"][0].get(
                                    "finish_reason", "stop"
                                ),
                            )
                        ],
                    )
                else:
                    # Legacy format - simple dict with role/content/finish_reason
                    return ChatCompletionResponse(
                        id=request_id,
                        created=created,
                        model=request.model,
                        choices=[
                            Choice(
                                message=Message(
                                    role=result.get("role", "assistant"),
                                    content=result.get("content", ""),
                                ),
                                finish_reason=result.get("finish_reason", "stop"),
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
            async def generate_stream() -> AsyncGenerator[str, None]:
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

                    # Send final [DONE] marker
                    yield "data: [DONE]\n\n"
                    logger.info(
                        f"[RESPONSES {request_id}] Streaming completed successfully"
                    )

                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    logger.exception(
                        f"[RESPONSES {request_id}] ERROR during streaming: {error_msg}"
                    )
                    # Error events are already formatted in the stream
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate_stream(),
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
