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

# Local
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
                    logger.exception(
                        f"[REQUEST {request_id}] ERROR during streaming: {type(e).__name__}: {str(e)}"
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
                logger.exception(
                    f"[REQUEST {request_id}] ERROR in non-streaming: {type(e).__name__}: {str(e)}"
                )
                raise HTTPException(status_code=500, detail=str(e)) from e

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            f"[REQUEST {request_id}] FATAL ERROR: {type(e).__name__}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        logger.info(f"[REQUEST {request_id}] END")
