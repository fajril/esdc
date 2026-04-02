# Standard library
import time
import uuid
from typing import AsyncGenerator

# Third-party
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

# Local
from esdc.server.tool_formatter import detect_native_format

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

router = APIRouter()


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

    # Detect format preference from headers
    headers = dict(request_obj.headers)
    use_native = detect_native_format(headers, request.stream)

    if request.stream:
        # Return streaming response
        async def generate_stream() -> AsyncGenerator[str, None]:
            """Generate SSE stream."""
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

            except Exception as e:
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
        try:
            result = await generate_response(
                messages=request.messages,
                model=request.model,
                temperature=request.temperature or 0.7,
                use_native_format=use_native,
            )

            return ChatCompletionResponse(
                id=request_id,
                created=created,
                model=request.model,
                choices=[
                    Choice(
                        message=Message(
                            role=result["role"],
                            content=result["content"],
                        ),
                        finish_reason=result["finish_reason"],
                    )
                ],
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
