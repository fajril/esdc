# Standard library
import time
import uuid
from typing import AsyncGenerator

# Third-party
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

# Local
from esdc.server.agent_wrapper import generate_response, generate_streaming_response
from esdc.server.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceDelta,
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
):
    """Create chat completion.

    Handles both streaming and non-streaming chat completion requests
    in OpenAI-compatible format.

    Args:
        request: Chat completion request

    Returns:
        StreamingResponse for streaming requests, ChatCompletionResponse otherwise
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if request.stream:
        # Return streaming response
        async def generate_stream() -> AsyncGenerator[str, None]:
            """Generate SSE stream."""
            try:
                async for chunk in generate_streaming_response(
                    messages=request.messages,
                    model=request.model,
                    temperature=request.temperature or 0.7,
                ):
                    # Check if this is the final message
                    finish_reason = chunk.get("finish_reason")

                    if finish_reason:
                        # Final chunk
                        response_chunk = ChatCompletionChunk(
                            id=request_id,
                            created=created,
                            model=request.model,
                            choices=[
                                ChoiceDelta(
                                    delta=Message(role="assistant", content=""),
                                    finish_reason=finish_reason,
                                )
                            ],
                        )
                        yield f"data: {response_chunk.model_dump_json()}\n\n"
                        yield "data: [DONE]\n\n"
                    else:
                        # Regular chunk
                        content = chunk.get("content", "")
                        tool_calls = chunk.get("tool_calls")

                        delta = Message(role="assistant", content=content)
                        if tool_calls:
                            delta.tool_calls = tool_calls

                        response_chunk = ChatCompletionChunk(
                            id=request_id,
                            created=created,
                            model=request.model,
                            choices=[
                                ChoiceDelta(
                                    delta=delta,
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield f"data: {response_chunk.model_dump_json()}\n\n"

            except Exception as e:
                # Send error as final chunk
                response_chunk = ChatCompletionChunk(
                    id=request_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChoiceDelta(
                            delta=Message(role="assistant", content=f"Error: {str(e)}"),
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {response_chunk.model_dump_json()}\n\n"
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
