# Standard library
from typing import Literal

# Third-party
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message model."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    output: list[dict] | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request model."""

    model: str = Field(default="iris", description="Model ID to use")
    messages: list[Message] = Field(..., description="Conversation messages")
    stream: bool = Field(default=False, description="Whether to stream the response")
    temperature: float | None = Field(default=0.7, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=1, ge=0, le=1)
    frequency_penalty: float | None = Field(default=0, ge=-2, le=2)
    presence_penalty: float | None = Field(default=0, ge=-2, le=2)
    user: str | None = Field(
        default=None, description="Unique identifier for the end-user"
    )


class Choice(BaseModel):
    """Chat completion choice."""

    index: int = Field(default=0)
    message: Message
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", None] = (
        None
    )


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: dict | None = None


class ChoiceDelta(BaseModel):
    """Chat completion streaming delta."""

    index: int = Field(default=0)
    delta: Message
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", None] = (
        None
    )


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible chat completion streaming chunk."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChoiceDelta]


class ModelInfo(BaseModel):
    """Model information."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "IRIS"


class ModelList(BaseModel):
    """List of available models."""

    object: Literal["list"] = "list"
    data: list[ModelInfo]


class ErrorResponse(BaseModel):
    """Error response model."""

    error: dict[str, str | int | None]
