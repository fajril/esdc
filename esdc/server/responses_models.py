"""Responses API models with discriminated unions and flexible input handling."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Field, field_validator


def normalize_content_type(v: str) -> str:
    """Normalize various content type names to standard format.

    OpenWebUI may send 'text' instead of 'input_text'.
    This ensures consistent type names.
    """
    if v == "text":
        return "input_text"
    return v


class FlexibleContentPart(BaseModel):
    """Content part that accepts various formats and normalizes them.

    Handles different type names ('text', 'input_text', 'output_text')
    and allows extra fields for forward compatibility.
    """

    type: str = "input_text"
    text: str = ""

    model_config = {"extra": "allow"}


def _normalize_content(v: Any) -> str | list[dict[str, Any]] | None:
    """Normalize content to accept mixed string/dict arrays.

    Converts:
    - Plain strings -> keep as is
    - List of mixed items -> list of dicts
    - None -> None
    """
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        result = []
        for item in v:
            if isinstance(item, str):
                result.append({"type": "input_text", "text": item})
            elif isinstance(item, dict):
                result.append(item)
            else:
                result.append({"type": "input_text", "text": str(item)})
        return result
    return v


def _get_input_item_type(v: Any) -> str:
    """Get the type from a dict or model for discriminated union.

    Args:
        v: Input item (dict or BaseModel)

    Returns:
        Type string for discrimination
    """
    if isinstance(v, dict):
        return v.get("type", "message")
    return getattr(v, "type", "message")


class ResponseInputMessageItem(BaseModel):
    """Input message item from user/assistant/system.

    This is one variant of the discriminated union for ResponseInputItem.
    """

    type: Literal["message"] = "message"
    role: Literal["user", "assistant", "system"]
    content: str | list[FlexibleContentPart] | list[dict[str, Any]]

    model_config = {"extra": "allow"}


class ResponseInputFunctionCallItem(BaseModel):
    """Input function call item from previous assistant response.

    This is one variant of the discriminated union for ResponseInputItem.
    Used when OpenWebUI sends conversation history containing tool calls.
    """

    type: Literal["function_call"] = "function_call"
    id: str = Field(default="", description="Function call ID")
    name: str = Field(default="", description="Function name")
    arguments: str = Field(default="", description="Function arguments (JSON)")
    call_id: str = Field(default="", description="Call ID for tracking")
    status: Literal["in_progress", "completed", "failed"] = "completed"

    model_config = {"extra": "allow"}


class ResponseInputFunctionCallOutputItem(BaseModel):
    """Input function call output item (tool result from client).

    This is one variant of the discriminated union for ResponseInputItem.
    """

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str = Field(description="ID of the function call being answered")
    output: str | list[FlexibleContentPart] = Field(
        description="Tool result (string or content parts)"
    )
    status: Literal["completed", "in_progress", "failed"] = "completed"

    model_config = {"extra": "allow"}


# Discriminated union type for input items
ResponseInputItemUnion = Annotated[
    ResponseInputMessageItem | ResponseInputFunctionCallItem | ResponseInputFunctionCallOutputItem,
    Discriminator(_get_input_item_type),
]


class ResponseInputItem(BaseModel):
    """Backward-compatible wrapper for ResponseInputItemUnion.

    This class allows existing code that uses ResponseInputItem(type="message", ...)
    to continue working, while internally using the discriminated union.

    Can be:
    - message: user/assistant/system message
    - function_call: previous assistant tool call
    - function_call_output: tool result from client
    """

    type: Literal["message", "function_call", "function_call_output"]
    role: Literal["user", "assistant", "system"] | None = None
    content: Any = None
    call_id: str | None = None
    output: Any = None
    id: str | None = None
    name: str | None = None
    arguments: str | None = None
    status: str = "completed"

    model_config = {"extra": "allow"}

    @field_validator("content", mode="before")
    @classmethod
    def _normalize_content_field(cls, v: Any) -> Any:
        """Normalize content to accept mixed string/dict arrays."""
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            result = []
            for item in v:
                if isinstance(item, str):
                    result.append({"type": "input_text", "text": item})
                elif isinstance(item, dict):
                    result.append(item)
                else:
                    result.append({"type": "input_text", "text": str(item)})
            return result
        return v

    @field_validator("output", mode="before")
    @classmethod
    def _normalize_output_field(cls, v: Any) -> Any:
        """Normalize output to accept mixed string/dict arrays."""
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            result = []
            for item in v:
                if isinstance(item, str):
                    result.append({"type": "input_text", "text": item})
                elif isinstance(item, dict):
                    result.append(item)
                else:
                    result.append({"type": "input_text", "text": str(item)})
            return result
        return v


class OutputTextContent(BaseModel):
    """Output text content part."""

    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[dict[str, Any]] = Field(default_factory=list)


class InputTextContent(BaseModel):
    """Input text content part (legacy, kept for backward compatibility)."""

    type: Literal["input_text"] = "input_text"
    text: str


class ResponseOutputMessage(BaseModel):
    """Output message item."""

    id: str
    type: Literal["message"] = "message"
    status: Literal["in_progress", "completed", "failed", "incomplete"]
    role: Literal["assistant"] = "assistant"
    content: list[OutputTextContent]


class ResponseFunctionCall(BaseModel):
    """Output function call item."""

    id: str
    type: Literal["function_call"] = "function_call"
    status: Literal["in_progress", "completed", "failed", "incomplete"]
    name: str
    call_id: str
    arguments: str


class ResponseFunctionCallResult(BaseModel):
    """Output function call result item (server-side tool execution result)."""

    id: str
    type: Literal["function_call_output"] = "function_call_output"
    status: Literal["completed", "failed"]
    call_id: str
    output: str | list[dict[str, Any]]


class ResponseOutputItem(BaseModel):
    """Output item - discriminated union.

    Can be:
    - message: assistant message with content
    - function_call: model wants to call a tool
    - function_call_output: server-side tool execution result
    """

    model_config = {"extra": "allow"}

    id: str
    type: Literal["message", "function_call", "function_call_output"]
    status: Literal["in_progress", "completed", "failed", "incomplete"]

    role: Literal["assistant"] | None = None
    content: list[OutputTextContent] | None = None
    name: str | None = None
    call_id: str | None = None
    arguments: str | None = None
    output: str | list[dict[str, Any]] | None = None


class ResponsesRequest(BaseModel):
    """POST /v1/responses request.

    Based on Open Responses specification:
    https://www.openresponses.org/specification
    """

    model: str = Field(default="esdc-agent", description="Model ID to use")
    input: str | list[ResponseInputItem] = Field(
        ..., description="Input to the model (string or list of items)"
    )
    instructions: str | None = Field(
        default=None, description="System message / instructions"
    )
    tools: list[dict[str, Any]] | None = Field(
        default=None, description="Tools available to the model"
    )
    tool_choice: str | dict[str, Any] | None = Field(
        default="auto", description="How the model should use tools"
    )
    stream: bool = Field(default=True, description="Whether to stream the response")
    temperature: float | None = Field(default=0.7, ge=0, le=2)
    max_output_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=1, ge=0, le=1)


class Response(BaseModel):
    """Non-streaming response object."""

    id: str
    object: Literal["response"] = "response"
    created_at: float
    model: str
    status: Literal["completed", "failed", "in_progress", "incomplete"]
    output: list[dict[str, Any]]
    usage: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    @property
    def output_text(self) -> str:
        """Convenience property to get all output text.

        Aggregates all output_text content from message items.
        """
        texts: list[str] = []
        for item in self.output:
            if not isinstance(item, dict):
                continue

            if item.get("type") == "message":
                content = item.get("content", [])
                if not isinstance(content, list):
                    continue

                for part in content:
                    if not isinstance(part, dict):
                        continue

                    if part.get("type") == "output_text":
                        text = part.get("text", "")
                        if isinstance(text, str):
                            texts.append(text)

        return "".join(texts)
