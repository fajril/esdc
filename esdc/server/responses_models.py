# Standard library
from typing import Any, Literal

# Third-party
from pydantic import BaseModel, Field


class InputTextContent(BaseModel):
    """Input text content part."""

    type: Literal["input_text"] = "input_text"
    text: str


class OutputTextContent(BaseModel):
    """Output text content part."""

    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[dict[str, Any]] = Field(default_factory=list)


class ResponseInputMessage(BaseModel):
    """Input message item."""

    type: Literal["message"] = "message"
    role: Literal["user", "assistant", "system"]
    content: str | list[InputTextContent]


class ResponseFunctionCallOutput(BaseModel):
    """Function call output item (tool result submitted by client)."""

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str
    status: Literal["completed", "in_progress", "failed"] = "completed"


class ResponseInputItem(BaseModel):
    """Input item for Responses API.

    Can be:
    - message: user/assistant/system message
    - function_call_output: tool result from client
    """

    # Discriminator field
    type: Literal["message", "function_call_output"]

    # Message fields
    role: Literal["user", "assistant", "system"] | None = None
    content: str | list[InputTextContent] | None = None

    # Function call output fields
    call_id: str | None = None
    output: str | None = None


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
    output: str


class ResponseOutputItem(BaseModel):
    """Output item - discriminated union.

    Can be:
    - message: assistant message with content
    - function_call: model wants to call a tool
    - function_call_output: server-side tool execution result
    """

    # Using model as a container for discriminated union
    # The actual type is determined by the 'type' field
    model_config = {"extra": "allow"}

    id: str
    type: Literal["message", "function_call", "function_call_output"]
    status: Literal["in_progress", "completed", "failed", "incomplete"]

    # Message fields
    role: Literal["assistant"] | None = None
    content: list[OutputTextContent] | None = None

    # Function call fields
    name: str | None = None
    call_id: str | None = None
    arguments: str | None = None

    # Function call output fields
    output: str | None = None


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

    # Stateless mode (ESDC doesn't support previous_response_id)
    # previous_response_id is intentionally omitted


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
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        texts.append(content.get("text", ""))
        return "".join(texts)
