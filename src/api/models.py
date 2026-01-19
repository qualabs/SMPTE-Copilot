"""Pydantic models for OpenAI-compatible API."""


from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message."""

    role: str = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Content of the message")


class ChatCompletionRequest(BaseModel):
    """Request body for chat completions."""

    model: str = Field(default="smpte-copilot", description="Model identifier")
    messages: list[Message] = Field(..., description="List of messages in the conversation")
    temperature: float | None = Field(default=0.7, ge=0, le=2, description="Sampling temperature")
    max_tokens: int | None = Field(default=None, description="Maximum tokens to generate")
    stream: bool | None = Field(default=False, description="Whether to stream responses")
    top_p: float | None = Field(default=1.0, ge=0, le=1, description="Nucleus sampling parameter")


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""

    index: int
    message: Message
    finish_reason: str = "stop"


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Response body for chat completions."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class RAGQueryRequest(BaseModel):
    """Request body for RAG query endpoint."""

    query: str = Field(..., description="User's question or query text")


class Citation(BaseModel):
    """A single citation from retrieved documents."""

    id: int = Field(..., description="Citation index (1-based)")
    source: str | None = Field(default=None, description="Source document name or path")
    page: int | None = Field(default=None, description="Page number in the source document")
    score: float = Field(..., description="Relevance score")
    content: str = Field(..., description="Content of the retrieved chunk")


class RAGQueryResponse(BaseModel):
    """Response body for RAG query endpoint with citations."""

    response: str = Field(..., description="LLM-generated response with citation markers")
    citations: list[Citation] = Field(default_factory=list, description="List of citations")
