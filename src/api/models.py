"""Pydantic models for OpenAI-compatible API."""

from typing import List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message."""

    role: str = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Content of the message")


class ChatCompletionRequest(BaseModel):
    """Request body for chat completions."""

    model: str = Field(default="smpte-copilot", description="Model identifier")
    messages: List[Message] = Field(..., description="List of messages in the conversation")
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(default=False, description="Whether to stream responses")
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1, description="Nucleus sampling parameter")


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
    choices: List[ChatCompletionChoice]
    usage: Usage
