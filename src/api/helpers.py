"""Helper functions for API response construction."""

import time
import uuid

from .models import ChatCompletionChoice, ChatCompletionResponse, Message, Usage


def estimate_token_usage(prompt: str | None, answer: str) -> Usage:
    """Estimate token usage for the query and response.
    
    Parameters
    ----------
    prompt
        The prompt sent to the LLM
    answer
        The generated answer
        
    Returns
    -------
    Usage object with estimated token counts
    """
    prompt_tokens = len(prompt.split()) if prompt else 0
    completion_tokens = len(answer.split())
    total_tokens = prompt_tokens + completion_tokens
    
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def build_chat_response(
    answer: str,
    model: str,
    usage: Usage,
) -> ChatCompletionResponse:
    """Build an OpenAI-compatible chat completion response.
    
    Parameters
    ----------
    answer
        The generated answer text
    model
        The model identifier to include in the response
    usage
        Token usage statistics
        
    Returns
    -------
    ChatCompletionResponse object
    """
    response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created_timestamp = int(time.time())
    
    choice = ChatCompletionChoice(
        index=0,
        message=Message(role="assistant", content=answer),
        finish_reason="stop",
    )
    
    return ChatCompletionResponse(
        id=response_id,
        created=created_timestamp,
        model=model,
        choices=[choice],
        usage=usage,
    )
