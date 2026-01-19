"""Helper functions for API response construction."""

import logging
import time
import uuid

from fastapi import HTTPException

from src.components import RAGComponents, execute_query
from src.pipeline import PipelineStatus, QueryContext
from src.user_resolvers import UserRoleResolver

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


def resolve_user_role_from_headers(
    user_email: str | None,
    user_id: str | None,
    user_resolver: UserRoleResolver,
    default_role: str,
    logger: logging.Logger,
) -> str:
    """Resolve user role from OpenWebUI headers.

    Parameters
    ----------
    user_email
        User's email from X-OpenWebUI-User-Email header
    user_id
        User's ID from X-OpenWebUI-User-Id header
    user_resolver
        User role resolver instance
    default_role
        Default role to use when no headers are present
    logger
        Logger instance for logging

    Returns
    -------
    str
        Resolved user role
    """
    if user_email or user_id:
        user_role = user_resolver.resolve_role(
            user_email=user_email,
            user_id=user_id,
        )
        logger.info(f"Resolved role for user '{user_email}': {user_role}")
        return user_role
    else:
        logger.info(f"No user headers, using default role: {default_role}")
        return default_role


def execute_rag_query_with_error_handling(
    components: RAGComponents,
    query: str,
    user_role: str,
    role_mapping: dict[str, list[str]] | None,
    logger: logging.Logger,
    is_initialized: bool,
) -> QueryContext:
    """Execute RAG query with common error handling.

    This function encapsulates the common logic for executing a RAG query
    and handling errors consistently across endpoints.

    Parameters
    ----------
    components
        Initialized RAG components
    query
        User's query text
    user_role
        User's role for access control
    role_mapping
        Role-to-tags mapping for access control
    logger
        Logger instance for logging
    is_initialized
        Whether the server is initialized

    Returns
    -------
    QueryContext
        Pipeline context containing query results

    Raises
    ------
    HTTPException
        If service is not initialized, pipeline fails, or other errors occur
    """
    if not is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Please ensure vector database is available.",
        )

    try:
        context = execute_query(
            components,
            query,
            user_role=user_role,
            role_mapping=role_mapping,
        )

        if context.status == PipelineStatus.FAILED:
            logger.error(f"Pipeline failed: {context.error}")
            raise HTTPException(
                status_code=500,
                detail=f"RAG pipeline failed: {context.error}",
            )

        return context

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {e!s}",
        ) from e
