#!/usr/bin/env python3
"""FastAPI server exposing OpenAI-compatible chat completions endpoint"""

import logging
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src import Config
from src.api.helpers import (
    build_chat_response,
    estimate_token_usage,
    execute_rag_query_with_error_handling,
    resolve_user_role_from_headers,
)
from src.api.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Citation,
    RAGQueryRequest,
    RAGQueryResponse,
)
from src.components import RAGComponents, initialize_rag_components
from src.logger import Logger
from src.user_resolvers import UserRoleResolverFactory


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Initialize state attributes
    app.state.logger = logging.getLogger(__name__)
    app.state.initialized = False

    # Startup
    try:
        config = Config.get_config()
        Logger.setup(config)
        app.state.logger.info("Initializing RAG components...")
        app.state.components = initialize_rag_components()

        # Load access control configuration
        app.state.default_user_role = config.access_control.default_user_role
        app.state.role_mapping = config.access_control.get_role_mapping()

        # Initialize user role resolver for dynamic role resolution
        # Pass default_role from access_control to avoid duplication
        resolver_config = dict(config.user_resolver.resolver_config or {})
        resolver_config["default_role"] = app.state.default_user_role

        app.state.user_resolver = UserRoleResolverFactory.create(
            config.user_resolver.resolver_name,
            **resolver_config,
        )
        app.state.logger.info(
            f"User resolver initialized: type={config.user_resolver.resolver_name.value}, "
            f"default_role={app.state.user_resolver.default_role}"
        )

        if app.state.role_mapping:
            app.state.logger.info(
                f"Access control enabled: role mapping loaded with {len(app.state.role_mapping)} roles"
            )

        app.state.initialized = True
        app.state.logger.info("Server startup complete")
    except Exception:
        app.state.logger.exception("Failed to initialize components")
        app.state.initialized = False

    yield

    app.state.logger.info("Server shutting down")


app = FastAPI(
    title="SMPTE-Copilot RAG API",
    description="OpenAI-compatible API for SMPTE document question answering",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if app.state.initialized else "initializing",
        "initialized": app.state.initialized,
    }


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:

    return {
        "object": "list",
        "data": [
            {
                "id": "smpte-copilot",
                "object": "model",
                "owned_by": "smpte",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    x_openwebui_user_email: Annotated[str | None, Header(alias="X-OpenWebUI-User-Email")] = None,
    x_openwebui_user_id: Annotated[str | None, Header(alias="X-OpenWebUI-User-Id")] = None,
    x_openwebui_user_name: Annotated[str | None, Header(alias="X-OpenWebUI-User-Name")] = None,
    x_openwebui_user_role: Annotated[str | None, Header(alias="X-OpenWebUI-User-Role")] = None,
) -> ChatCompletionResponse:
    """OpenAI-compatible chat completions endpoint.

    This endpoint processes chat messages, extracts the user query,
    runs it through the RAG pipeline, and returns a response in
    OpenAI-compatible format.

    When used with OpenWebUI (with ENABLE_FORWARD_USER_INFO_HEADERS=true),
    the user's email is passed in headers and used to resolve their role
    for access-controlled document retrieval.
    """
    components: RAGComponents = app.state.components
    logger = app.state.logger

    logger.info(f"x_openwebui_user_email: {x_openwebui_user_email}")
    logger.info(f"x_openwebui_user_id: {x_openwebui_user_id}")
    logger.info(f"x_openwebui_user_name: {x_openwebui_user_name}")
    logger.info(f"x_openwebui_user_role: {x_openwebui_user_role}")

    # Resolve user role from headers (OpenWebUI integration)
    user_role = resolve_user_role_from_headers(
        user_email=x_openwebui_user_email,
        user_id=x_openwebui_user_id,
        user_resolver=app.state.user_resolver,
        default_role=app.state.default_user_role,
        logger=logger,
    )

    # Extract the last user message as the query
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(
            status_code=400,
            detail="No user message found in request",
        )

    query = user_messages[-1].content
    logger.info(f"Processing query: {query}")

    context = execute_rag_query_with_error_handling(
        components=components,
        query=query,
        user_role=user_role,
        role_mapping=app.state.role_mapping,
        logger=logger,
        is_initialized=app.state.initialized,
    )

    answer = context.llm_response or "I don't know based on the provided documents."
    usage = estimate_token_usage(context.prompt, answer)
    response = build_chat_response(
        answer=answer,
        model=request.model,
        usage=usage,
    )
    logger.info("Query processed successfully")

    return response


@app.post("/v1/rag/query")
async def rag_query(
    request: RAGQueryRequest,
    x_openwebui_user_email: Annotated[str | None, Header(alias="X-OpenWebUI-User-Email")] = None,
    x_openwebui_user_id: Annotated[str | None, Header(alias="X-OpenWebUI-User-Id")] = None,
    x_openwebui_user_name: Annotated[str | None, Header(alias="X-OpenWebUI-User-Name")] = None,
    x_openwebui_user_role: Annotated[str | None, Header(alias="X-OpenWebUI-User-Role")] = None,
) -> RAGQueryResponse:
    """RAG query endpoint that returns response with citations.

    This endpoint is designed to work with OpenWebUI Pipes that need
    access to citation data for emitting citation events.
    """
    components: RAGComponents = app.state.components
    logger = app.state.logger

    logger.info(f"x_openwebui_user_email: {x_openwebui_user_email}")
    logger.info(f"x_openwebui_user_id: {x_openwebui_user_id}")
    logger.info(f"x_openwebui_user_name: {x_openwebui_user_name}")
    logger.info(f"x_openwebui_user_role: {x_openwebui_user_role}")

    # Resolve user role from headers (OpenWebUI integration)
    user_role = resolve_user_role_from_headers(
        user_email=x_openwebui_user_email,
        user_id=x_openwebui_user_id,
        user_resolver=app.state.user_resolver,
        default_role=app.state.default_user_role,
        logger=logger,
    )

    logger.info(f"Processing RAG query: {request.query}")

    context = execute_rag_query_with_error_handling(
        components=components,
        query=request.query,
        user_role=user_role,
        role_mapping=app.state.role_mapping,
        logger=logger,
        is_initialized=app.state.initialized,
    )

    response_text = context.llm_response or "I don't know based on the provided documents."

    citations = [
        Citation(
            id=c["id"],
            source=c.get("source"),
            page=c.get("page"),
            score=c.get("score", 0.0),
            content=c.get("content", ""),
        )
        for c in (context.citations or [])
    ]

    logger.info(f"RAG query processed successfully with {len(citations)} citations")

    return RAGQueryResponse(response=response_text, citations=citations)
