#!/usr/bin/env python3
"""FastAPI server exposing OpenAI-compatible chat completions endpoint"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src import Config
from src.api.helpers import build_chat_response, estimate_token_usage
from src.api.models import ChatCompletionRequest, ChatCompletionResponse
from src.components import RAGComponents, execute_query, initialize_rag_components
from src.logger import Logger
from src.pipeline import PipelineStatus


def load_role_mapping(mapping_file: str) -> dict[str, list[str]]:
    """Load role-to-tags mapping from JSON file.

    Parameters
    ----------
    mapping_file : str
        Path to the JSON file containing role-to-tags mapping.

    Returns
    -------
    dict[str, list[str]]
        Role-to-tags mapping, or empty dict if file doesn't exist.
    """
    logger = logging.getLogger(__name__)
    try:
        mapping_path = Path(mapping_file)
        if mapping_path.exists():
            with open(mapping_path, "r") as f:
                return json.load(f)
        else:
            logger.warning(f"Role mapping file not found: {mapping_file}")
    except Exception as e:
        logger.warning(f"Could not load role mapping: {e}")
    return {}


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
        app.state.components = initialize_rag_components(config)
        
        # Load access control configuration
        app.state.user_role = config.access_control.default_user_role
        app.state.role_mapping = None
        
        if config.access_control.role_mapping_file:
            app.state.role_mapping = load_role_mapping(str(config.access_control.role_mapping_file))
            if app.state.user_role:
                app.state.logger.info(
                    f"Access control enabled: role='{app.state.user_role}', "
                    f"mapping loaded with {len(app.state.role_mapping)} roles"
                )
        
        app.state.initialized = True
        app.state.logger.info("Server startup complete")
    except Exception as e:
        app.state.logger.error(f"Failed to initialize components: {e}")
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
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """OpenAI-compatible chat completions endpoint.

    This endpoint processes chat messages, extracts the user query,
    runs it through the RAG pipeline, and returns a response in
    OpenAI-compatible format.
    """
    if not app.state.initialized:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Please ensure vector database is available.",
        )

    components: RAGComponents = app.state.components
    logger = app.state.logger

    # Extract the last user message as the query
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(
            status_code=400,
            detail="No user message found in request",
        )

    query = user_messages[-1].content
    logger.info(f"Processing query: {query}")

    try:
        context = execute_query(
            components,
            query,
            user_role=app.state.user_role,
            role_mapping=app.state.role_mapping,
        )

        if context.status == PipelineStatus.FAILED:
            logger.error(f"Pipeline failed: {context.error}")
            raise HTTPException(
                status_code=500,
                detail=f"RAG pipeline failed: {context.error}",
            )

        answer = context.llm_response or "I don't know based on the provided documents."
        usage = estimate_token_usage(context.prompt, answer)
        response=build_chat_response(
            answer=answer,
            model=request.model,
            usage=usage,
        )
        logger.info("Query processed successfully")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}",
        )
