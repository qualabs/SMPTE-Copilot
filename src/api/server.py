#!/usr/bin/env python3
"""FastAPI server exposing OpenAI-compatible chat completions endpoint"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src import Config
from src.api.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    Usage,
)
from src.components import RAGComponents, execute_query, initialize_rag_components
from src.logger import Logger
from src.pipeline import PipelineStatus


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    try:
        config = Config.get_config()
        Logger.setup(config)
        logger = logging.getLogger()
        logger.info("Initializing RAG components...")
        
        app.state.components = initialize_rag_components(config)
        app.state.logger = logger
        app.state.initialized = True
        
        logger.info("Server startup complete")
    except Exception as e:
        logging.error(f"Failed to initialize components: {e}")
        app.state.initialized = False
    
    yield
    
    logger = getattr(app.state, "logger", logging.getLogger())
    logger.info("Server shutting down")


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
    initialized = getattr(app.state, "initialized", False)
    return {
        "status": "healthy" if initialized else "initializing",
        "initialized": initialized,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """OpenAI-compatible chat completions endpoint.

    This endpoint processes chat messages, extracts the user query,
    runs it through the RAG pipeline, and returns a response in
    OpenAI-compatible format.
    """
    if not getattr(app.state, "initialized", False):
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
        context = execute_query(components, query)

        if context.status == PipelineStatus.FAILED:
            logger.error(f"Pipeline failed: {context.error}")
            raise HTTPException(
                status_code=500,
                detail=f"RAG pipeline failed: {context.error}",
            )

        answer = context.llm_response or "I don't know based on the provided documents."

        # Build OpenAI-compatible response
        response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created_timestamp = int(time.time())

        # Estimate token usage
        prompt_tokens = len(context.prompt.split()) if context.prompt else 0
        completion_tokens = len(answer.split())
        total_tokens = prompt_tokens + completion_tokens

        response = ChatCompletionResponse(
            id=response_id,
            created=created_timestamp,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=answer),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
