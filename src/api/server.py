#!/usr/bin/env python3
"""FastAPI server exposing OpenAI-compatible chat completions endpoint."""

import logging
import time
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import Config
from src.components import RAGComponents, initialize_rag_components
from src.logger import Logger
from src.pipeline import PipelineExecutor, PipelineStatus, QueryContext
from src.pipeline.steps import GenerationStep, QueryEmbeddingStep, RetrieveStep

# Initialize FastAPI app
app = FastAPI(
    title="SMPTE-Copilot RAG API",
    description="OpenAI-compatible API for SMPTE document question answering",
    version="1.0.0",
)

# Add CORS middleware to allow requests from web UIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for OpenAI-compatible API
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


@app.on_event("startup")
async def startup_event():
    """Initialize components when the server starts."""
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
        # Allow server to start but requests will fail with proper error


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
    # Check if components are initialized
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
        # Create query context
        context = QueryContext(user_query=query)

        # Build pipeline steps
        steps = [
            QueryEmbeddingStep(components.embedding_model),
            RetrieveStep(components.retriever),
            GenerationStep(components.llm, components.config.llm.llm_name),
        ]

        # Execute pipeline
        executor = PipelineExecutor(steps)
        context = executor.execute(context)

        # Check for pipeline failure
        if context.status == PipelineStatus.FAILED:
            logger.error(f"Pipeline failed: {context.error}")
            raise HTTPException(
                status_code=500,
                detail=f"RAG pipeline failed: {context.error}",
            )

        # Get the response
        answer = context.llm_response or "I don't know based on the provided documents."

        # Build OpenAI-compatible response
        response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created_timestamp = int(time.time())

        # Estimate token usage (rough approximation)
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
