"""Pipeline configuration for ingestion and query steps."""


from pydantic import BaseModel, Field


class IngestionPipelineConfig(BaseModel):
    """Configuration for ingestion pipeline steps.

    Each step can be enabled or disabled independently.
    """

    load_enabled: bool = Field(
        default=True,
        description="Enable/disable the load step (converts files to markdown)",
    )
    preprocess_enabled: bool = Field(
        default=True,
        description="Enable/disable the preprocessing step (removes duplicates, etc.)",
    )
    chunk_enabled: bool = Field(
        default=True,
        description="Enable/disable the chunking step",
    )
    save_enabled: bool = Field(
        default=True,
        description="Enable/disable the save step (includes embedding generation and vector database storage)",
    )
    parallel_enabled: bool = Field(
        default=False,
        description="Enable/disable parallel processing of multiple files using threading",
    )
    max_workers: int | None = Field(
        default=None,
        description="Maximum number of parallel workers (None = CPU count, 1 = sequential)",
    )


class QueryPipelineConfig(BaseModel):
    """Configuration for query pipeline steps.

    Each step can be enabled or disabled independently.
    """

    retrieve_enabled: bool = Field(
        default=True,
        description="Enable/disable the retrieval step (includes query embedding)",
    )
    generation_enabled: bool = Field(
        default=True,
        description="Enable/disable the LLM generation step",
    )


class PipelineConfig(BaseModel):
    """Configuration for both ingestion and query pipelines."""

    ingestion: IngestionPipelineConfig | None = Field(
        default_factory=IngestionPipelineConfig,
        description="Ingestion pipeline configuration",
    )
    query: QueryPipelineConfig | None = Field(
        default_factory=QueryPipelineConfig,
        description="Query pipeline configuration",
    )

