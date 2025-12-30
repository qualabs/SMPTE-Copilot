"""Vector store configuration."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

from src.vector_stores.constants import DEFAULT_COLLECTION_NAME
from src.vector_stores.types import VectorStoreType


class VectorStoreConfig(BaseSettings):
    """Vector store configuration."""

    store_name: VectorStoreType = Field(
        default=VectorStoreType.CHROMADB,
        description="Vector store type",
    )
    persist_directory: Path = Field(
        default=Path("./vector_db"),
        description=(
            "Directory to persist vector store data "
            "(relative to current working directory)"
        ),
    )
    collection_name: str = Field(
        default=DEFAULT_COLLECTION_NAME,
        description="Collection name in the vector store",
    )
    url: Optional[str] = Field(
        default="http://localhost:6333",
        description=(
            "URL for Qdrant server "
            "(only used when store_name is 'qdrant')"
        ),
    )
    store_config: Optional[dict] = Field(
        default=None,
        description="Additional store-specific configuration",
    )

