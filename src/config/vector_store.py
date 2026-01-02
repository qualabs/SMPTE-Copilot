"""Vector store configuration."""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

from src.vector_stores.types import VectorStoreType


class VectorStoreConfig(BaseSettings):
    """Vector store configuration."""

    store_name: VectorStoreType = Field(
        default=VectorStoreType.CHROMADB,
        description="Vector store type",
    )
    store_config: Optional[dict] = Field(
        default=None,
        description=(
            "Store-specific configuration dictionary. "
            "Required keys: persist_directory (str), collection_name (str). "
            "For Qdrant: also include 'url' (str, default: 'http://localhost:6333'). "
            "For ChromaDB: no additional config needed beyond persist_directory and collection_name."
        ),
    )

