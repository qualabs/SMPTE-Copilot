"""Main configuration class combining all sub-configurations."""

import threading
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

from .access_control import AccessControlConfig
from .chunking import ChunkingConfig
from .constants import CONFIG_FILE_NAME
from .embedding import EmbeddingConfig
from .llm import LLMConfig
from .loader import LoaderConfig
from .logging import LoggingConfig
from .paths import PathsConfig
from .pipeline import PipelineConfig
from .preprocessing import PreprocessingConfig
from .reranking import RerankingConfig
from .retrieval import RetrievalConfig
from .vector_store import VectorStoreConfig


class Config(BaseSettings):
    """Main configuration class combining all sub-configurations."""

    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    access_control: AccessControlConfig = Field(default_factory=AccessControlConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    @classmethod
    def from_file(cls, config_path: Path) -> "Config":
        """Load configuration from a YAML file."""
        config_path = Path(config_path).expanduser().resolve()

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if config_path.suffix.lower() != ".yaml":
            raise ValueError(f"Configuration file must be YAML (.yaml), got: {config_path.suffix}")

        with config_path.open() as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    @staticmethod
    def get_config() -> "Config":
        """Get the global configuration instance (thread-safe singleton pattern).

        The configuration is loaded once from config.yaml
        and cached for subsequent calls. If the config file doesn't exist,
        uses default values.

        This method is thread-safe and ensures that only one instance
        is created even when called concurrently from multiple threads.

        Returns
        -------
        Config
            The global configuration instance (same instance on subsequent calls).
        """
        global _config, _config_lock

        if _config is None:
            with _config_lock:
                if _config is None:
                    config_path = Path(CONFIG_FILE_NAME)
                    _config = Config.from_file(config_path) if config_path.exists() else Config()

        return _config

# Global configuration instance (thread-safe singleton pattern)
_config: Config | None = None
_config_lock = threading.Lock()
