"""Main configuration class combining all sub-configurations."""

from pathlib import Path
from typing import Optional
import threading
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

from .constants import CONFIG_FILE_NAME
from .chunking import ChunkingConfig
from .embedding import EmbeddingConfig
from .loader import LoaderConfig
from .vector_store import VectorStoreConfig
from .retrieval import RetrievalConfig
from .paths import PathsConfig
from .logging import LoggingConfig

class Config(BaseSettings):
    """Main configuration class combining all sub-configurations."""
    
    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    @classmethod
    def from_file(cls, config_path: Path) -> "Config":
        """Load configuration from a YAML file."""
        config_path = Path(config_path).expanduser().resolve()
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if config_path.suffix.lower() != ".yaml":
            raise ValueError(f"Configuration file must be YAML (.yaml), got: {config_path.suffix}")
        
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
        
        # Create config from nested dict
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
        
        # Double-checked locking pattern for thread safety
        if _config is None:
            with _config_lock:
                # Check again inside the lock to prevent race conditions
                if _config is None:
                    config_path = Path(CONFIG_FILE_NAME)
                    if config_path.exists():
                        _config = Config.from_file(config_path)
                    else:
                        _config = Config()
        
        return _config

# Global configuration instance (thread-safe singleton pattern)
_config: Optional[Config] = None
_config_lock = threading.Lock()