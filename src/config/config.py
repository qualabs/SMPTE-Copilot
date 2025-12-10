"""Main configuration class combining all sub-configurations."""

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings

from .chunking import ChunkingConfig
from .embedding import EmbeddingConfig
from .loader import LoaderConfig
from .vector_store import VectorStoreConfig
from .retrieval import RetrievalConfig
from .paths import PathsConfig

class Config(BaseSettings):
    """Main configuration class combining all sub-configurations."""
    
    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    
    @classmethod
    def from_file(cls, config_path: Path) -> "Config":
        """Load configuration from a YAML file."""
        config_path = Path(config_path).expanduser().resolve()
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        suffix = config_path.suffix.lower()
        
        if suffix not in (".yaml", ".yml"):
            raise ValueError(f"Configuration file must be YAML (.yaml or .yml), got: {suffix}")
        
        import yaml
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
        
        # Create config from nested dict
        return cls(**data)

# Global configuration instance (singleton pattern)
_config: Optional[Config] = None

def get_config() -> Config:
    """Get the global configuration instance (singleton pattern).
    
    The configuration is loaded once from config.yaml (or config.yml) 
    and cached for subsequent calls. If the config file doesn't exist,
    uses default values.
    
    Returns
    -------
    Config
        The global configuration instance (same instance on subsequent calls).
    """
    global _config
    
    if _config is None:
        file_path = None
        for path_str in ["config.yaml", "config.yml"]:
            path = Path(path_str)
            if path.exists():
                file_path = path
                break
        
        if file_path:
            _config = Config.from_file(file_path)
        else:
            _config = Config()
    
    return _config