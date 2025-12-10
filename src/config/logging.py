"""Logging configuration."""

import logging
from pydantic import Field
from pydantic_settings import BaseSettings


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    
    def get_level(self) -> int:
        """Convert string level to logging constant.
        
        Returns
        -------
        int
            Logging level constant.
        """
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        level_upper = self.level.upper()
        if level_upper not in level_map:
            return logging.INFO  # Default fallback
        return level_map[level_upper]

