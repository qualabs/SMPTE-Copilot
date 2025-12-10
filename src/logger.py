"""Logger utility class for application-wide logging configuration."""

import logging
import sys
from typing import Optional

class Logger:
    """Static logger utility class for configuring and accessing loggers."""
    
    @staticmethod
    def setup(config) -> None:
        logging.basicConfig(
            level=config.logging.get_level(),
            format="%(message)s",
            stream=sys.stdout,
            force=True,  # Override any existing configuration
        )
