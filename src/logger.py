"""Logger utility class for application-wide logging configuration."""

import logging
import sys


class Logger:
    """Static logger utility class for configuring and accessing loggers."""

    @staticmethod
    def setup(config) -> None:
        logging.basicConfig(
            level=config.logging.get_level(),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
            force=True,  # Override any existing configuration
        )
