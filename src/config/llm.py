"""LLM configuration."""


from pydantic import Field
from pydantic_settings import BaseSettings

from src.llms.types import LLMType


class LLMConfig(BaseSettings):
    """LLM configuration."""

    llm_name: LLMType = Field(
        default=LLMType.GEMINI,
        description="LLM backend type",
    )
    llm_config: dict | None = Field(
        default=None,
        description="Additional LLM-specific keyword arguments",
    )
