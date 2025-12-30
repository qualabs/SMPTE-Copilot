from __future__ import annotations

"""Whisper-based video/audio loader implementation."""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union

import whisper
from langchain.schema import Document

from ..constants import DEFAULT_ENCODING
from .protocol import DocumentLoader

PageSpecifier = Union[Sequence[int], range, None]


class WhisperLoader(DocumentLoader):
    """Load video/audio files using Whisper and convert to text/markdown.

    This loader uses OpenAI's Whisper model to transcribe audio/video files
    and converts the transcription to LangChain Document objects and Markdown.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the loader with a configuration dictionary.

        Parameters
        ----------
        config
            Configuration dictionary. Must contain 'file_path' key.
            Optional keys:
            - output_dir: Directory for output markdown files
            - model_name: Whisper model to use (default: "base")
            - device: Device to run on ("cpu" or "cuda", default: "cpu")
            - language: Language code (e.g., "en", "es"). If None, auto-detects.
            - Any other keys are stored and can be accessed via self.config

        Raises
        ------
        ValueError
            If 'file_path' is missing or the file is not a video/audio file.
        FileNotFoundError
            If the file does not exist.
        """
        self.logger = logging.getLogger(__name__)
        
        file_path = config.get("file_path")
        if not file_path:
            raise ValueError("'file_path' is required in loader configuration")

        self.input_path = Path(file_path).expanduser().resolve()
        if not self.input_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.input_path}")

        supported_extensions = {".mp4", ".mp3", ".wav", ".m4a", ".avi", ".mov", ".mkv"}
        if self.input_path.suffix.lower() not in supported_extensions:
            raise ValueError(
                f"Unsupported file type: {self.input_path.suffix}. "
                f"Supported: {', '.join(supported_extensions)}"
            )

        output_dir = config.get("output_dir")
        self.output_dir = Path(output_dir).expanduser().resolve() if output_dir else None

        self.model_name = config.get("model_name", "base")
        self.device = config.get("device", "cpu")
        self.language = config.get("language", "en")

        self._transcription_cache: Optional[str] = None
        self._model: Optional[whisper.Whisper] = None

    def _load_model(self) -> whisper.Whisper:
        """Load the Whisper model (lazy loading).

        Returns
        -------
        Loaded Whisper model instance.
        """
        if self._model is None:
            self.logger.info(f"Loading Whisper model: {self.model_name}")
            self._model = whisper.load_model(self.model_name, device=self.device)
        return self._model

    def _transcribe(self) -> str:
        """Transcribe the video/audio file using Whisper.

        Returns
        -------
        Transcribed text as a string.

        Raises
        ------
        RuntimeError
            If transcription fails.
        """
        if self._transcription_cache is not None:
            return self._transcription_cache

        try:
            self.logger.info(f"Transcribing audio from: {self.input_path}")
            model = self._load_model()

            transcribe_kwargs = {}
            if self.language:
                transcribe_kwargs["language"] = self.language

            result = model.transcribe(str(self.input_path), **transcribe_kwargs)
            transcription = result["text"].strip()

            if not transcription:
                self.logger.warning(f"Empty transcription for {self.input_path}")
                transcription = ""

            self._transcription_cache = transcription
            return transcription
        except Exception as e:
            raise RuntimeError(f"Failed to transcribe {self.input_path}: {e}") from e

    def load_documents(self) -> list[Document]:
        """Load the video/audio file into LangChain Document objects.

        Returns
        -------
        List of Document objects representing the transcription.

        Raises
        ------
        RuntimeError
            If transcription fails.
        """
        transcription = self._transcribe()

        return [
            Document(
                page_content=transcription,
                metadata={
                    "source": str(self.input_path),
                    "file_name": self.input_path.name,
                    "loader": "WhisperLoader",
                    "file_type": self.input_path.suffix.lower(),
                    "model": self.model_name,
                    "language": self.language or "auto-detected",
                }
            )
        ]

    def to_markdown_text(self, pages: PageSpecifier = None) -> str:
        """Return the transcription as Markdown text.

        Parameters
        ----------
        pages
            Not used for audio/video files (kept for protocol compatibility).

        Returns
        -------
        Transcription as Markdown-formatted string.

        Raises
        ------
        RuntimeError
            If transcription fails.
        """
        transcription = self._transcribe()

        if not transcription:
            return ""

        markdown = f"# Transcription\n\n{transcription}"
        return markdown

    def _resolve_output_path(self, output_path: Optional[Path]) -> Path:
        """Resolve the output path for the markdown file.

        Parameters
        ----------
        output_path
            Optional explicit output path. If None, generates a default path
            based on the video file name in the output directory.

        Returns
        -------
        Resolved output path.
        """
        if output_path is not None:
            return Path(output_path).expanduser().resolve()

        target_dir = self.output_dir or self.input_path.parent
        return target_dir / f"{self.input_path.stem}.md"


def create_whisper_loader(config: dict[str, Any]) -> DocumentLoader:
    """Create a Whisper loader from configuration.

    Parameters
    ----------
    config
        Configuration dictionary. Must contain:
        - file_path (required): Path to the video/audio file
        Optional keys:
        - output_dir (optional): Directory for output markdown files
        - model_name (optional): Whisper model name (default: "base")
        - device (optional): Device to run on (default: "cpu")
        - language (optional): Language code for transcription

    Returns
    -------
    DocumentLoader instance.

    Raises
    ------
    ValueError
        If 'file_path' is missing or the file is not a video/audio file.
    FileNotFoundError
        If the file does not exist.
    """
    return WhisperLoader(config=config)

