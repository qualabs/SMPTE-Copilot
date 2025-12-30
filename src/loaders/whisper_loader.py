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
        self.include_timestamps = config.get("include_timestamps", True)

        self._transcription_result_cache: Optional[dict[str, Any]] = None
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

    def _get_transcription_result(self) -> dict[str, Any]:
        """Transcribe the video/audio file using Whisper and return full result.

        Returns
        -------
        Full transcription result dictionary with segments and text.
        """
        if self._transcription_result_cache is not None:
            return self._transcription_result_cache

        try:
            self.logger.info(f"Transcribing audio from: {self.input_path}")
            model = self._load_model()

            transcribe_kwargs = {}
            if self.language:
                transcribe_kwargs["language"] = self.language

            result = model.transcribe(str(self.input_path), **transcribe_kwargs)
            
            if not result.get("text", "").strip():
                self.logger.warning(f"Empty transcription for {self.input_path}")

            self._transcription_result_cache = result
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to transcribe {self.input_path}: {e}") from e

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS.mmm.

        Parameters
        ----------
        seconds
            Time in seconds.

        Returns
        -------
        Formatted timestamp string.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def load_documents(self) -> list[Document]:
        """Load the video/audio file into LangChain Document objects.

        Returns
        -------
        List of Document objects representing the transcription.
        """
        result = self._get_transcription_result()
        transcription = result["text"].strip()

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
        """Return the transcription as Markdown text with timestamps.

        Parameters
        ----------
        pages
            Not used for audio/video files (kept for protocol compatibility).

        Returns
        -------
        Transcription as Markdown-formatted string with timestamps.
        """
        result = self._get_transcription_result()
        
        if not result.get("text", "").strip():
            return ""

        if not self.include_timestamps:
            return f"# Transcription\n\n{result['text'].strip()}"

        segments = result.get("segments", [])
        if not segments:
            return f"# Transcription\n\n{result['text'].strip()}"

        markdown_lines = ["# Transcription\n"]
        
        for segment in segments:
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "").strip()
            
            if not text:
                continue

            timestamp_str = f"[{self._format_timestamp(start_time)} - {self._format_timestamp(end_time)}]"
            markdown_lines.append(f"{timestamp_str}\n{text}\n")

        return markdown_lines

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
        - language (optional): Language code for transcription (default: "en")
        - include_timestamps (optional): Include timestamps in output (default: True)

    Returns
    -------
    DocumentLoader instance.
    """
    return WhisperLoader(config=config)

