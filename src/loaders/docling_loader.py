import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
    TableFormerMode,
    TableStructureOptions
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption
)
from langchain.schema import Document

from .protocol import DocumentLoader

PageSpecifier = Union[Sequence[int], range, None]

DEFAULT_IMAGE_DESCRIPTION_PROMPT = (
    "You are an expert technical analyst converting visual data into text for a retrieval system. "
    "Analyze the image exhaustively. Do not summarize; extract details."
    "\n\n"
    "Follow these strict rules based on image type:"
    "\n"
    "1. **Charts & Graphs:**\n"
    "   - State the Title, X-axis label, and Y-axis label.\n"
    "   - Transcribe the specific data points or values visible for each category/timeframe.\n"
    "   - Explicitly state the trend (e.g., 'Rising from 10% to 50%').\n"
    "2. **Diagrams & Flowcharts:**\n"
    "   - Transcribe every text node/box in the image.\n"
    "   - Describe the relationships using logical flow (e.g., 'The process starts at [A], which splits into [B] and [C].').\n"
    "3. **Tables (as images):**\n"
    "   - Convert the image data into a Markdown table format.\n"
    "4. **Screenshots/UI:**\n"
    "   - List all visible menu items, buttons, and active fields.\n"
    "\n"
    "Output format: specific, dense, and factual. Avoid filler words."
)


class DoclingLoader(DocumentLoader):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        
        file_path = config.get("file_path")
        if not file_path:
            raise ValueError("'file_path' is required in loader configuration")

        self.doc_path = Path(file_path).expanduser().resolve()
        self.pdf_path = self.doc_path # Support for deprecated 'pdf_path' key
        if not self.doc_path.exists():
            raise FileNotFoundError(f"Doc not found: {self.doc_path}")

        output_dir = config.get("output_dir")
        self.output_dir = Path(output_dir).expanduser().resolve() if output_dir else None
        
        llm_api_key = self.config.get("llm_api_key") or os.getenv("LLM_API_KEY")
        llm_endpoint = self.config.get("llm_endpoint") or os.getenv("LLM_ENDPOINT")
        llm_model = self.config.get("llm_model") or os.getenv("LLM_MODEL")
        
        prompt = self.config.get("image_description_prompt", DEFAULT_IMAGE_DESCRIPTION_PROMPT)

        # Configure Pipeline Options
        pipeline_options = PdfPipelineOptions(
            enable_remote_services=True,
            do_table_structure=True,
            allow_external_plugins=True,
            do_ocr=self.config.get("do_ocr", False),
            do_picture_description=True,
            table_structure_options=TableStructureOptions(
                do_cell_matching=True,
                table_former_mode=TableFormerMode.ACCURATE
            ),
        )

        # Only configure picture description if credentials are available
        if llm_api_key and llm_endpoint and llm_model:
            pipeline_options.picture_description_options = PictureDescriptionApiOptions(
                url=llm_endpoint,
                headers={
                    "Authorization": "Bearer " + llm_api_key,
                    "Content-Type": "application/json",
                },
                prompt=prompt,
                params={
                    "model": llm_model
                }
            )

        self.converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.DOCX: WordFormatOption()
        })

    def _get_conversion_result(self):
        try:
            return self.converter.convert(str(self.doc_path))
        except Exception as e:
            raise RuntimeError(
                f"Docling conversion failed for {self.doc_path}: {e}"
            ) from e

    def load_documents(self) -> list[Document]:
        result = self._get_conversion_result()
        md_text = result.document.export_to_markdown()
        
        return [
            Document(
                page_content=md_text,
                metadata={
                    "source": str(self.doc_path),
                    "file_name": self.doc_path.name,
                    "loader": "DoclingLoader",
                    "file_type": self.doc_path.suffix.lower()
                }
            )
        ]

    def to_markdown_text(self, pages: PageSpecifier = None) -> str:
        result = self._get_conversion_result()
        return result.document.export_to_markdown()


def create_docling_loader(config: dict[str, Any]) -> DocumentLoader:
    return DoclingLoader(config=config)