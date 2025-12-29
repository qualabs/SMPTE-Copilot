import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
    TableFormerMode,
    TableStructureOptions,
    ConvertPipelineOptions
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption
)
from langchain.schema import Document

from src.constants import DEFAULT_IMAGE_DESCRIPTION_PROMPT

from .protocol import DocumentLoader

PageSpecifier = Union[Sequence[int], range, None]


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
        
        llm_api_key = self.config.get("llm_api_key")
        if llm_api_key is None:
            raise ValueError("LLM api key is required for docling")
        llm_endpoint = self.config.get("llm_endpoint")
        if llm_endpoint is None:
            raise ValueError("LLM endpoint is required for docling")

        llm_model = self.config.get("llm_model") or os.getenv("LLM_MODEL")
        if llm_model is None:
            raise ValueError("LLM model is required for docling")
        
        prompt = self.config.get("image_description_prompt", DEFAULT_IMAGE_DESCRIPTION_PROMPT)

        can_do_picture_description = (llm_api_key is not None and
                                       llm_endpoint is not None and
                                         llm_model is not None)
        # Configure Pipeline Options
        pdf_pipeline_options = PdfPipelineOptions(
            enable_remote_services=True,
            do_table_structure=True,
            allow_external_plugins=True,
            do_ocr=False,
            do_picture_description=can_do_picture_description,
            table_structure_options=TableStructureOptions(
                do_cell_matching=True,
                table_former_mode=TableFormerMode.ACCURATE
            ),
        )

        docx_pipeline_options = ConvertPipelineOptions(
            allow_external_plugins=True,
            enable_remote_services=True,
            do_picture_description=can_do_picture_description,
        )

        # Only configure picture description if credentials are available
        if can_do_picture_description:
            picture_description_options = PictureDescriptionApiOptions(
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
            pdf_pipeline_options.picture_description_options = picture_description_options
            docx_pipeline_options.picture_description_options = picture_description_options

        self.converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
            InputFormat.DOCX: WordFormatOption(pipeline_options=docx_pipeline_options),
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
