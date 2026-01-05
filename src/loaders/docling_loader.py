import os
import logging

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

from src.constants import DEFAULT_IMAGE_DESCRIPTION_PROMPT, DEFAULT_IMAGE_DESCRIPTION_TIMEOUT

from .protocol import DocumentLoader

PageSpecifier = Union[Sequence[int], range, None]

class DoclingLoader(DocumentLoader):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        
        file_path = self.config.get("file_path")
        if not file_path:
            raise ValueError("'file_path' is required in loader configuration")

        self.input_path = Path(file_path).expanduser().resolve()
        if not self.input_path.exists():
            raise FileNotFoundError(f"Doc not found: {self.input_path}")

        output_dir = self.config.get("output_dir")
        self.output_dir = Path(output_dir).expanduser().resolve() if output_dir else None
        
        prompt = self.config.get("image_description_prompt", DEFAULT_IMAGE_DESCRIPTION_PROMPT)

        llm_api_key = self.config.get("llm_api_key")
        llm_endpoint = self.config.get("llm_endpoint")
        llm_model = self.config.get("llm_model")
        
        can_do_picture_description = (llm_api_key is not None and
                                       llm_endpoint is not None and
                                         llm_model is not None)
        # Configure Pipeline Options
        pdf_pipeline_options = PdfPipelineOptions(
            enable_remote_services=True,
            do_table_structure=True,
            allow_external_plugins=True,
            do_ocr=not can_do_picture_description,
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
            do_ocr=not can_do_picture_description,
        )

        # Only configure picture description if credentials are available
        if can_do_picture_description:
            image_description_timeout = self.config.get("image_description_timeout", DEFAULT_IMAGE_DESCRIPTION_TIMEOUT)
            picture_description_options = PictureDescriptionApiOptions(
                url=llm_endpoint,
                headers={
                    "Authorization": "Bearer " + llm_api_key,
                    "Content-Type": "application/json",
                },
                prompt=prompt,
                params={
                    "model": llm_model
                },
                timeout=image_description_timeout
            )
            pdf_pipeline_options.picture_description_options = picture_description_options
            docx_pipeline_options.picture_description_options = picture_description_options

        self.converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
            InputFormat.DOCX: WordFormatOption(pipeline_options=docx_pipeline_options),
        })

    def _get_conversion_result(self):
        try:
            return self.converter.convert(str(self.input_path))
        except Exception as e:
            raise RuntimeError(
                f"Docling conversion failed for {self.input_path}: {e}"
            ) from e

    def load_documents(self) -> list[Document]:

        result = self._get_conversion_result()
        md_text = result.document.export_to_markdown()
        
        metadata = {
            "source": str(self.input_path),
            "file_name": self.input_path.name,
            "loader": "DoclingLoader",
            "file_type": self.input_path.suffix.lower()
        }
        
        if hasattr(result.document, "meta") and result.document.meta:
            doc_meta = result.document.meta
            if hasattr(doc_meta, "title") and doc_meta.title:
                metadata["title"] = doc_meta.title
            if hasattr(doc_meta, "author") and doc_meta.author:
                metadata["author"] = doc_meta.author
            if hasattr(doc_meta, "creation_date") and doc_meta.creation_date:
                metadata["creation_date"] = str(doc_meta.creation_date)
            if hasattr(doc_meta, "modification_date") and doc_meta.modification_date:
                metadata["modification_date"] = str(doc_meta.modification_date)
        
        return [
            Document(
                page_content=md_text,
                metadata=metadata
            )
        ]

    def to_markdown_text(self, pages: PageSpecifier = None) -> str:

        result = self._get_conversion_result()
        return result.document.export_to_markdown()


def create_docling_loader(config: dict[str, Any]) -> DocumentLoader:
    return DoclingLoader(config=config)
