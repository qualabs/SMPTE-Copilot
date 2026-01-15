import logging
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ConvertPipelineOptions,
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from langchain_core.documents import Document

from src.constants import DEFAULT_IMAGE_DESCRIPTION_PROMPT, DEFAULT_IMAGE_DESCRIPTION_TIMEOUT

from .protocol import DocumentLoader

LLM_API_KEY = "llm_api_key"
LLM_ENDPOINT = "llm_endpoint"
LLM_MODEL = "llm_model"

class DoclingLoader(DocumentLoader):
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._logger = logging.getLogger(__name__)
        self._picture_description_enabled = self._config.get("picture_description_enabled", False)
        self._converter = self._create_converter()
        self.input_path = self._resolve_input_path()
        self.output_dir = self._resolve_output_dir()

    def _resolve_input_path(self) -> Path:
        file_path = self._config.get("file_path")
        if not file_path:
            raise ValueError("'file_path' is required in loader configuration")

        resolved_path = Path(file_path).expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Doc not found: {resolved_path}")

        return resolved_path

    def _resolve_output_dir(self) -> Path | None:
        output_dir = self._config.get("output_dir")
        return Path(output_dir).expanduser().resolve() if output_dir else None

    def _create_converter(self) -> DocumentConverter:
        self._validate_picture_description_config(self._picture_description_enabled)
        self._logger.info(f"Picture description: {'enabled' if self._picture_description_enabled else 'disabled'}")

        pdf_options = self._create_pdf_pipeline_options(self._picture_description_enabled)
        docx_options = self._create_docx_pipeline_options(self._picture_description_enabled)

        if self._picture_description_enabled:
            description_options = self._create_picture_description_options()
            pdf_options.picture_description_options = description_options
            docx_options.picture_description_options = description_options

        return DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
            InputFormat.DOCX: WordFormatOption(pipeline_options=docx_options),
        })

    def _validate_picture_description_config(self, enabled: bool) -> None:
        if not enabled:
            return

        missing = [field for field in (LLM_API_KEY, LLM_ENDPOINT, LLM_MODEL) if not self._config.get(field)]
        if missing:
            raise ValueError(
                f"picture_description_enabled=True but missing required fields: {', '.join(missing)}. "
                "Either provide all credentials or set picture_description_enabled=False."
            )

    def _create_pdf_pipeline_options(self, picture_description_enabled: bool) -> PdfPipelineOptions:
        return PdfPipelineOptions(
            enable_remote_services=True,
            do_table_structure=True,
            allow_external_plugins=True,
            do_ocr=False,
            do_picture_description=picture_description_enabled,
            table_structure_options=TableStructureOptions(
                do_cell_matching=True,
                table_former_mode=TableFormerMode.ACCURATE,
            ),
        )

    def _create_docx_pipeline_options(self, picture_description_enabled: bool) -> ConvertPipelineOptions:
        return ConvertPipelineOptions(
            allow_external_plugins=True,
            enable_remote_services=True,
            do_picture_description=picture_description_enabled,
            do_ocr=False,
        )

    def _create_picture_description_options(self) -> PictureDescriptionApiOptions:
        return PictureDescriptionApiOptions(
            url=self._config[LLM_ENDPOINT],
            headers={
                "Authorization": f"Bearer {self._config[LLM_API_KEY]}",
                "Content-Type": "application/json",
            },
            prompt=self._config.get("image_description_prompt", DEFAULT_IMAGE_DESCRIPTION_PROMPT),
            params={"model": self._config[LLM_MODEL]},
            timeout=self._config.get("image_description_timeout", DEFAULT_IMAGE_DESCRIPTION_TIMEOUT),
        )

    def _build_metadata(self) -> dict[str, str]:
        return {
            "source": str(self.input_path),
            "file_name": self.input_path.name,
            "loader": "DoclingLoader",
            "file_type": self.input_path.suffix.lower(),
        }

    def _has_failed_description(self, picture) -> bool:
        """Check if a picture has a description annotation but with empty text (API failure)."""
        if not picture.annotations:
            return False
        return any(
            getattr(ann, "kind", None) == "description" and not getattr(ann, "text", None)
            for ann in picture.annotations
        )

    def _check_picture_conversion_success(self, document) -> None:
        if not self._picture_description_enabled:
            return

        all_pictures = document.pictures
        failed_pictures = [p for p in all_pictures if self._has_failed_description(p)]

        if failed_pictures:
            raise RuntimeError(
                f"Picture descriptions: {len(failed_pictures)}/{len(all_pictures)} failed "
                f"(had 'description' annotation but empty text - likely API error)"
            )

        # Count pictures that were actually processed (have description annotation with text)
        described = sum(
            1 for p in all_pictures
            if any(
                getattr(ann, "kind", None) == "description" and getattr(ann, "text", None)
                for ann in (p.annotations or [])
            )
        )
        skipped = len(all_pictures) - described
        self._logger.info(f"Picture descriptions: {described} described, {skipped} skipped (decorative/small)")

    def load_documents(self) -> list[Document]:
        try:
            result = self._converter.convert(str(self.input_path))
        except Exception as e:
            raise RuntimeError(f"Docling conversion failed for {self.input_path}: {e}") from e

        if result.errors:
            error_messages = "; ".join(str(e) for e in result.errors)
            raise RuntimeError(f"Docling conversion had errors for {self.input_path}: {error_messages}")

        self._check_picture_conversion_success(result.document)

        return [Document(page_content=result.document.export_to_markdown(), metadata=self._build_metadata())]


def create_docling_loader(config: dict[str, Any]) -> DocumentLoader:
    return DoclingLoader(config=config)
