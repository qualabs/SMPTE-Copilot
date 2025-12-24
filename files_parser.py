"""
File parser module for processing multiple document formats (PDF, DOCX, JSON/Whisper).
MODIFIED: Only accepts JSON for transcripts (ignores .txt to avoid duplicates).
"""

from pathlib import Path
import re
import json
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from typing import Tuple, Optional, List, Union

# ==========================================
# HELPER: Time Formatting
# ==========================================
def seconds_to_timestamp(seconds: float) -> str:
    """Converts raw seconds (14.7) to HH:MM:SS."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

# ==========================================
# HELPER: JSON Reconstruction (Preferred)
# ==========================================
def transform_whisper_json_to_markdown(json_path: Path) -> Optional[str]:
    """Reconstructs paragraphs from Whisper JSON with Inline Timestamps."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if "segments" not in data or "text" not in data:
            return None
            
        segments = data["segments"]
        markdown_output = [f"# Transcript: {json_path.stem.replace('_result', '')}\n"]
        
        current_paragraph_text = []
        current_start_time = None
        
        # Logic Constants
        MIN_PARAGRAPH_LEN = 500
        MAX_PARAGRAPH_LEN = 2000 

        for i, segment in enumerate(segments):
            text = segment.get("text", "").strip()
            start = segment.get("start", 0.0)
            
            if not text: continue
            
            if current_start_time is None:
                current_start_time = start
                
            current_paragraph_text.append(text)
            
            current_len = sum(len(s) for s in current_paragraph_text)
            has_terminal_punct = text[-1] in ['.', '?', '!'] if len(text) > 0 else False
            
            should_flush = (
                (current_len > MIN_PARAGRAPH_LEN and has_terminal_punct) or
                (current_len > MAX_PARAGRAPH_LEN) or
                (i == len(segments) - 1)
            )
            
            if should_flush:
                full_text = " ".join(current_paragraph_text)
                timestamp_str = seconds_to_timestamp(current_start_time)
                block = f"**(Time: {timestamp_str})** {full_text}\n"
                markdown_output.append(block)
                current_paragraph_text = []
                current_start_time = None

        return "\n".join(markdown_output)
    except Exception as e:
        print(f"⚠️ Error parsing whisper JSON: {e}")
        return None

# ==========================================
# CORE PARSER LOGIC
# ==========================================
def create_converter(allowed_formats: Optional[List[InputFormat]] = None) -> DocumentConverter:
    if allowed_formats is None:
        allowed_formats = [InputFormat.PDF, InputFormat.DOCX, InputFormat.MD]
    
    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_cls=StandardPdfPipeline, backend=PyPdfiumDocumentBackend),
        InputFormat.DOCX: WordFormatOption(pipeline_cls=SimplePipeline)
    }
    return DocumentConverter(allowed_formats=allowed_formats, format_options=format_options)

def parse_file(
    file_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    allowed_formats: Optional[List[InputFormat]] = None,
) -> Tuple[object, str, str]:
    file_path = Path(file_path)
    if output_dir is None: output_dir = Path("markdown")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    final_output_path = output_dir / f"{file_path.stem}.md"
    md_content = None

    # --- ONLY JSON LOGIC ENABLED FOR WHISPER ---
    if file_path.suffix.lower() == ".json":
        print(f"  > Detected .json file. Reconstructing...")
        md_content = transform_whisper_json_to_markdown(file_path)

    # Note: .txt logic removed to prevent duplicates as requested

    if md_content:
        with open(final_output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        converter = create_converter(allowed_formats=[InputFormat.MD])
        result = converter.convert(str(final_output_path))
        return result, md_content, str(final_output_path)

    # Standard PDF/DOCX
    converter = create_converter(allowed_formats=allowed_formats)
    result = converter.convert(str(file_path))
    markdown_content = result.document.export_to_markdown()
    
    with open(final_output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    return result, markdown_content, str(final_output_path)

def parse_directory(
    directory_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    allowed_formats: Optional[List[InputFormat]] = None,
    recursive: bool = False,
) -> List[Tuple[object, str, str]]:
    directory_path = Path(directory_path)
    
    if recursive:
        all_files = list(directory_path.rglob("*"))
    else:
        all_files = list(directory_path.glob("*"))
    
    # --- MODIFIED: REMOVED .txt FROM VALID EXTENSIONS ---
    valid_exts = {'.pdf', '.docx', '.md', '.json'} # No .txt
    
    files_to_process = [f for f in all_files if f.is_file() and f.suffix.lower() in valid_exts]
    
    print(f"Found {len(files_to_process)} file(s) to process (JSON/PDF only).")
    
    results = []
    for p in files_to_process:
        try:
            print(f"Processing: {p.name}")
            res = parse_file(p, output_dir, allowed_formats)
            if res:
                results.append(res)
                print("  ✓ Done")
            else:
                print("  ⚠️ Skipped (Invalid content)")
        except Exception as e:
            print(f"❌ Error processing {p.name}: {e}")
            
    return results